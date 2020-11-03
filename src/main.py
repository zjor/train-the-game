import sys
import pygame
import numpy as np
import pygame as pg

import multiprocessing

from constants import Colors
from car import Car
from road import Road

from torch_cortex import TorchCortex, CortexWorker
from pg_utils import draw_matrix


class Game:
    COMMAND_LEFT = 0
    COMMAND_STRAIGHT = 1
    COMMAND_RIGHT = 2

    MODE_MANUAL = 0
    MODE_AUTOPILOT = 1

    def __init__(self):
        self.mode = Game.MODE_MANUAL

        self.font = pg.font.SysFont('verdanattf', 14)
        self.font_bold = pg.font.SysFont('verdanaboldttf', 14)

        self.training_data = []
        self.data_row = []

        self.size = self.width, self.height = 800, 600
        self.road_size = self.road_width, self.road_height = 400, 590

        self.screen = pg.display.set_mode(self.size)

        self.road_surface = pg.Surface(self.road_size)
        self.road = Road(self.road_width, self.road_height)

        self.car = Car(origin=(self.road_width / 2, self.road_height - 50), lidar_count=24)
        self.road_shift = 0.0

        self.left = int((self.width - self.road_width) / 2)
        self.top = int((self.height - self.road_height) / 2)

        # 0 - turn left, 1 - stay straight, 2 - turn right 
        self.last_command = Game.COMMAND_STRAIGHT
        self.paused = False
        self.display_lidars = True

        self.torch_cortex = TorchCortex()
        self.nn_slices = self.update_nn_slices()
        self.last_accuracy = None

        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.cortex_worker = CortexWorker(self.task_queue, self.result_queue)
        self.cortex_worker.start()

    def update_nn_slices(self):
        slices = []
        dx = 3
        for param in self.torch_cortex.model.parameters():
            weights = param.detach().numpy()
            if len(weights.shape) == 2:
                slices.append(draw_matrix(weights, dx))
        return slices

    def handle_keyboard(self):
        keys = pg.key.get_pressed()
        if keys[pg.K_LEFT]:
            self.car.turn_left()
            self.last_command = Game.COMMAND_LEFT
        elif keys[pg.K_RIGHT]:
            self.car.turn_right()
            self.last_command = Game.COMMAND_RIGHT
        elif keys[pg.K_t]:
            self.mode = Game.MODE_MANUAL
        elif keys[pg.K_a]:
            self.mode = Game.MODE_AUTOPILOT
        elif keys[pg.K_l]:
            self.display_lidars = not self.display_lidars
        elif keys[pg.K_p]:
            self.paused = not self.paused
        else:
            self.last_command = Game.COMMAND_STRAIGHT

    def draw_road(self):
        self.road_shift += self.car.get_vy()
        if self.road_shift > 1:
            for i in range(int(self.road_shift)):
                self.road.scroll()
            self.road_shift -= int(self.road_shift)

        self.road.draw(self.road_surface)

    def draw_road_surface(self):
        frame_width = 5

        pg.draw.rect(self.screen,
                     Colors.light_grey,
                     pg.Rect(
                         self.left - frame_width,
                         self.top - frame_width,
                         self.road_width + frame_width * 2,
                         self.road_height + frame_width * 2))
        self.screen.blit(self.road_surface, (self.left, self.top))

    def draw_lidars(self):
        self.data_row = []

        road_mask = self.road.get_mask()
        x, y = int(self.car.x), int(self.car.y)
        threshold = self.road.road_width
        readings = self.car.get_lidar_readings(road_mask, threshold)
        for r in readings:
            if self.display_lidars:
                pg.draw.line(self.road_surface, Colors.yellow, (x, y), r[0])
                pg.draw.circle(self.road_surface, Colors.red, r[0], 5)
            self.data_row.append(r[1] / threshold)

        if self.mode == Game.MODE_MANUAL:
            self.training_data.append(self.data_row + [self.last_command])

            # send training data to CortexWorker
            batch_size = 1000
            if len(self.training_data) % batch_size == 0:
                self.task_queue.put(np.matrix(self.training_data[-batch_size:]))

            # update model params if training completed
            if not self.result_queue.empty():
                state, acc = self.result_queue.get(False)
                self.torch_cortex.load_state(state)
                self.nn_slices = self.update_nn_slices()
                self.last_accuracy = acc
                print(f"Accuracy: {acc}")

    def detect_collision(self):
        car_surface = self.car.draw()
        car_width, car_height = car_surface.get_size()

        car_mask = self.car.get_mask()
        road_mask = self.road.get_mask()

        x, y = int(self.car.x), int(self.car.y)
        overlap = road_mask.overlap(car_mask, (x - car_width // 2, y - car_height // 2))
        if overlap:
            pg.draw.circle(self.road_surface, Colors.red, overlap, 10)

    def drive_or_collect_data(self):
        if len(self.training_data) == 0:
            return

        if self.mode == Game.MODE_MANUAL:
            pass
        else:
            self.last_command = self.torch_cortex.predict(self.data_row)
            if self.last_command == Game.COMMAND_LEFT:
                self.car.turn_left()
            elif self.last_command == Game.COMMAND_RIGHT:
                self.car.turn_right()
            else:
                pass  # going straight

    def draw_car(self):
        self.car.advance()
        self.car.draw(self.road_surface)

    def draw_text(self, position, font, text, color=(150, 250, 150)):
        surf = font.render(text, True, color)
        surf_rect = surf.get_rect().move(position)
        self.screen.blit(surf, surf_rect)
        return surf_rect.bottom

    def draw_info(self):
        lidar_readings = np.array(self.data_row).reshape(-1, 4)
        y = self.draw_text((5, 5), self.font_bold, "Lidar readings")
        for row in lidar_readings:
            text = " ".join(list(map(lambda v: f"{v:.2f}", row)))
            y = self.draw_text((5, y), self.font, text, color=(100, 200, 100))

        commands = ["LEFT", "STRAIGHT", "RIGHT"]
        y = self.draw_text((5, y + 5), self.font, f"Command: {commands[self.last_command]}")

        modes = ["Manual", "Self-driving"]
        y = self.draw_text((5, y + 5), self.font, f"Mode: {modes[self.mode]}")

        y = self.draw_text((5, y + 5), self.font, f"Training samples: {len(self.training_data)}")

        if self.last_accuracy:
            y = self.draw_text((5, y + 5), self.font, f"Accuracy: {self.last_accuracy:.3f}")

        y = self.draw_text((5, y + 5), self.font_bold, "Legend")
        y = self.draw_text((5, y + 5), self.font, "t - manual driving")
        y = self.draw_text((5, y + 5), self.font, "a - self-driving")
        y = self.draw_text((5, y + 5), self.font, "l - toggle lidar display")
        y = self.draw_text((5, y + 5), self.font, "p - toggle pause")
        return y

    def draw_nn_slices(self):
        x = self.left + self.road_width + 10
        y = self.draw_text((x, 5), self.font_bold, "NN weights")
        for nn_slice in self.nn_slices:
            rect = nn_slice.get_rect()
            self.screen.blit(nn_slice, rect.move(x, y + 5))
            y += rect.height + 5

    def loop(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        self.handle_keyboard()
        if not self.paused:
            self.screen.fill(Colors.black)
            self.draw_road()
            self.draw_lidars()
            self.draw_car()
            self.detect_collision()
            self.draw_road_surface()
            self.drive_or_collect_data()
            self.draw_info()
            self.draw_nn_slices()

        pg.display.flip()


def main():
    np.random.seed(77)

    pg.init()
    pg.display.set_caption("Train the Game")
    clock = pg.time.Clock()
    game = Game()

    while True:
        clock.tick(45)
        game.loop()


if __name__ == "__main__":
    main()

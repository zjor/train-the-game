import sys
import math
import pygame
import random

import numpy as np
import pygame as pg

import multiprocessing

from math import sin, cos, pi

from constants import Colors
from car import Car
from road import Road

from torch_cortex import TorchCortex, CortexWorker


class Game:
    COMMAND_LEFT = 0
    COMMAND_STRAIGHT = 1    
    COMMAND_RIGHT = 2

    MODE_COLLECT_DATA = 0
    MODE_AUTOPILOT = 1

    def __init__(self, mode=MODE_COLLECT_DATA):
        self.mode = mode

        self.font = pygame.font.Font('freesansbold.ttf', 18)

        self.training_data_filename = "training_data.txt"
        self.training_data = []
        self.recording_training_data = True
        self.data_row = []

        self.size = self.width, self.height = 800, 600  
        self.road_size = self.road_width, self.road_height = 400, 400

        self.screen = pg.display.set_mode(self.size)

        pg.display.set_caption("Train the Game")

        self.road_surface = pg.Surface(self.road_size)
        self.road = Road(self.road_width, self.road_height)

        self.car = Car(origin=(self.road_width / 2, self.road_height - 50), lidar_count=24)
        self.road_shift = 0.0

        self.left = int((self.width - self.road_width) / 2)
        self.top = int((self.height - self.road_height) / 2)
        
        # 0 - turn left, 1 - stay straight, 2 - turn right 
        self.last_command = Game.COMMAND_STRAIGHT
        self.paused = False     

        self.torch_cortex = TorchCortex()

        if mode == Game.MODE_AUTOPILOT:
            self.torch_cortex.load("model.pt")

        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.cortex_worker = CortexWorker(self.task_queue, self.result_queue)
        self.cortex_worker.start()


    def dump_training_data(self):
        with open(self.training_data_filename, "w") as f:
            for line in self.training_data:
                f.write(" ".join(map(str, line)))
                f.write("\n")


    def handle_keyboard(self):
        keys = pg.key.get_pressed()
        if keys[pg.K_LEFT]:
            self.car.turn_left()
            self.last_command = Game.COMMAND_LEFT
        elif keys[pg.K_RIGHT]:
            self.car.turn_right()
            self.last_command = Game.COMMAND_RIGHT
        else:

            if keys[pg.K_t]:
                # self.paused = True
                # self.dump_training_data()
                self.mode = Game.MODE_COLLECT_DATA
            elif keys[pg.K_a]:
                self.mode = Game.MODE_AUTOPILOT
            elif keys[pg.K_r]:
                self.recording_training_data = False
            elif keys[pg.K_e]:
                self.recording_training_data = True

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
            pygame.Rect(
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
            pg.draw.line(self.road_surface, Colors.yellow, (x, y), r[0])
            pg.draw.circle(self.road_surface, Colors.red, r[0], 5)
            self.data_row.append(r[1] / threshold)
        
        if self.mode == Game.MODE_COLLECT_DATA:
            if self.recording_training_data:
                self.training_data.append(self.data_row + [self.last_command])

            batch_size = 500
            if len(self.training_data) % batch_size == 0:
                self.task_queue.put(np.matrix(self.training_data[-batch_size:]))

            if not self.result_queue.empty():
                state, acc = self.result_queue.get(False)
                self.torch_cortex.load_state(state)
                print(f"Accuracy: {acc}")                


    def detect_collision(self):
        car_surface = self.car.draw()
        car_width, car_height = car_surface.get_size()

        car_mask = self.car.get_mask()
        road_mask = self.road.get_mask()

        # road_mask.to_surface(self.screen)

        x, y = int(self.car.x), int(self.car.y)
        # car_mask.to_surface(self.screen, dest=(x, y))
        overlap = road_mask.overlap(car_mask, (x - car_width // 2, y - car_height // 2))
        if overlap:
            pg.draw.circle(self.road_surface, Colors.red, overlap, 10)      

        
    def drive_or_collect_data(self):
        if len(self.training_data) == 0:
            return

        if self.mode == Game.MODE_COLLECT_DATA:
            pass                
        else:
            command = self.torch_cortex.predict(self.data_row)
            if command == Game.COMMAND_LEFT:
                self.car.turn_left()
            elif command == Game.COMMAND_RIGHT:
                self.car.turn_right()
            else:
                pass # going straight


    def draw_car(self):
        self.car.advance()
        car_surface = self.car.draw(self.road_surface)


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


        # printing lidar readings
        lidar_readings = self.training_data[-1][:-1]
        y = 0
        for i, v in enumerate(lidar_readings):
            text = self.font.render(f"{i}: {v:.3f}", True, (250, 50, 50))
            text_rect = text.get_rect()
            y = 10 + i * text_rect.height
            self.screen.blit(text, text_rect.move(10, y))

        text = self.font.render(f"|training_data| = {len(self.training_data)}", False, (150, 250, 150))
        text_rect = text.get_rect()
        y += text_rect.height
        self.screen.blit(text, text_rect.move(10, y))

        text = self.font.render(f"{'rec' if self.recording_training_data else 'not rec'}", False, (250, 150, 150))
        text_rect = text.get_rect()
        y += text_rect.height
        self.screen.blit(text, text_rect.move(10, y))
        

        pygame.display.flip()


def main():
    np.random.seed(77)

    pygame.init()
    clock = pygame.time.Clock()
    game = Game(mode=Game.MODE_COLLECT_DATA)
    # game = Game(mode=Game.MODE_AUTOPILOT)

    while True:
        clock.tick(45)
        game.loop()     


if __name__ == "__main__":
    main()
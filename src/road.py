import math
import pygame as pg
import numpy as np
from math import sin, cos, pi

from constants import Colors


class Road:
    def __init__(self, width, height, layer_height=100):
        self.width = width
        self.height = height
        self.layer_height = layer_height

        self.surface = pg.Surface((width, height + layer_height))

        self.traffic_line_surface = pg.Surface((width, height + layer_height))
        self.traffic_line_surface.set_colorkey(Colors.black)
        
        self.offset = layer_height
        self.road_width = int(3 * width / 5)
        self.t = 0.0

        self.dx = 5
        # road curve x
        self.x = [0] * (height + layer_height)

        self.k = np.random.rand(3) / 2
        self.d = np.random.rand(3) / 2
        self.p = np.random.rand(3)

        self.generate_layer(True)       


    def generate_curvature(self, t):
        w = 2.0 * pi * t / 20.0
        return np.sin(w * self.d + self.p).dot(self.k)


    def generate_layer(self, full_height=False):
        max_height = (self.layer_height + self.height) if full_height else self.layer_height
        dx = self.dx
        n = max_height // dx

        ys = [0] * n * dx

        for i in range(n):
            f = self.generate_curvature(self.t)
            self.t += 1

            cx = int((f * (self.road_width / 2) + self.width) / 2)
            self.x[(n - i - 1) * dx: (n - i)*dx] = [cx] * dx

            y = (n - i - 1) * dx
            ys[y] = cx
            left = int(cx - self.road_width / 2)
            right = int(cx + self.road_width / 2)

            pg.draw.rect(self.surface, Colors.grey, pg.Rect(0, y, self.width, dx))
            pg.draw.rect(self.surface, Colors.green, pg.Rect(0, y, left, dx))
            pg.draw.rect(self.surface, Colors.green, pg.Rect(right, y, self.width - right, dx))

            pg.draw.rect(self.surface, 
                Colors.red if int(i / 2) % 2 == 0 else Colors.light_grey, 
                pg.Rect(right, y, dx, dx))
            pg.draw.rect(self.surface, 
                Colors.red if int(i / 2) % 2 == 0 else Colors.light_grey, 
                pg.Rect(left - dx, y, dx, dx))

            if i % 5 != 0:
                pg.draw.rect(self.traffic_line_surface, Colors.white, pg.Rect(cx, y, dx, dx))

        p = 0.3
        should_draw_obstacle = np.random.rand() < p
        if should_draw_obstacle:
            y = self.layer_height // 2
            cx = ys[y]
            left = int(cx - self.road_width / 2)
            right = int(cx + self.road_width / 2)
            ox = (left + cx) // 2 if np.random.rand() > 0.5 else (right + cx) // 2
            self.draw_obstacle((ox, y), self.road_width * 3 // 5 // 2)        

        self.offset = self.layer_height


    def draw_obstacle(self, position, width, height=50):
        surf = pg.Surface((width, height))
        surf.fill(Colors.red)
        surf = pg.transform.rotate(surf, np.random.randint(-45, 45))
        rect = surf.get_rect()
        self.surface.blit(surf, rect.move(position).move(-rect.centerx, -rect.centery))


    def _scroll(self, surface):
        buf = surface.copy()
        surface.fill(Colors.black)
        surface.blit(buf, (0, 1))


    def scroll(self):
        self._scroll(self.surface)
        self._scroll(self.traffic_line_surface)

        self.x = [0] + self.x[:-1]

        self.offset -= 1
        if self.offset <= 0:
            self.generate_layer()


    def draw(self, screen):
        screen.blit(self.surface, (0, -self.layer_height))
        copy = screen.copy()
        screen.blit(self.traffic_line_surface, (0, -self.layer_height))
        return copy

    
    def get_mask(self):
        mask_surface = pg.Surface((self.width, self.height))
        mask_surface.blit(self.surface, (0, -self.layer_height))
        mask_surface.set_colorkey(Colors.grey)
        return pg.mask.from_surface(mask_surface)



if __name__ == "__main__":
    import sys
    from car import Car

    np.random.seed(101)

    pg.init()
    pg.display.set_caption("Road Test")
    size = width, height = 800, 600
    screen = pg.display.set_mode(size)

    road = Road(width, height)
    road.draw_obstacle((200, 400), 50)
    road.draw(screen)
    mask = road.get_mask()
    # mask.to_surface(screen)

    origin = (width // 2, height - 150)

    car = Car(origin=origin)
    readings = car.get_lidar_readings(mask, 400)
    for r in readings:
        pg.draw.line(screen, Colors.yellow, origin, r[0])
        pg.draw.circle(screen, Colors.red, r[0], 10)

    car.draw(screen)

    pg.display.flip()

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit()

    

import math
import pygame as pg
import numpy as np
from math import sin, cos, pi

from constants import Colors

class Ray:
    def __init__(self, origin=(.0, .0), theta=.0):
        self.origin = origin
        self.theta = theta


    def get_collision(self, mask, threshold):
        d = 0
        x, y = self.origin
        t_x, t_y = x, y
        bounds = mask.get_rect()

        while d < threshold and bounds.collidepoint(t_x, t_y) and mask.get_at((int(t_x), int(t_y))) == 0:
            t_x, t_y = x + d * cos(self.theta), y - d * sin(self.theta)            
            d += 1
            
        return ((int(t_x), int(t_y)), d)


class Car:
    def __init__(self, origin, velocity=2.0, lidar_count=6, angle_margin=pi/12):
        self.width = 30
        self.height = 50
        self.x, self.y = origin
        self.velocity = velocity
        self.angle = 0
        self.lidars = np.linspace(angle_margin, np.pi - angle_margin, lidar_count)
        self.sprite = pg.image.load("./images/car.bmp")


    def get_lidar_readings(self, road_mask, threshold):
        readings = []
        for i, theta in enumerate(self.lidars):
            ray = Ray(origin=(self.x, self.y), theta=(theta + math.radians(self.angle)))
            readings.append(ray.get_collision(road_mask, threshold))

        return readings


    def draw(self, surface=None):
        s = pg.transform.rotate(self.sprite.copy(), self.angle)
        
        car_rect = s.get_rect()

        if surface:
            surface.blit(s, (self.x - car_rect.centerx, self.y - car_rect.centery))
        return s


    def advance(self):
        self.x -= self.velocity * sin(math.radians(self.angle))


    def get_vy(self):
        return self.velocity * cos(math.radians(self.angle))


    def get_mask(self):
        return pg.mask.from_surface(self.draw())


    def turn_left(self):
        self.angle += 1


    def turn_right(self):
        self.angle -= 1


if __name__ == "__main__":
    import sys  
    pg.init()
    pg.display.set_caption("Car Test")
    size = width, height = 800, 600
    screen = pg.display.set_mode(size)

    car = Car((100, 100))
    car.angle = 0
    car.draw(screen)

    pg.display.flip()

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit()


import math
import pygame as pg
import numpy as np
from math import sin, cos, pi

from constants import Colors
from ray import Ray


class Car:
	def __init__(self, origin, velocity=2.0, lidar_count=6):
		self.width = 30
		self.height = 50
		self.x, self.y = origin
		self.velocity = velocity
		self.angle = 0
		self.lidars = np.linspace(.0, np.pi, lidar_count)


	def get_lidar_readings(self, road_mask, threshold):
		readings = []
		for theta in self.lidars:
			ray = Ray(origin=(self.x, self.y), theta=(theta + math.radians(self.angle)))
			readings.append(ray.get_collision(road_mask, threshold))

		return readings


	def draw(self, surface=None):		
		s = pg.Surface((self.width, self.height))
		s.fill(Colors.white)
		pg.draw.rect(s, Colors.blue, pg.Rect(0, 0, self.width, self.height))
		
		pg.draw.rect(s, Colors.yellow, pg.Rect(0, 0, 10, 10))
		pg.draw.rect(s, Colors.yellow, pg.Rect(self.width - 10, 0, 10, 10))
		
		pg.draw.rect(s, Colors.red, pg.Rect(0, self.height - 10, 5, 10))
		pg.draw.rect(s, Colors.red, pg.Rect(self.width - 5, self.height - 10, 5, 10))

		s = pg.transform.rotate(s, self.angle)
		
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

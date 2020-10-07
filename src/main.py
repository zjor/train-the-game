import sys
import math
import pygame
import random

import pygame as pg

from math import sin, cos, pi

from constants import Colors
from car import Car
from road import Road

class Game:
	def __init__(self):
		self.size = self.width, self.height = 800, 600	
		self.road_size = self.road_width, self.road_height = 400, 400

		self.screen = pygame.display.set_mode(self.size)

		pygame.display.set_caption("Train the Game")

		self.road_surface = pygame.Surface(self.road_size)
		self.road = Road(self.road_width, self.road_height)

		self.car = Car(self.road_width)
		self.road_shift = 0.0

		self.left = int((self.width - self.road_width) / 2)
		self.top = int((self.height - self.road_height) / 2)



	def handle_keyboard(self):
		keys = pygame.key.get_pressed()
		if keys[pygame.K_LEFT]:
			self.car.turn_left()
		elif keys[pygame.K_RIGHT]:
			self.car.turn_right()


	def draw_road(self):
		self.road_shift += self.car.vy
		if self.road_shift > 1:
			for i in range(int(self.road_shift)):
				self.road.scroll()
			self.road_shift -= int(self.road_shift)

		self.road.draw(self.road_surface)

		frame_width = 5

		pygame.draw.rect(self.screen, 
			Colors.light_grey, 
			pygame.Rect(
				self.left - frame_width, 
				self.top - frame_width, 
				self.road_width + frame_width * 2, 
				self.road_height + frame_width * 2))

		self.screen.blit(self.road_surface, (self.left, self.top))


	def detect_collision(self):
		car_surface = self.car.draw()
		car_width, car_height = car_surface.get_size()

		car_mask = pg.mask.from_surface(self.car.draw())
		road_surface = self.road_surface.copy()
		road_surface.set_colorkey(Colors.grey)
		road_mask = pg.mask.from_surface(road_surface)

		road_mask.to_surface(self.screen)
		x = int(self.car.x - car_width / 2)
		y = int(self.road_height - 1.5 * car_height)
		car_mask.to_surface(self.screen, dest=(x, y))
		overlap = road_mask.overlap(car_mask, (x, y))

		if overlap:
			pg.draw.circle(self.screen, Colors.red, (overlap[0] + self.left, overlap[1] + self.top), 10)


	def draw_car(self):
		car_surface = self.car.draw()
		w, h = car_surface.get_size()
		x = int((self.width - self.road_width) / 2 + self.car.x - w / 2)
		y = int((self.height + self.road_height) / 2 - h * 1.5)

		self.screen.blit(car_surface, (x, y))


	def loop(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				sys.exit()

		self.handle_keyboard()
		self.draw_road()
		self.draw_car()
		self.detect_collision()
		pygame.display.flip()


def main():
	pygame.init()
	clock = pygame.time.Clock()
	game = Game()

	while True:
		clock.tick(45)
		game.loop()		


if __name__ == "__main__":
	main()
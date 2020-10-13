import sys
import math
import pygame
import random

import numpy as np
import pygame as pg

from math import sin, cos, pi

from constants import Colors
from car import Car
from road import Road

from cortex import Cortex
from torch_cortex import TorchCortex


class Game:
	COMMAND_STRAIGHT = 0
	COMMAND_LEFT = -1
	COMMAND_RIGHT = 1

	MODE_COLLECT_DATA = 0
	MODE_AUTOPILOT = 1

	def __init__(self, mode=MODE_COLLECT_DATA):
		self.mode = mode

		self.training_data_filename = "training_data.txt"
		self.training_data = []

		self.size = self.width, self.height = 800, 600	
		self.road_size = self.road_width, self.road_height = 400, 400

		self.screen = pg.display.set_mode(self.size)

		pg.display.set_caption("Train the Game")

		self.road_surface = pg.Surface(self.road_size)
		self.pure_road_surface = pg.Surface(self.road_size)
		self.road = Road(self.road_width, self.road_height)

		self.car = Car(self.road_width)
		self.road_shift = 0.0

		self.left = int((self.width - self.road_width) / 2)
		self.top = int((self.height - self.road_height) / 2)
		
		# -1 - turn left, 0 - stay straight, 1 - turn right 
		self.last_command = Game.COMMAND_STRAIGHT
		self.paused = False

		self.cortex = Cortex()
		self.torch_cortex = TorchCortex()

		if mode == Game.MODE_AUTOPILOT:
			self.cortex.load("model.dump")
			self.torch_cortex.load("torch_model.dump")


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
				self.paused = True
				self.dump_training_data()

			self.last_command = Game.COMMAND_STRAIGHT


	def draw_road(self):
		self.road_shift += self.car.vy
		if self.road_shift > 1:
			for i in range(int(self.road_shift)):
				self.road.scroll()
			self.road_shift -= int(self.road_shift)

		self.pure_road_surface = self.road.draw(self.road_surface)

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
		road_surface = self.pure_road_surface.copy()
		road_surface.set_colorkey(Colors.grey)
		road_mask = pg.mask.from_surface(road_surface)

		# road_mask.to_surface(self.screen)
		x = int(self.car.x - car_width / 2)
		y = int(self.road_height - 1.5 * car_height)
		# car_mask.to_surface(self.screen, dest=(x, y))
		overlap = road_mask.overlap(car_mask, (x, y))

		if overlap:
			pg.draw.circle(self.screen, Colors.red, (overlap[0] + self.left, overlap[1] + self.top), 10)

		data_row = [self.road.x[y - car_height], self.road.x[y], self.car.x]
		if self.mode == Game.MODE_COLLECT_DATA:
				self.training_data.append(data_row + [self.car.angle])
		else:
			angle = self.car.angle
			desired_angle = self.torch_cortex.predict(data_row)
			# desired_angle = self.cortex.predict([self.road.x[y], self.car.x])
			if angle > desired_angle:
				self.car.turn_right()
			elif angle < desired_angle:
				self.car.turn_left()


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
		if not self.paused:
			self.draw_road()
			self.draw_car()
			self.detect_collision()
		pygame.display.flip()


def main():
	pygame.init()
	clock = pygame.time.Clock()
	game = Game(mode=Game.MODE_AUTOPILOT)

	while True:
		clock.tick(45)
		game.loop()		


if __name__ == "__main__":
	main()
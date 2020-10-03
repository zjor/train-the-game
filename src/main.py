import sys
import math
import pygame
import random

from math import sin, cos, pi

class Colors:	
	white = 255, 255, 255
	black = 0, 0, 0
	green = 50, 150, 50
	grey = 50, 50, 50


class Road:
	def __init__(self, width, height, layer_height=100):
		self.width = width
		self.height = height
		self.layer_height = layer_height

		self.surface = pygame.Surface((width, height + layer_height))
		self.offset = layer_height
		self.road_width = int(3 * width / 5)
		self.t = 0.0

		self.generate_layer(True)


	def generate_layer(self, full_height=False):
		max_height = (self.layer_height + self.height) if full_height else self.layer_height
		dx = 5
		for i in range(int(max_height / dx)):
			w = 2.0 * pi * self.t / 20.0
			f = (0.2 * sin(w / 3) + 0.5 * cos(w / 5) + 0.5 * sin(w / 7))
			self.t += 1

			cx = int((f * (self.road_width / 2) + self.width) / 2)		
			y = (int(max_height / dx) - i - 1) * dx
			left = int(cx - self.road_width / 2)
			right = int(cx + self.road_width / 2)

			pygame.draw.rect(self.surface, Colors.grey, pygame.Rect(0, y, self.width, dx))
			pygame.draw.rect(self.surface, Colors.green, pygame.Rect(0, y, left, dx))
			pygame.draw.rect(self.surface, Colors.green, pygame.Rect(right, y, self.width - right, dx))

			if i % 5 != 0:
				pygame.draw.rect(self.surface, Colors.white, pygame.Rect(cx, y, dx, dx))

		self.offset = self.layer_height


	def scroll(self, velocity=1):
		buf = self.surface.copy()
		self.surface.fill(Colors.black)
		self.surface.blit(buf, (0, velocity))
		self.offset -= velocity
		if self.offset <= 0:
			self.generate_layer()


	def draw(self, screen):
		screen.blit(self.surface, (0, -self.layer_height))



def main():
	pygame.init()
	clock = pygame.time.Clock()

	size = width, height = 800, 600	
	black = (0, 0, 0)

	screen = pygame.display.set_mode(size)

	pygame.display.set_caption("Hello PyGame")

	road = Road(int(width / 2), height, 100)
	road.generate_layer()


	while True:
		clock.tick(60)

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				sys.exit()

		road.scroll()
		road.draw(screen)

		pygame.display.flip()


if __name__ == "__main__":
	main()
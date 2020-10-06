import pygame

from math import sin, cos, pi

from constants import Colors


class Road:
	def __init__(self, width, height, layer_height=100):
		self.width = width
		self.height = height
		self.layer_height = layer_height

		self.surface = pygame.Surface((width, height + layer_height))
		self.offset = 1.0 * layer_height
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

			pygame.draw.rect(self.surface, 
				Colors.red if int(i / 2) % 2 == 0 else Colors.light_grey, 
				pygame.Rect(right, y, dx, dx))
			pygame.draw.rect(self.surface, 
				Colors.red if int(i / 2) % 2 == 0 else Colors.light_grey, 
				pygame.Rect(left - dx, y, dx, dx))

			if i % 5 != 0:
				pygame.draw.rect(self.surface, Colors.white, pygame.Rect(cx, y, dx, dx))

		self.offset = 1.0 * self.layer_height


	def scroll(self, velocity=1.0):
		buf = self.surface.copy()
		self.surface.fill(Colors.black)
		self.surface.blit(buf, (0, int(velocity)))
		self.offset -= velocity
		if self.offset <= 0:
			self.generate_layer()


	def draw(self, screen):
		screen.blit(self.surface, (0, -self.layer_height))

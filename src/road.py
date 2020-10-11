import pygame
import pygame as pg

from math import sin, cos, pi

from constants import Colors


class Road:
	def __init__(self, width, height, layer_height=100):
		self.width = width
		self.height = height
		self.layer_height = layer_height

		self.surface = pygame.Surface((width, height + layer_height))

		self.traffic_line_surface = pygame.Surface((width, height + layer_height))
		self.traffic_line_surface.set_colorkey(Colors.black)
		
		self.offset = layer_height
		self.road_width = int(3 * width / 5)
		self.t = 0.0

		self.dx = 5
		# road curve x
		self.x = [0] * (height + layer_height)

		self.generate_layer(True)		


	def generate_layer(self, full_height=False):
		max_height = (self.layer_height + self.height) if full_height else self.layer_height
		dx = self.dx
		n = max_height // dx
		for i in range(n):
			w = 2.0 * pi * self.t / 20.0
			f = (0.2 * sin(w / 3) + 0.5 * cos(w / 5) + 0.5 * sin(w / 7))
			self.t += 1

			cx = int((f * (self.road_width / 2) + self.width) / 2)
			self.x[(n - i - 1) * dx: (n - i)*dx] = [cx] * dx

			y = (n - i - 1) * dx
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
				pygame.draw.rect(self.traffic_line_surface, Colors.white, pygame.Rect(cx, y, dx, dx))

		self.offset = self.layer_height


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

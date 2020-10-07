import math
import pygame
from math import sin, cos, pi

from constants import Colors


class Car:
	def __init__(self, screen_width):
		self.width = 30
		self.height = 50
		self.angle = 0
		self.x = 1.0 * screen_width / 2
		self.velocity = 2.5
		self.vx = 0.0
		self.vy = 1.0 * self.velocity

	def draw(self):		
		s = pygame.Surface((self.width, self.height))
		s.fill(Colors.white)
		pygame.draw.rect(s, Colors.blue, pygame.Rect(0, 0, self.width, self.height))
		
		pygame.draw.rect(s, Colors.yellow, pygame.Rect(0, 0, 10, 10))
		pygame.draw.rect(s, Colors.yellow, pygame.Rect(self.width - 10, 0, 10, 10))
		
		pygame.draw.rect(s, Colors.red, pygame.Rect(0, self.height - 10, 5, 10))
		pygame.draw.rect(s, Colors.red, pygame.Rect(self.width - 5, self.height - 10, 5, 10))

		s = pygame.transform.rotate(s, self.angle)

		self.x += self.vx

		return s


	def update_velocity(self):
		rad = math.radians(self.angle)
		self.vx = -1.0 * self.velocity * sin(rad)
		self.vy = 1.0 * self.velocity * cos(rad)		


	def turn_left(self):
		self.angle += 1
		self.update_velocity()


	def turn_right(self):
		self.angle -= 1
		self.update_velocity()

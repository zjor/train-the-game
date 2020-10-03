import sys
import pygame
import random


class Road:

	black = 0, 0, 0

	def __init__(self, width, height, layer_height=100):
		self.width = width
		self.height = height
		self.layer_height = layer_height

		self.surface = pygame.Surface((width, height + layer_height))
		self.offset = layer_height


	def generate_layer(self):
		color = (random.randint(50, 250), random.randint(50, 250), random.randint(50, 250))
		left_top = (random.randint(0, 150), random.randint(0, 50))
		pygame.draw.rect(self.surface, color, pygame.Rect(left_top, (25, 25)))
		self.offset = self.layer_height


	def scroll(self, velocity=1):
		buf = self.surface.copy()
		self.surface.fill(Road.black)
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

	road = Road(width, height)
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
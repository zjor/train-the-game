import sys
import math
import pygame
import random

from math import sin, cos, pi

from constants import Colors
from road import Road

def main():
	pygame.init()
	clock = pygame.time.Clock()

	size = width, height = 800, 600	
	road_size = road_width, road_height = 400, 400

	screen = pygame.display.set_mode(size)

	pygame.display.set_caption("Train the Game")


	road_surface = pygame.Surface(road_size)
	road = Road(road_width, road_height)

	while True:
		clock.tick(60)

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				sys.exit()

		road.scroll()
		road.draw(road_surface)

		frame_width = 5
		left = int((width - road_width) / 2)
		top = int((height - road_height) / 2)

		pygame.draw.rect(screen, 
			Colors.light_grey, 
			pygame.Rect(
				left - frame_width, 
				top - frame_width, 
				road_width + frame_width * 2, 
				road_height + frame_width * 2))

		screen.blit(road_surface, (left, top))

		pygame.display.flip()


if __name__ == "__main__":
	main()
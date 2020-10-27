import pygame as pg


def draw_matrix(dest_surf, matrix, dx, top_left=(0, 0)):
    rows, cols = matrix.shape
    surf = pg.Surface((cols * dx, rows * dx))

    max_val = matrix.max()
    min_val = matrix.min()


    for i in range(rows):
        for j in range(cols):
            color = int(255.0 * (matrix[i, j] - min_val) / (max_val - min_val))

            pg.draw.rect(surf, (color, color, color), pg.Rect(j * dx, i * dx, dx, dx))

    dest_surf.blit(surf, surf.get_rect().move(top_left))


if __name__ == "__main__":
    import sys      
    import numpy as np
    matrix = np.random.randn(15, 15)
    pg.init()
    pg.display.set_caption("Car Test")
    size = width, height = 800, 600
    screen = pg.display.set_mode(size)

    draw_matrix(screen, matrix, 20, top_left=(100, 100))

    pg.display.flip()

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit()  

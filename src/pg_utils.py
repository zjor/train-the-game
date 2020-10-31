import pygame as pg


def draw_matrix(matrix, dx):
    rows, cols = matrix.shape
    surf = pg.Surface((cols * dx, rows * dx))

    max_val = matrix.max()
    min_val = matrix.min()


    for i in range(rows):
        for j in range(cols):
            color = int(255.0 * (matrix[i, j] - min_val) / (max_val - min_val))

            pg.draw.rect(surf, (255 - color, 255 - color, color), pg.Rect(j * dx, i * dx, dx, dx))

    return surf


if __name__ == "__main__":
    import sys      
    import numpy as np
    matrix = np.random.randn(15, 15)
    pg.init()
    pg.display.set_caption("Matrix Test")
    size = width, height = 800, 600
    screen = pg.display.set_mode(size)

    surf = draw_matrix(matrix, 20)

    screen.blit(surf, surf.get_rect().move(100, 100))


    pg.display.flip()

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                sys.exit()  

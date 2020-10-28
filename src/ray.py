from math import sin, cos


class Ray:
    def __init__(self, origin=(.0, .0), theta=.0):
        self.origin = origin
        self.theta = theta


    def get_collision(self, mask, threshold):
        d = 0
        x, y = self.origin
        t_x, t_y = x, y
        bounds = mask.get_rect()

        while d < threshold and bounds.collidepoint(t_x, t_y) and mask.get_at((int(t_x), int(t_y))) == 0:
            t_x, t_y = x + d * cos(self.theta), y - d * sin(self.theta)            
            d += 1
            
        return ((int(t_x), int(t_y)), d)
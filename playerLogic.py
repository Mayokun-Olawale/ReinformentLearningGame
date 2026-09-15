import pygame

# from game import window

class Player(pygame.sprite.Sprite):
    COLOR = (255, 0, 0)
    PLAYER_VEL = 5
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.x_vel = 0
        self.y_vel = 0
        self.mask = None
        self.animation_count = 0
        self.direction =  "left"

    def move(self, dx, dy):
        self.rect.x += dx
        self.rect.y += dy

    def move_left(self, vel):
        self.x_vel = -vel
        if self.direction != "left":
            self.direction =  "left"
            self.animation_count = 0

    def move_right(self, vel):
        self.x_vel = vel
        if self.direction != "right":
            self.direction =  "right"
            self.animation_count = 0

    def loop(self, fps):
        self.move(self.x_vel, self.y_vel)

    def draw(self, window):
        pygame.draw.rect(window, self.COLOR, self.rect)

    def handle_move(self):
        keys = pygame.key.get_pressed()

        self.x_vel = 0

        if keys[pygame.K_LEFT]:
            self.move_left(self.PLAYER_VEL)

        if keys[pygame.K_RIGHT]:
            self.move_right(self.PLAYER_VEL)
        




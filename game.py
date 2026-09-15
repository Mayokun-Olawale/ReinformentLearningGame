import os
import random
import math
import pygame
from os import listdir
from os.path import isfile, join

from playerLogic import Player

pygame.init()
pygame.display.set_caption("Platformer")

BG_COLOR = (255, 255, 255)

WIDTH, HEIGHT = 1050, 800

FPS = 60

PLAYER_VEL = 5

window = pygame.display.set_mode((WIDTH,HEIGHT))

def get_background(name):
    image = pygame.image.load(join("assets", "Background", name))
    _, _, width, height = image.get_rect()
    tiles = []

    for i in range(WIDTH // width + 1):
        for j in range(HEIGHT // height + 1):
            pos = (i * width, j * height)
            tiles.append(pos)
    return tiles, image

def draw(window, background, bg_image, player):
    for tile in background:
        window.blit(bg_image, tile)
    player.draw(window)
    pygame.display.update()


def main(window):
    clock = pygame.time.Clock()
    background, bg_img = get_background("Pink.png")

    player = Player(100, 100, 50, 50)

    run = True

    while run:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
        player.loop(FPS)
        player.handle_move()
        draw(window, background,bg_img,player)
    
    pygame.quit()
    quit()

if __name__ == "__main__":
    main(window)


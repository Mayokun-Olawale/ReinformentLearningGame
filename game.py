import os
import random
import math
import pygame
from os import listdir
from os.path import isfile, join
from sprite_utils import load_sprite_sheets

pygame.init()
pygame.display.set_caption("Platformer")

BG_COLOR = (255, 255, 255)

WIDTH, HEIGHT = 1050, 800

FPS = 60

PLAYER_VEL = 5

window = pygame.display.set_mode((WIDTH,HEIGHT))

from playerLogic import Player
from gameObject import Block, Fire

def get_background(name):
    image = pygame.image.load(join("assets", "Background", name))
    _, _, width, height = image.get_rect()
    tiles = []

    for i in range(WIDTH // width + 1):
        for j in range(HEIGHT // height + 1):
            pos = (i * width, j * height)
            tiles.append(pos)
    return tiles, image

def draw(window, background, bg_image, player, objects, offset_x):
    for tile in background:
        window.blit(bg_image, tile)
    for obj in objects:
        obj.draw(window, offset_x)
    player.draw(window, offset_x)
    pygame.display.update()

def main(window):
    clock = pygame.time.Clock()
    background, bg_img = get_background("Pink.png")
    block_size = 96
    floor = [Block(i * block_size, HEIGHT-block_size, block_size) for i in range(-WIDTH//block_size, WIDTH*2//block_size)]
    player = Player(100, HEIGHT - block_size - 64, 50, 50)
    fire = Fire(100, HEIGHT - block_size - 64, 16, 32)
    fire.on()
    objects = [*floor, Block(0, HEIGHT - block_size * 2, block_size), Block(block_size*3, HEIGHT - block_size * 4, block_size), fire]

    run = True
    offset_x = 0
    scroll_area_width = 200

    while run:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and player.jump_count < 2:
                    player.jump()
        player.loop(FPS)
        fire.loop()
        player.handle_move(objects)
        draw(window, background,bg_img,player, objects, offset_x)


        if (player.x_vel > 0 and (player.rect.right - offset_x >= WIDTH - scroll_area_width)) or ((player.rect.left - offset_x <= scroll_area_width) and player.x_vel < 0):
            offset_x += player.x_vel
    

    
    pygame.quit()
    quit()

if __name__ == "__main__":
    main(window)


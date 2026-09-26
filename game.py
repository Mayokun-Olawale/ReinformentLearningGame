import os
import random
import math
import pygame
from os import listdir
from os.path import isfile, join
from sprite_utils import load_sprite_sheets
import random

pygame.init()
pygame.display.set_caption("Platformer")

BG_COLOR = (255, 255, 255)

WIDTH, HEIGHT = 1050, 800

FPS = 60

PLAYER_VEL = 5

window = pygame.display.set_mode((WIDTH,HEIGHT))

from playerLogic import Player
from gameObject import Block, Fire, StartFlag

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
    floor = [Block(i * block_size, HEIGHT-block_size, block_size) for i in range(0, WIDTH*5//block_size)]
    number_of_box_groups = 6
    first_group_start = 4
    group_span = len(floor) - first_group_start - 2
    box_group_starts = [
        first_group_start + i * group_span // number_of_box_groups
        for i in range(number_of_box_groups)
    ]
    box_positions = [
        start + box_offset
        for start in box_group_starts
        for box_offset in range(random.randint(2, 3))
    ]
    floating_box = [
        Block(i * block_size, HEIGHT - block_size * 3.5, block_size)
        for i in box_positions
    ]
    player = Player(block_size*2, HEIGHT - block_size - 64, 50, 50)
    # fire = Fire(100, HEIGHT - block_size - 64, 16, 32)
    startFlag = StartFlag(0, HEIGHT - block_size - 130, 70, 90)

    floor_left = startFlag.rect.left
    floor_right = floor[-1].rect.right
    for i in range(5):
        idx = random.randint(0,len(floor) - 1)
        floor.pop(idx)
    # fire.on()
    objects = [*floor, *floating_box, startFlag]

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
        player.handle_move(objects)
        player.loop(FPS)
        if player.rect.left <= floor_left:
            player.rect.left = floor_left
            player.x_vel = 0
        elif player.rect.right > floor_right:
            player.rect.right = floor_right
            player.x_vel = 0
        # fire.loop()
        startFlag.loop()
        draw(window, background,bg_img,player, objects, offset_x)


        if (player.x_vel > 0 and (player.rect.right - offset_x >= WIDTH - scroll_area_width)) or ((player.rect.left - offset_x <= scroll_area_width) and player.x_vel < 0):
            offset_x += player.x_vel

        offset_x = max(floor_left, min(offset_x, floor_right - WIDTH))
    

    
    pygame.quit()
    quit()

if __name__ == "__main__":
    main(window)


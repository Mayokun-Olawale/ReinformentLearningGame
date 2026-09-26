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
HUD_FONT = pygame.font.Font(None, 32)
HEART_SPRITE = pygame.image.load(join("assets", "Other", "heart.png")).convert_alpha()
HEART_SPRITE = HEART_SPRITE.subsurface(HEART_SPRITE.get_bounding_rect()).copy()
HEART_SPRITE = pygame.transform.scale(HEART_SPRITE, (24, 24))
FRUIT_SHEET = pygame.image.load(join("assets", "Items", "Fruits", "Strawberry.png")).convert_alpha()
FRUIT_SPRITE = FRUIT_SHEET.subsurface(pygame.Rect(0, 0, 32, 32)).copy()
FRUIT_SPRITE = FRUIT_SPRITE.subsurface(FRUIT_SPRITE.get_bounding_rect()).copy()
FRUIT_SPRITE = pygame.transform.scale(FRUIT_SPRITE, (24, 24))

from playerLogic import Player
from gameObject import Block, Enemy, StartFlag, EndTrophy, Fruit

def get_background(name):
    image = pygame.image.load(join("assets", "Background", name))
    _, _, width, height = image.get_rect()
    tiles = []

    for i in range(WIDTH // width + 1):
        for j in range(HEIGHT // height + 1):
            pos = (i * width, j * height)
            tiles.append(pos)
    return tiles, image

def draw_hud_item(window, icon, text, x, center_y):
    icon_rect = icon.get_rect(midleft=(x, center_y))
    text_rect = text.get_rect(midleft=(icon_rect.right + 6, center_y))
    window.blit(icon, icon_rect)
    window.blit(text, text_rect)
    return text_rect.right

def draw_hud(window, player):
    items = [
        (HEART_SPRITE, HUD_FONT.render(str(player.life_count), True, (255, 255, 255))),
        (FRUIT_SPRITE, HUD_FONT.render(str(player.fruit_count), True, (255, 255, 255))),
    ]
    icon_gap = 6
    section_gap = 18
    hud_padding = 10
    hud_width = sum(icon.get_width() + icon_gap + text.get_width() for icon, text in items)
    hud_width += section_gap
    hud_rect = pygame.Rect(
        window.get_width() - hud_width - 16,
        16,
        hud_width,
        max(max(icon.get_height() for icon, _ in items), HUD_FONT.get_height()),
    )
    background_rect = hud_rect.inflate(hud_padding * 2, hud_padding * 2)
    pygame.draw.rect(window, (35, 35, 35), background_rect, border_radius=6)
    x = hud_rect.left
    x = draw_hud_item(window, *items[0], x, hud_rect.centery)
    x += section_gap
    draw_hud_item(window, *items[1], x, hud_rect.centery)

def draw(window, background, bg_image, player, objects, offset_x):
    for tile in background:
        window.blit(bg_image, tile)
    for obj in objects:
        obj.draw(window, offset_x)
    player.draw(window, offset_x)
    draw_hud(window, player)
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
    startFlag = StartFlag(0, HEIGHT - block_size - 150, 100, 150)
    endTrophy = EndTrophy((block_size*len(floor)) - block_size, HEIGHT - block_size * 3.5, 100, 150)

    floor_left = startFlag.rect.left
    floor_right = floor[-1].rect.right
    for i in range(5):
        idx = random.randint(5,len(floor) - 4)
        floor.pop(idx)

    floor_by_x = {block.rect.x: block for block in floor}
    valid_enemy_starts = [
        block.rect.x
        for block in sorted(floor, key=lambda block: block.rect.x)
        if block.rect.x >= block_size * 4
        and block.rect.x <= floor_right - block_size * 4
        and all(block.rect.x + offset * block_size in floor_by_x for offset in range(3))
    ]
    enemy_starts = [
        valid_enemy_starts[index * (len(valid_enemy_starts) - 1) // 3]
        for index in range(4)
    ]
    enemies = [
        Enemy(x, HEIGHT - block_size - 64, 32, 32)
        for x in enemy_starts
    ]
    fruit_size = 64
    fruit_positions = []
    fruit_start_x = player.rect.x
    ground_blocks = [*floor, *floating_box]
    available_ground = [
        block for block in ground_blocks if block.rect.x > fruit_start_x
    ]
    fruit_block_count = max(3, len(available_ground) // 5)
    fruit_blocks = random.sample(available_ground, fruit_block_count)
    stacked_blocks = random.sample(fruit_blocks, 3)

    for block in fruit_blocks:
        block_image_rect = block.image.get_rect(topleft=block.rect.topleft)
        fruit_x = block_image_rect.centerx - fruit_size // 2
        fruit_y = block_image_rect.top - fruit_size
        fruit_positions.append((fruit_x, fruit_y))

        if block in stacked_blocks:
            fruit_positions.append((fruit_x, fruit_y - fruit_size))

    fruits = [Fruit(x, y, fruit_size, fruit_size) for x, y in fruit_positions]
    objects = [*floor, *floating_box, *enemies, startFlag, endTrophy, *fruits]

    run = True
    quit_requested = False
    offset_x = 0
    scroll_area_width = 200

    while run:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                quit_requested = True
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and player.jump_count < 2:
                    player.jump()
        player.handle_move(objects)
        if player.finished:
            run = False
            continue
        player.loop(FPS)
        if player.rect.top > HEIGHT + 20:
            run = False
            player.end_reason = "fell"
            continue
        if player.rect.left <= floor_left:
            player.rect.left = floor_left
            player.x_vel = 0
        elif player.rect.right > floor_right:
            player.rect.right = floor_right
            player.x_vel = 0
        startFlag.loop()
        endTrophy.loop()
        for enemy in enemies:
            enemy.loop()
        for fruit in fruits:
            fruit.loop()
        fruits = [fruit for fruit in fruits if not fruit.remove]
        draw(window, background,bg_img,player, objects, offset_x)

        if (player.x_vel > 0 and (player.rect.right - offset_x >= WIDTH - scroll_area_width)) or ((player.rect.left - offset_x <= scroll_area_width) and player.x_vel < 0):
            offset_x += player.x_vel

        offset_x = max(floor_left, min(offset_x, floor_right - WIDTH))

    return None if quit_requested else player

if __name__ == "__main__":
    main(window)


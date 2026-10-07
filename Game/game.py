import os
import random
import math
import pygame
from os import listdir
from os.path import isfile, join
from .sprite_utils import load_sprite_sheets
import random

pygame.init()
pygame.display.set_caption("Platformer")

ASSET_DIR = join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

BG_COLOR = (255, 255, 255)

WIDTH, HEIGHT = 1050, 800

FPS = 60

PLAYER_VEL = 5

window = pygame.display.set_mode((WIDTH,HEIGHT))
from .hud import draw_hud

from .playerLogic import Player
from .gameObject import Block, Enemy, StartFlag, EndTrophy, Fruit

def get_background(name):
    image = pygame.image.load(join(ASSET_DIR, "Background", name))
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
    draw_hud(window, player)
    pygame.display.update()

def main(window):
    from .engine import GameEngine

    engine = GameEngine(render_mode="human", randomize_obstacles=True)
    engine.screen = window
    engine.clock = pygame.time.Clock()
    while not engine.player.finished:
        events = list(pygame.event.get())
        if any(event.type == pygame.QUIT for event in events):
            return None
        action = engine.handle_human_action(events, pygame.key.get_pressed())
        engine.step(action)
        engine.render()
    return engine.player


if __name__ == "__main__":
    main(window)

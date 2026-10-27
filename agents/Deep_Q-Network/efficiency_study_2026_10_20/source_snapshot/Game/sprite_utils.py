import pygame
import os
from functools import lru_cache
from os import listdir
from os.path import isfile, join

ASSET_DIR = join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def flip(sprites):
    return [pygame.transform.flip(sprite, True, False) for sprite in sprites]


@lru_cache(maxsize=None)
def load_sprite_sheets(dir1, dir2, width, height, direction=False):
    path = join(ASSET_DIR, dir1, dir2)
    # print(path)
    images = [f for f in listdir(path) if isfile(join(path, f))]
    all_sprites = {}

    for image in images:
        sprite_sheet = pygame.image.load(join(path, image))
        sprites = []

        for i in range(sprite_sheet.get_width() // width):
            surface = pygame.Surface((width, height), pygame.SRCALPHA, 32)
            rect = pygame.Rect(i * width, 0, width, height)
            surface.blit(sprite_sheet, (0, 0), rect)
            sprites.append(pygame.transform.scale2x(surface))

        name = image.replace(".png", "")
        if direction:
            all_sprites[name + "_right"] = sprites
            all_sprites[name + "_left"] = flip(sprites)
        else:
            all_sprites[name] = sprites

    return all_sprites

@lru_cache(maxsize=None)
def get_block(size):
    path = join(ASSET_DIR, "Terrain", "Terrain.png")
    image = pygame.image.load(path)
    surface = pygame.Surface((size, size), pygame.SRCALPHA, 32)
    rect = pygame.Rect(96, 0, size, size)
    surface.blit(image, (0,0),rect)
    return pygame.transform.scale2x(surface)

@lru_cache(maxsize=4096)
def sprite_mask(surface):
    """Reuse masks for immutable animation frames; collisions only read them."""
    return pygame.mask.from_surface(surface)

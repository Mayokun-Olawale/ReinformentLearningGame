"""Shared lives and fruit scoreboard; importing this module creates no display."""
import pygame
from os.path import join
from .sprite_utils import ASSET_DIR

HEART_SPRITE = pygame.image.load(join(ASSET_DIR, "Other", "heart.png"))
HEART_SPRITE = HEART_SPRITE.subsurface(HEART_SPRITE.get_bounding_rect()).copy()
HEART_SPRITE = pygame.transform.scale(HEART_SPRITE, (24, 24))
FRUIT_SHEET = pygame.image.load(join(ASSET_DIR, "Items", "Fruits", "Strawberry.png"))
FRUIT_SPRITE = FRUIT_SHEET.subsurface(pygame.Rect(0, 0, 32, 32)).copy()
FRUIT_SPRITE = FRUIT_SPRITE.subsurface(FRUIT_SPRITE.get_bounding_rect()).copy()
FRUIT_SPRITE = pygame.transform.scale(FRUIT_SPRITE, (24, 24))

def draw_hud_item(window, icon, text, x, center_y):
    icon_rect = icon.get_rect(midleft=(x, center_y))
    text_rect = text.get_rect(midleft=(icon_rect.right + 6, center_y))
    window.blit(icon, icon_rect)
    window.blit(text, text_rect)
    return text_rect.right

def draw_hud(window, player):
    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.Font(None, 32)
    items = [
        (HEART_SPRITE, font.render(str(player.life_count), True, (255, 255, 255))),
        (FRUIT_SPRITE, font.render(str(player.fruit_count), True, (255, 255, 255))),
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
        max(max(icon.get_height() for icon, _ in items), font.get_height()),
    )
    background_rect = hud_rect.inflate(hud_padding * 2, hud_padding * 2)
    pygame.draw.rect(window, (35, 35, 35), background_rect, border_radius=6)
    x = hud_rect.left
    x = draw_hud_item(window, *items[0], x, hud_rect.centery)
    x += section_gap
    draw_hud_item(window, *items[1], x, hud_rect.centery)

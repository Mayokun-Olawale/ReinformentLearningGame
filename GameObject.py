import pygame
from sprite_utils import get_block,load_sprite_sheets
class Object(pygame.sprite.Sprite):
    def __init__(self,x,y,width,height,name=None):
        super().__init__()
        self.rect = pygame.Rect(x, y, width, height) #Creates a box with specified width height and starting point
        self.image = pygame.Surface((width, height), pygame.SRCALPHA) #transforms transperant images
        self.width = width
        self.height = height
        self.name = name

    def draw(self,window, offset_x):
        window.blit(self.image, (self.rect.x - offset_x, self.rect.y))

class Block(Object):
    def __init__(self,x,y,size):
        super().__init__(x,y,size,size)
        block = get_block(size)
        self.image.blit(block, (0,0))
        self.mask = pygame.mask.from_surface(self.image)

class Fire(Object):
    ANIMATION_DELAY = 8
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height, "fire")
        self.fire = load_sprite_sheets("Traps", "Fire", width, height)
        self.image = self.fire["off"][0]
        self.mask = pygame.mask.from_surface(self.image)
        self.animation_count = 0
        self.animation_name = "off"

    def on(self):
        self.animation_name = "on"

    def off(self):
        self.animation_name = "off"

    def loop(self):
        sprites = self.fire[self.animation_name]
        sprite_idx = (self.animation_count // self.ANIMATION_DELAY) % len(sprites)
        self.image = sprites[sprite_idx]
        self.animation_count += 1

        self.rect = self.image.get_rect(topleft=(self.rect.x, self.rect.y))
        self.mask = pygame.mask.from_surface(self.image)

        if self.animation_count // self.ANIMATION_DELAY > len(sprites):
            self.animation_count = 0

class StartFlag(Object):
    ANIMATION_DELAY = 6

    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height, "startFlag")
        self.startFlag = load_sprite_sheets("Items/Checkpoints", "Start", width, height)
        self.image = self.startFlag["moving"][0]
        self.mask = pygame.mask.from_surface(self.image)
        self.animation_count = 0

    def loop(self):
        sprites = self.startFlag["moving"]
        sprite_idx = (self.animation_count // self.ANIMATION_DELAY) % len(sprites)
        self.image = sprites[sprite_idx]
        self.animation_count += 1
        
        self.rect = self.image.get_rect(topleft=(self.rect.x, self.rect.y))
        self.mask = pygame.mask.from_surface(self.image)
        
        if self.animation_count // self.ANIMATION_DELAY > len(sprites):
            self.animation_count = 0

class Fruit(Object):
    ANIMATION_DELAY = 6
    
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height, "startFlag")
        self.startFlag = load_sprite_sheets("Items/Checkpoints", "Start", width, height)
        self.image = self.startFlag["moving"][0]
        self.mask = pygame.mask.from_surface(self.image)
        self.animation_count = 0
    
    def loop(self):
        sprites = self.startFlag["moving"]
        sprite_idx = (self.animation_count // self.ANIMATION_DELAY) % len(sprites)
        self.image = sprites[sprite_idx]
        self.animation_count += 1
            
        self.rect = self.image.get_rect(topleft=(self.rect.x, self.rect.y))
        self.mask = pygame.mask.from_surface(self.image)
            
        if self.animation_count // self.ANIMATION_DELAY > len(sprites):
            self.animation_count = 0

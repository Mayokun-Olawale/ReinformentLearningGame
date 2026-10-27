import pygame
from .sprite_utils import get_block,load_sprite_sheets,sprite_mask
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

class Enemy(Object):
    ANIMATION_DELAY = 3
    PATROL_DISTANCE = 96 * 2
    PATROL_SPEED = 3
    SPRITES = load_sprite_sheets("MainCharacters", "MaskDude", 32, 32, True)
    def __init__(self, x, y, width, height, name=None):
        super().__init__(x, y, width, height, "enemy")
        self.start_x = x
        self.x_vel = 0
        self.animation_count = 0
        self.direction = "right"
        self.image = self.SPRITES["run_right"][0]
        self.mask = sprite_mask(self.image)

    def loop(self):
        self.x_vel = self.PATROL_SPEED if self.direction == "right" else -self.PATROL_SPEED
        self.rect.x += self.x_vel

        if self.rect.x >= self.start_x + self.PATROL_DISTANCE:
            self.rect.x = self.start_x + self.PATROL_DISTANCE
            self.direction = "left"
        elif self.rect.x <= self.start_x:
            self.rect.x = self.start_x
            self.direction = "right"

        sprites = self.SPRITES[f"run_{self.direction}"]
        sprite_idx = (self.animation_count // self.ANIMATION_DELAY) % len(sprites)
        self.image = sprites[sprite_idx]
        self.animation_count += 1
        self.rect = self.image.get_rect(topleft=(self.rect.x, self.rect.y))
        self.mask = sprite_mask(self.image)

class Block(Object):
    def __init__(self,x,y,size):
        super().__init__(x,y,size,size)
        block = get_block(size)
        self.image.blit(block, (0,0))
        self.mask = sprite_mask(self.image)

class StartFlag(Object):
    ANIMATION_DELAY = 3

    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height, "startFlag")
        source_start_flag = load_sprite_sheets("Items/Checkpoints", "Start", 64, 64)
        self.startFlag = {
            "moving": [
                pygame.transform.scale(sprite, (width, height))
                for sprite in source_start_flag["moving"]
            ]
        }
        self.image = self.startFlag["moving"][0]
        self.mask = sprite_mask(self.image)
        self.animation_count = 0

    def loop(self):
        sprites = self.startFlag["moving"]
        sprite_idx = (self.animation_count // self.ANIMATION_DELAY) % len(sprites)
        flag_anchor = self.rect.midbottom
        self.image = sprites[sprite_idx]
        self.animation_count += 1
        
        self.rect = self.image.get_rect(midbottom=flag_anchor)
        self.mask = sprite_mask(self.image)
        
        if self.animation_count // self.ANIMATION_DELAY > len(sprites):
            self.animation_count = 0

class EndTrophy(Object):
    ANIMATION_DELAY = 9

    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height, "endTrophy")
        source_end_trophy = load_sprite_sheets("Items/Checkpoints", "End", 64, 64)
        self.endTrophy = {
            "moving": [
                pygame.transform.scale(sprite, (width, height))
                for sprite in source_end_trophy["moving"]
            ]
        }
        self.image = self.endTrophy["moving"][0]
        self.mask = sprite_mask(self.image)
        self.animation_count = 0

    def loop(self):
        sprites = self.endTrophy["moving"]
        sprite_idx = (self.animation_count // self.ANIMATION_DELAY) % len(sprites)
        trophy_anchor = self.rect.midbottom
        self.image = sprites[sprite_idx]
        self.animation_count += 1
        
        self.rect = self.image.get_rect(midbottom=trophy_anchor)
        self.mask = sprite_mask(self.image)
        
        if self.animation_count // self.ANIMATION_DELAY > len(sprites):
            self.animation_count = 0

class Fruit(Object):
    ANIMATION_DELAY = 6
    
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height, "fruit")
        source_fruit = load_sprite_sheets("Items", "Fruits", 32, 32)
        self.fruit = {
            "Strawberry": [
                pygame.transform.scale(sprite, (width, height))
                for sprite in source_fruit["Strawberry"]
            ]
        }
        self.collected_sprites = [
            pygame.transform.scale(sprite, (width, height))
            for sprite in source_fruit["Collected"]
        ]
        self.image = self.fruit["Strawberry"][0]
        self.mask = sprite_mask(self.image)
        self.animation_count = 0
        self.collected = False
        self.remove = False

    def collect(self):
        if self.collected:
            return
        self.collected = True
        self.animation_count = 0
        self.mask = pygame.Mask(self.image.get_size())

    def loop(self):
        sprites = self.collected_sprites if self.collected else self.fruit["Strawberry"]
        sprite_idx = (self.animation_count // self.ANIMATION_DELAY) % len(sprites)
        fruit_anchor = self.rect.midbottom
        self.image = sprites[sprite_idx]
        self.animation_count += 1
            
        self.rect = self.image.get_rect(midbottom=fruit_anchor)
        if not self.collected:
            self.mask = sprite_mask(self.image)
            
        if self.collected and self.animation_count >= self.ANIMATION_DELAY * len(sprites):
            self.remove = True
        elif self.animation_count // self.ANIMATION_DELAY > len(sprites):
            self.animation_count = 0

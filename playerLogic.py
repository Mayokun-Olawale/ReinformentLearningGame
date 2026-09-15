import pygame

from sprite_utils import load_sprite_sheets

class Player(pygame.sprite.Sprite):
    COLOR = (255, 0, 0)
    PLAYER_VEL = 5
    GRAVITY = 1
    SPRITES = load_sprite_sheets("MainCharacters", "PinkMan", 32, 32, True)
    ANIMATION_DELAY = 3


    def __init__(self, x, y, width, height):
        super().__init__()
        self.rect = pygame.Rect(x, y, width, height)
        self.x_vel = 0
        self.y_vel = 0
        self.mask = None
        self.animation_count = 0
        self.direction =  "left"
        self.fall_count = 0
        self.jump_count = 0
        self.update_sprite()

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
        self.y_vel += min(1, ((self.fall_count/fps) * self.GRAVITY))
        self.move(self.x_vel, self.y_vel)

        self.fall_count += 1

        self.update_sprite()

    def draw(self, window, offset_x):
        window.blit(self.sprite, (self.rect.x - offset_x,self.rect.y))

    def handle_move(self, objects):
        keys = pygame.key.get_pressed()

        self.x_vel = 0

        collide_left = self.collide(objects, -self.PLAYER_VEL * 2)
        collide_right = self.collide(objects, self.PLAYER_VEL * 2)

        if keys[pygame.K_LEFT] and  not collide_left:
            self.move_left(self.PLAYER_VEL)

        if keys[pygame.K_RIGHT] and not collide_right:
            self.move_right(self.PLAYER_VEL)

    
        self.handle_vertical_collision(objects, self.y_vel)
        

    def update_sprite(self):
        sprite_sheet = "idle"
        if self.y_vel < 0:
            if self.jump_count == 1:
                sprite_sheet = "jump"
            elif self.jump_count == 2:
                sprite_sheet = "double_jump"
        elif self.y_vel > self.GRAVITY * 2:
            sprite_sheet = "fall"
        elif self.x_vel != 0:
            sprite_sheet = "run"
        sprite_sheet_name = sprite_sheet + "_" + self.direction

        sprites = self.SPRITES[sprite_sheet_name]
        sprite_idx = (self.animation_count // self.ANIMATION_DELAY) % len(sprites)
        self.sprite = sprites[sprite_idx]
        self.animation_count += 1
        self.update()

    def update(self):
        self.rect = self.sprite.get_rect(topleft=(self.rect.x, self.rect.y))
        self.mask = pygame.mask.from_surface(self.sprite)

    def landed(self):
        self.fall_count = 0
        self.y_vel = 0
        self.jump_count = 0

    def hit_head(self):
        self.fall_count = 0
        self.y_vel *= -1

    def handle_vertical_collision(self, objects, dy):
        collied_objects = []

        for obj in objects:
            if pygame.sprite.collide_mask(self, obj):
                if dy > 0:
                    self.rect.bottom = obj.rect.top
                    self.landed()
                elif dy < 0:
                    self.rect.top = obj.rect.bottom
                    self.hit_head()
            collied_objects.append(obj)
        return collied_objects

    def jump(self):
        if self.jump_count >= 2:
            return

        self.y_vel = -self.GRAVITY * 8
        self.animation_count = 0
        self.jump_count +=1 
        if self.jump_count == 1:
            self.fall_count = 0

    def collide(self, objects, dx):
        self.move(dx, 0)
        self.update()
        collided_obj = None
        for obj in objects:
            if pygame.sprite.collide_mask(self, obj):
                collided_obj = obj
                break
        self.move(-dx,0)
        self.update()
        return collided_obj










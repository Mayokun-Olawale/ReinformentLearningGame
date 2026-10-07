import pygame

from .sprite_utils import load_sprite_sheets, sprite_mask

class Player(pygame.sprite.Sprite):
    COLOR = (255, 0, 0)
    PLAYER_VEL = 5
    GRAVITY = 1
    SPRITES = load_sprite_sheets("MainCharacters", "PinkMan", 32, 32, True)
    ANIMATION_DELAY = 3


    def __init__(self, x, y, width, height, character="PinkMan"):
        super().__init__()
        self.visual_sprites = load_sprite_sheets("MainCharacters", character, 32, 32, True)
        self.rect = pygame.Rect(x, y, width, height)
        self.x_vel = 0
        self.y_vel = 0
        self.vx = 0.0
        self.vy = 0.0
        self.mask = None
        self.animation_count = 0
        self.direction =  "left"
        self.fall_count = 0
        self.jump_count = 0
        self.hit = False
        self.hit_count = 0
        self.update_sprite()
        self.life_count = 3
        self.fruit_count = 0
        self.finished = False
        self.end_reason = None
        self.is_grounded = False
        self.can_double_jump = True

    def apply_action(self, action: int):
        if action == 0:
            self.x_vel = -self.PLAYER_VEL
            self.direction = "left"
        elif action == 1:
            self.x_vel = self.PLAYER_VEL
            self.direction = "right"
        elif action == 2:
            if self.is_grounded:
                self.y_vel = -self.GRAVITY * 9.5
                self.is_grounded = False
                self.can_double_jump = True
                self.jump_count = 1
                self.fall_count = 0
                self.animation_count = 0
        elif action == 3:
            if not self.is_grounded and self.can_double_jump:
                self.y_vel = -self.GRAVITY * 8.5
                self.can_double_jump = False
                self.jump_count = 2
        elif action == 4:
            self.x_vel = 0
        else:
            self.x_vel = 0

    def update_physics(self, gravity: float = 1.0):
        self.y_vel += min(1, ((self.fall_count / 60) * gravity))
        self.rect.x += self.x_vel
        self.rect.y += self.y_vel
        self.vx = float(self.x_vel)
        self.vy = float(self.y_vel)
        self.fall_count += 1
        if self.hit:
            self.hit_count += 1
            if self.hit_count > 60:
                self.hit = False
                self.hit_count = 0
        self.update_sprite()

    def reset_state(self, spawn_x: float, spawn_y: float):
        self.rect.x = spawn_x
        self.rect.y = spawn_y
        self.x_vel = 0
        self.y_vel = 0
        self.vx = 0.0
        self.vy = 0.0
        self.jump_count = 0
        self.fall_count = 0
        self.is_grounded = False
        self.can_double_jump = True
        self.life_count = 3
        self.fruit_count = 0
        self.finished = False
        self.end_reason = None
        self.hit = False
        self.hit_count = 0
        self.animation_count = 0
        self.direction = "left"
        self.update_sprite()

    @property
    def lives(self):
        return self.life_count

    @lives.setter
    def lives(self, value):
        self.life_count = value

    def make_hit(self):
        if self.hit:
            return
        self.hit = True
        self.hit_count = 0
        self.animation_count = 0
        self.update_sprite()
        self.life_count = max(0, self.life_count - 1)
        if self.life_count == 0:
            self.finished = True
            self.end_reason = "defeated"

    def collect_fruit(self, fruit):
        if fruit.collected:
            return
        fruit.collect()
        self.fruit_count += 1

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
        if self.hit:
            self.hit_count += 1
        if self.hit_count > fps * 1:
            self.hit = False
        self.update_sprite()

    def draw(self, window, offset_x):
        window.blit(self.visual_sprite, (self.rect.x - offset_x,self.rect.y))

    def handle_move(self, objects):
        keys = pygame.key.get_pressed()

        self.x_vel = 0

        collide_left = self.collide(objects, -self.PLAYER_VEL * 2)
        collide_right = self.collide(objects, self.PLAYER_VEL * 2)

        if keys[pygame.K_LEFT] and  not collide_left:
            self.move_left(self.PLAYER_VEL)

        if keys[pygame.K_RIGHT] and not collide_right:
            self.move_right(self.PLAYER_VEL)

    
        vertical_collide = self.handle_vertical_collision(objects, self.y_vel)
        to_check = [collide_left, collide_right, *vertical_collide]
        for obj in to_check:
            if obj and obj.name == "endTrophy":
                self.finished = True
                self.end_reason = "completed"
            elif obj and obj.name == "enemy" and not self.hit:
                self.make_hit()
            elif obj and obj.name == "fruit":
                self.collect_fruit(obj)
        

    def update_sprite(self):
        sprite_sheet = "idle"
        if self.hit:
            sprite_sheet = "hit"
        elif self.y_vel < 0:
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
        visual_frames = self.visual_sprites[sprite_sheet_name]
        self.visual_sprite = visual_frames[sprite_idx % len(visual_frames)]
        self.animation_count += 1
        self.update()

    def update(self):
        self.rect = self.sprite.get_rect(topleft=(self.rect.x, self.rect.y))
        self.mask = sprite_mask(self.sprite)

    def landed(self):
        self.is_grounded = True
        self.can_double_jump = True
        self.fall_count = 0
        self.y_vel = 0
        self.jump_count = 0

    def hit_head(self):
        self.fall_count = 0
        self.y_vel *= -1

    def handle_vertical_collision(self, objects, dy):
        collied_objects = []

        for obj in objects:
            if self.rect.colliderect(obj.rect) and pygame.sprite.collide_mask(self, obj):
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

        self.y_vel = -self.GRAVITY * 9.5
        self.animation_count = 0
        self.jump_count +=1 
        if self.jump_count == 1:
            self.fall_count = 0

    def collide(self, objects, dx):
        self.move(dx, 0)
        self.update()
        collided_obj = None
        for obj in objects:
            if self.rect.colliderect(obj.rect) and pygame.sprite.collide_mask(self, obj):
                collided_obj = obj
                break
        self.move(-dx,0)
        self.update()
        return collided_obj










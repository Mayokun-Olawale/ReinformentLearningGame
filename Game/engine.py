from __future__ import annotations

import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from .playerLogic import Player
from .gameObject import Block, Enemy, StartFlag, EndTrophy, Fruit
from .sprite_utils import ASSET_DIR
from .hud import draw_hud
from pathlib import Path


class Level:
    def __init__(self, width=1050, height=800):
        self.width, self.height = width, height
        self.reset_level()

    def reset_level(self, randomize=False, rng=None):
        rng = rng if randomize and rng is not None else random.Random(0)
        block_size = 96
        floor = [Block(i * block_size, self.height-block_size, block_size) for i in range(0, self.width*5//block_size)]
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
            for box_offset in range(rng.randint(2, 3))
        ]
        floating_box = [
            Block(i * block_size, self.height - block_size * 3.5, block_size)
            for i in box_positions
        ]
        player = Player(block_size*2, self.height - block_size - 64, 50, 50)
        startFlag = StartFlag(0, self.height - block_size - 150, 100, 150)
        endTrophy = EndTrophy((block_size*len(floor)) - block_size, self.height - block_size * 3.5, 100, 150)

        floor_left = startFlag.rect.left
        floor_right = floor[-1].rect.right
        for i in range(5):
            idx = rng.randint(5,len(floor) - 4)
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
            Enemy(x, self.height - block_size - 64, 32, 32)
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
        fruit_blocks = rng.sample(available_ground, fruit_block_count)
        stacked_blocks = rng.sample(fruit_blocks, 3)

        for block in fruit_blocks:
            block_image_rect = block.image.get_rect(topleft=block.rect.topleft)
            fruit_x = block_image_rect.centerx - fruit_size // 2
            fruit_y = block_image_rect.top - fruit_size
            fruit_positions.append((fruit_x, fruit_y))

            if block in stacked_blocks:
                fruit_positions.append((fruit_x, fruit_y - fruit_size))

        fruits = [Fruit(x, y, fruit_size, fruit_size) for x, y in fruit_positions]
        objects = [*floor, *floating_box, *enemies, startFlag, endTrophy, *fruits]

        self.spawn_x, self.spawn_y = player.rect.topleft
        self.floor_left, self.floor_right = floor_left, floor_right
        self.world_width = floor_right
        self.platforms = [*floor, *floating_box]
        self.hazards = enemies
        self.fruits = fruits
        self.goal = endTrophy
        self.objects = objects

    def get_distance_to_nearest_fruit(self, player):
        distances = [np.hypot(player.rect.centerx - fruit.rect.centerx,
                              player.rect.centery - fruit.rect.centery)
                     for fruit in self.fruits if not fruit.collected]
        return min(distances) / np.hypot(self.world_width, self.height) if distances else 1.0


class GameEngine:
    def __init__(self, render_mode=None, randomize_obstacles=False, width=1050, height=800,
                 max_episode_steps=18000, character="PinkMan"):
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("Unsupported render_mode")
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        self.render_mode = render_mode
        self.randomize_obstacles = randomize_obstacles
        self.width, self.height = width, height
        self.max_time = max_episode_steps
        self.gravity = Player.GRAVITY
        self.max_lives = 3
        self.level = Level(width, height)
        self.player = Player(self.level.spawn_x, self.level.spawn_y, 50, 50, character=character)
        self.screen = self.clock = None
        self.background = pygame.image.load(str(Path(ASSET_DIR) / "Background" / "Pink.png"))
        self.reset()

    def reset(self, rng=None):
        self.level.reset_level(self.randomize_obstacles, rng)
        self.player.reset_state(self.level.spawn_x, self.level.spawn_y)
        self.time_remaining = self.max_time
        self.state_events = dict.fromkeys(
            ("picked_fruit", "took_hit", "reached_goal", "fell_off_map", "time_expired"), False)
        return self.get_state_vector()

    def handle_human_action(self, event_queue=None, pressed_keys=None):
        pressed = {} if pressed_keys is None else pressed_keys
        if not hasattr(pressed, "get"):
            pressed = {key: bool(pressed[key]) for key in
                       (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_a, pygame.K_d)}
        keys = dict(pressed)
        keys.update(grounded=self.player.is_grounded,
                    can_double_jump=self.player.can_double_jump)
        return get_human_action(event_queue or [], keys)

    def step(self, action):
        if action not in range(5):
            raise ValueError("action must be an integer from 0 to 4")
        if self.player.finished:
            raise RuntimeError("Episode finished; call reset() before step()")
        player = self.player
        for obj in self.level.objects:
            if hasattr(obj, "loop"):
                obj.loop()
        player.apply_action(int(action))
        # Preserve the existing mask collisions and gravity/jump behavior.
        horizontal = player.collide(self.level.platforms, player.x_vel)
        if horizontal:
            player.x_vel = 0
        player.is_grounded = False
        player.update_physics(self.gravity)
        player.handle_vertical_collision(self.level.platforms, player.y_vel)
        # A resting sprite may not move a pixel with the fractional gravity.
        # Probe support so grounded does not flicker between physics ticks.
        if not player.is_grounded and player.y_vel >= 0:
            player.rect.y += 1
            supported = any(player.rect.colliderect(block.rect) and pygame.sprite.collide_mask(player, block)
                            for block in self.level.platforms
                            if player.rect.bottom <= block.rect.top + 1)
            player.rect.y -= 1
            if supported:
                player.landed()
        player.rect.left = max(self.level.floor_left, player.rect.left)
        player.rect.right = min(self.level.floor_right, player.rect.right)
        old_fruit, old_lives = player.fruit_count, player.life_count
        for fruit in self.level.fruits:
            if not fruit.collected and pygame.sprite.collide_mask(player, fruit):
                player.collect_fruit(fruit)
        for enemy in self.level.hazards:
            if pygame.sprite.collide_mask(player, enemy):
                player.make_hit()
        self.time_remaining = max(0, self.time_remaining - 1)
        events = {
            "picked_fruit": player.fruit_count > old_fruit,
            "took_hit": player.life_count < old_lives,
            "reached_goal": bool(pygame.sprite.collide_mask(player, self.level.goal)),
            "fell_off_map": player.rect.top > self.height + 20,
            "time_expired": self.time_remaining == 0,
        }
        if events["reached_goal"]:
            player.end_reason = "completed"
        elif events["fell_off_map"]:
            player.end_reason = "fell"
        elif player.life_count <= 0:
            player.end_reason = "defeated"
        elif events["time_expired"]:
            player.end_reason = "time_expired"
        player.finished = player.end_reason is not None
        player.vx, player.vy = float(player.x_vel), float(player.y_vel)
        self.state_events = events
        return self.get_state_vector(), events.copy(), player.finished

    OBSERVATION_SIZE = 51
    TERRAIN_OFFSETS = (-384, -192, -96, -32, 32, 96, 192, 384)

    def get_state_vector(self):
        player = self.player
        rect = player.rect
        state = [
            rect.centerx / self.level.world_width, rect.centery / self.height,
            player.x_vel / 10.0, player.y_vel / 15.0,
            player.life_count / self.max_lives, self.time_remaining / self.max_time,
            float(player.is_grounded), float(player.can_double_jump),
        ]
        def nearest(objects, count):
            return sorted(objects, key=lambda obj:
                          (obj.rect.centerx - rect.centerx) ** 2 +
                          (obj.rect.centery - rect.centery) ** 2)[:count]

        fruits = nearest([fruit for fruit in self.level.fruits if not fruit.collected], 3)
        for index in range(3):
            if index < len(fruits):
                target = fruits[index].rect
                state.extend([(target.centerx - rect.centerx) / self.width,
                              (target.centery - rect.centery) / self.height, 1.0])
            else:
                state.extend([0.0, 0.0, 0.0])
        enemies = nearest(self.level.hazards, 2)
        for index in range(2):
            if index < len(enemies):
                enemy = enemies[index]
                velocity = enemy.PATROL_SPEED if enemy.direction == "right" else -enemy.PATROL_SPEED
                state.extend([(enemy.rect.centerx - rect.centerx) / self.width,
                              (enemy.rect.centery - rect.centery) / self.height,
                              velocity / 10.0, 1.0])
            else:
                state.extend([0.0, 0.0, 0.0, 0.0])
        source = self.level.platforms
        if (getattr(self, "_terrain_source", None) is not source or
                getattr(self, "_terrain_count", None) != len(source)):
            self._terrain_source, self._terrain_count = source, len(source)
            self._terrain_columns = {}
            for block in source:
                for column in range(block.rect.left // 96, (block.rect.right - 1) // 96 + 1):
                    self._terrain_columns.setdefault(column, []).append(block)
        for offset in self.TERRAIN_OFFSETS:
            x = rect.centerx + offset
            platforms = [block for block in self._terrain_columns.get(x // 96, [])
                         if block.rect.left <= x < block.rect.right]
            below = any(block.rect.top >= rect.bottom - 2 for block in platforms)
            if platforms:
                closest = min(platforms, key=lambda block: abs(block.rect.top - rect.bottom))
                state.extend([float(below), (closest.rect.top - rect.bottom) / self.height, 1.0])
            else:
                state.extend([0.0, 0.0, 0.0])
        state.extend([(self.level.goal.rect.centerx - rect.centerx) / self.level.world_width,
                      (self.level.goal.rect.centery - rect.centery) / self.height])
        return np.clip(state, -1.0, 1.0).tolist()

    def render(self):
        if self.render_mode is None:
            return None
        if self.screen is None:
            if self.render_mode == "human":
                pygame.display.init()
                pygame.font.init()
                self.screen = pygame.display.set_mode((self.width, self.height))
                self.clock = pygame.time.Clock()
            else:
                self.screen = pygame.Surface((self.width, self.height))
        offset = max(0, min(self.player.rect.centerx - self.width // 2,
                            self.level.world_width - self.width))
        for x in range(0, self.width, self.background.get_width()):
            for y in range(0, self.height, self.background.get_height()):
                self.screen.blit(self.background, (x, y))
        for obj in self.level.objects:
            if getattr(obj, "remove", False) is not True:
                obj.draw(self.screen, offset)
        self.player.draw(self.screen, offset)
        draw_hud(self.screen, self.player)
        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.render_mode == "human" and self.screen is not None:
            pygame.display.quit()
        self.screen = self.clock = None


class PlatformerGameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, randomize_obstacles=False, max_episode_steps=18000,
                 character="PinkMan"):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(GameEngine.OBSERVATION_SIZE,), dtype=np.float32)
        self.game = GameEngine(render_mode, randomize_obstacles,
                               max_episode_steps=max_episode_steps, character=character)
        self._has_reset = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        state = self.game.reset(rng=random.Random(int(self.np_random.integers(0, 2**63))))
        self.furthest_x = self.game.player.rect.centerx
        self.progress_pixels = 0.0
        self._has_reset = True
        if self.render_mode == "human":
            self.render()
        return np.asarray(state, dtype=np.float32), {}

    def step(self, action):
        if not self._has_reset:
            raise gym.error.ResetNeeded("Call reset() before step()")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        state, events, done = self.game.step(int(action))
        new_x = self.game.player.rect.centerx
        progress = max(0, new_x - self.furthest_x)
        self.furthest_x = max(self.furthest_x, new_x)
        self.progress_pixels += progress
        progress_reward = progress * 0.002
        terminated = bool(events["reached_goal"] or events["fell_off_map"]
                          or self.game.player.life_count <= 0)
        truncated = bool(events["time_expired"])
        if self.render_mode == "human":
            self.render()
        return (np.asarray(state, dtype=np.float32), self._compute_reward(events) + progress_reward,
                terminated, truncated, {"events": events,
                                       "progress_reward": progress_reward,
                                       "progress_pixels": self.progress_pixels})

    def _compute_reward(self, events):
        reward = float(events["picked_fruit"]) - float(events["took_hit"])
        if events["reached_goal"]:
            reward += 2.0
        elif events["fell_off_map"] or events["time_expired"]:
            reward -= 2.0
        return reward

    def render(self):
        return self.game.render()

    def close(self):
        self.game.close()


def get_human_action(event_queue: list | None = None, pressed_keys: dict | None = None) -> int:
    """Map human keyboard state into the same action protocol as the RL engine."""
    keys = {} if pressed_keys is None else pressed_keys

    def is_pressed(names: tuple[str, ...], key_codes: tuple[int, ...]) -> bool:
        getter = getattr(keys, "get", None)
        if getter is not None and any(bool(getter(name, False)) for name in names):
            return True
        if getter is not None and any(bool(getter(key_code, False)) for key_code in key_codes):
            return True
        if getter is not None:
            return False
        return any(bool(keys[key_code]) for key_code in key_codes)

    left = bool(
        is_pressed(
            ("left", "a", "left_arrow", "LEFT"),
            (pygame.K_LEFT, pygame.K_a),
        )
    )
    right = bool(
        is_pressed(
            ("right", "d", "right_arrow", "RIGHT"),
            (pygame.K_RIGHT, pygame.K_d),
        )
    )
    jump = bool(
        is_pressed(
            ("jump", "space", "up", "w", "SPACE", "W"),
            (pygame.K_SPACE, pygame.K_UP, pygame.K_w),
        )
    )
    for event in event_queue or []:
        if getattr(event, "type", None) == pygame.KEYDOWN and getattr(event, "key", None) in (
            pygame.K_SPACE,
            pygame.K_UP,
            pygame.K_w,
        ):
            jump = True
            break

    getter = getattr(keys, "get", None)
    grounded = bool(getter("grounded", True)) if getter is not None else True
    can_double_jump = bool(getter("can_double_jump", False)) if getter is not None else False

    if jump:
        if grounded:
            return 2
        if can_double_jump:
            return 3
    if left and not right:
        return 0
    if right and not left:
        return 1
    return 4

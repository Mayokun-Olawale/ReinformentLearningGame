import numpy as np

from Game.engine import GameEngine, PlatformerGameEnv, get_human_action


def test_get_human_action_maps_keys():
    assert get_human_action([], {"left": True, "right": False, "jump": False}) == 0
    assert get_human_action([], {"left": False, "right": True, "jump": False}) == 1
    assert get_human_action([], {"left": False, "right": False, "jump": True}) == 2
    assert get_human_action([], {"left": False, "right": False, "jump": False}) == 4


def test_get_human_action_maps_pygame_jump_event():
    import pygame

    jump_event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)
    assert get_human_action([jump_event], {}) == 2


def test_game_engine_step_returns_vector_and_events():
    engine = GameEngine(render_mode=None, randomize_obstacles=False)
    state = engine.reset()
    assert len(state) == 51
    state, events, done = engine.step(4)
    assert len(state) == 51
    assert isinstance(events, dict)
    assert set(events).issuperset({"picked_fruit", "took_hit", "reached_goal", "fell_off_map", "time_expired"})
    assert isinstance(done, bool)


def test_engine_routes_actions_through_player_action_handler(monkeypatch):
    engine = GameEngine(render_mode=None, randomize_obstacles=False)
    received_actions = []

    def record_action(action):
        received_actions.append(action)

    monkeypatch.setattr(engine.player, "apply_action", record_action)
    engine.step(1)

    assert received_actions == [1]


def test_time_expiration_synchronizes_terminal_engine_state():
    engine = GameEngine(render_mode=None, randomize_obstacles=False)
    engine.reset()
    engine.time_remaining = 1

    _, events, done = engine.step(4)

    assert events["time_expired"] is True
    assert done is True
    assert engine.player.finished is True
    assert engine.player.end_reason == "time_expired"


def test_game_engine_uses_human_action_mapper():
    engine = GameEngine(render_mode=None, randomize_obstacles=False)
    assert engine.handle_human_action([], {"right": True}) == 1
    engine.player.is_grounded = True
    assert engine.handle_human_action([], {"jump": True}) == 2


def test_engine_stays_headless_when_render_mode_is_none():
    engine = GameEngine(render_mode=None, randomize_obstacles=False)
    assert engine.clock is None
    assert engine.render() is None


def test_env_reset_and_step_are_compatible_with_gymnasium():
    env = PlatformerGameEnv(render_mode=None, randomize_obstacles=False)
    obs, info = env.reset(seed=123)
    assert obs.shape == (51,)
    assert isinstance(info, dict)
    obs, reward, terminated, truncated, info = env.step(4)
    assert obs.shape == (51,)
    assert isinstance(reward, (float, int, np.floating))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env
    env = PlatformerGameEnv()
    check_env(env, skip_render_check=True)
    env.close()


def test_course_uses_original_objects_and_player():
    from Game.playerLogic import Player
    from Game.gameObject import Block, Enemy, Fruit, EndTrophy
    env = PlatformerGameEnv()
    env.reset()
    assert isinstance(env.game.player, Player)
    assert env.game.level.world_width > env.game.width * 4
    assert all(isinstance(obj, Block) for obj in env.game.level.platforms)
    assert all(isinstance(obj, Enemy) for obj in env.game.level.hazards)
    assert all(isinstance(obj, Fruit) for obj in env.game.level.fruits)
    assert isinstance(env.game.level.goal, EndTrophy)


def test_seeded_randomization_and_fixed_layout():
    env = PlatformerGameEnv(randomize_obstacles=True)

    def layout(seed):
        env.reset(seed=seed)
        return [tuple(obj.rect) for obj in env.game.level.objects]

    assert layout(123) == layout(123)
    assert layout(123) != layout(456)
    fixed = PlatformerGameEnv()
    fixed.reset(seed=123)
    first = [tuple(obj.rect) for obj in fixed.game.level.objects]
    fixed.reset(seed=456)
    assert first == [tuple(obj.rect) for obj in fixed.game.level.objects]


def test_timeout_is_truncation_and_long_course_has_enough_time():
    env = PlatformerGameEnv(max_episode_steps=1)
    env.reset()
    _, reward, terminated, truncated, info = env.step(4)
    assert not terminated and truncated
    assert reward == -2.0
    assert info["events"]["time_expired"]
    assert PlatformerGameEnv().game.max_time == 60 * 300


def test_goal_and_fall_terminate():
    for reason in ("goal", "fall"):
        env = PlatformerGameEnv()
        env.reset()
        if reason == "goal":
            env.game.player.rect.topleft = env.game.level.goal.rect.topleft
        else:
            env.game.player.rect.y = env.game.height + 100
        _, _, terminated, truncated, _ = env.step(4)
        assert terminated and not truncated


def test_damage_emits_once_during_invulnerability_and_recovers():
    env = PlatformerGameEnv()
    env.reset()
    player = env.game.player
    enemy = env.game.level.hazards[0]
    player.rect.topleft = enemy.rect.topleft
    _, events, _ = env.game.step(4)
    assert events["took_hit"] and player.lives == 2
    player.rect.topleft = enemy.rect.topleft
    _, events, _ = env.game.step(4)
    assert not events["took_hit"] and player.lives == 2
    for _ in range(61):
        player.rect.topleft = (192, 640)
        env.game.step(4)
    assert not player.hit
    player.rect.topleft = enemy.rect.topleft
    _, events, _ = env.game.step(4)
    assert events["took_hit"] and player.lives == 1
    env.reset()
    assert not player.hit and player.lives == 3


def test_fruit_reward_only_once():
    env = PlatformerGameEnv()
    env.reset()
    fruit = env.game.level.fruits[0]
    env.game.player.rect.topleft = fruit.rect.topleft
    _, reward, _, _, info = env.step(4)
    assert info["events"]["picked_fruit"]
    assert np.isclose(reward - info["progress_reward"], 1.0)
    env.game.player.rect.topleft = fruit.rect.topleft
    _, reward, _, _, info = env.step(4)
    assert not info["events"]["picked_fruit"] and reward == 0.0


def test_rgb_array_and_headless_do_not_initialize_display():
    import pygame
    pygame.display.quit()
    env = PlatformerGameEnv()
    env.reset()
    env.step(4)
    assert not pygame.display.get_init()
    pixels = PlatformerGameEnv(render_mode="rgb_array")
    pixels.reset()
    frame = pixels.render()
    assert frame.shape == (800, 1050, 3)
    assert frame.dtype == np.uint8
    assert not pygame.display.get_init()
    pixels.close()


def test_jump_landing_and_walking_off_platform():
    engine = GameEngine()
    for _ in range(10):
        engine.step(4)
    assert engine.player.is_grounded
    engine.step(2)
    assert engine.player.y_vel < 0 and not engine.player.is_grounded
    engine.step(3)
    assert not engine.player.can_double_jump
    for _ in range(100):
        engine.step(4)
    assert engine.player.is_grounded and engine.player.can_double_jump
    # Place on a floating block and walk beyond its edge.
    block = engine.level.platforms[-1]
    engine.player.rect.bottom = block.rect.top
    engine.player.rect.left = block.rect.right - 1
    engine.step(1)
    assert not engine.player.is_grounded


def test_invalid_actions_and_reset_required():
    import pytest
    import gymnasium as gym
    env = PlatformerGameEnv(max_episode_steps=1)
    with pytest.raises(gym.error.ResetNeeded):
        env.step(4)
    env.reset()
    with pytest.raises(ValueError):
        env.step(5)
    env.step(4)
    with pytest.raises(RuntimeError):
        env.step(4)


def test_real_pygame_keyboard_wrapper_and_jump_priority():
    import pygame
    pygame.display.init()
    try:
        assert get_human_action([], pygame.key.get_pressed()) == 4
        engine = GameEngine()
        engine.player.is_grounded = True
        event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)
        assert engine.handle_human_action([event], {pygame.K_RIGHT: True}) == 2
    finally:
        pygame.display.quit()


def test_zero_lives_terminates_and_human_render_works():
    import pygame
    env = PlatformerGameEnv(render_mode="human")
    try:
        env.reset()
        assert pygame.display.get_surface() is not None
        env.game.player.life_count = 0
        _, _, terminated, truncated, _ = env.step(4)
        assert terminated and not truncated
    finally:
        env.close()


def test_renderer_draws_course_objects_and_skips_only_removed_fruit(monkeypatch):
    engine = GameEngine(render_mode="rgb_array")
    drawn = []
    for obj in engine.level.objects:
        monkeypatch.setattr(obj, "draw", lambda screen, offset, obj=obj: drawn.append(obj))
    removed = engine.level.fruits[0]
    removed.remove = True
    engine.render()
    assert removed not in drawn
    assert all(obj in drawn for obj in engine.level.platforms)
    assert all(obj in drawn for obj in engine.level.hazards)
    assert engine.level.goal in drawn
    assert next(obj for obj in engine.level.objects if obj.name == "startFlag") in drawn
    assert all(obj in drawn for obj in engine.level.fruits[1:])


def test_scoreboard_is_visible_updates_and_keeps_rgb_render_headless():
    import pygame
    pygame.display.quit()
    engine = GameEngine(render_mode="rgb_array")
    initial = engine.render()
    assert np.any(np.all(initial[6:70, 850:1044] == (35, 35, 35), axis=-1))
    engine.player.life_count = 1
    engine.player.fruit_count = 12
    updated = engine.render()
    assert not np.array_equal(initial[6:70, 850:1044], updated[6:70, 850:1044])
    assert not pygame.display.get_init()


def test_enemy_hit_shows_hit_sprite_immediately_while_moving():
    engine = GameEngine()
    player = engine.player
    player.rect.topleft = engine.level.hazards[0].rect.topleft
    _, events, _ = engine.step(1)
    assert events["took_hit"]
    assert player.sprite is player.SPRITES["hit_right"][0]
    assert player.animation_count == 1


def test_hit_animation_overrides_movement_and_returns_after_recovery():
    from Game.playerLogic import Player
    for velocity, jump_count in ((-9, 1), (-8, 2), (4, 0), (0, 0)):
        player = Player(192, 640, 50, 50)
        player.direction = "right"
        player.x_vel = 5
        player.y_vel = velocity
        player.jump_count = jump_count
        player.make_hit()
        for _ in range(6):
            player.update_sprite()
            assert any(player.sprite is frame for frame in player.SPRITES["hit_right"])
        assert player.sprite is player.SPRITES["hit_right"][2]
        player.x_vel = 0
        player.y_vel = 0
        for _ in range(61):
            player.update_physics()
        assert not player.hit
        assert not any(player.sprite is frame for frame in player.SPRITES["hit_right"])


def test_observations_expose_fruit_enemy_direction_jump_state_and_goal():
    import pygame
    from Game.gameObject import Fruit, Enemy
    engine = GameEngine()
    player = engine.player
    player.rect.topleft = (192, 640)
    player.is_grounded = True
    player.can_double_jump = False
    engine.level.fruits = [Fruit(player.rect.x + 100, player.rect.y - 80, 64, 64)]
    enemy = Enemy(player.rect.x - 100, player.rect.y, 32, 32)
    enemy.direction = "left"
    engine.level.hazards = [enemy]
    state = engine.get_state_vector()
    assert state[6:8] == [1.0, 0.0]
    assert state[8] > 0 and state[9] < 0 and state[10] == 1
    assert state[11:17] == [0.0] * 6
    assert state[17] < 0 and state[19] < 0 and state[20] == 1
    assert state[21:25] == [0.0] * 4
    assert state[49] > 0
    engine.level.fruits[0].collect()
    assert engine.get_state_vector()[8:17] == [0.0] * 9


def test_terrain_observation_distinguishes_floor_gap_and_raised_platform():
    from Game.gameObject import Block
    engine = GameEngine()
    player = engine.player
    player.rect.topleft = (192, 640)
    # Every sample except +96 has floor; +96 has a platform above the feet.
    x = player.rect.centerx + 96
    engine.level.platforms = [Block(x, 464, 96)]
    state = engine.get_state_vector()
    index = 25 + engine.TERRAIN_OFFSETS.index(96) * 3
    assert state[index] == 0 and state[index + 1] < 0 and state[index + 2] == 1
    engine.level.platforms = [Block(x, player.rect.bottom, 96)]
    state = engine.get_state_vector()
    assert state[index:index + 3] == [1.0, 0.0, 1.0]
    engine.level.platforms = []
    assert engine.get_state_vector()[25:49] == [0.0] * 24


def test_progress_rewards_only_new_furthest_position_and_resets():
    import pytest
    env = PlatformerGameEnv()
    env.reset()
    start = env.game.player.rect.centerx
    _, reward, _, _, info = env.step(1)
    assert reward == pytest.approx(0.01)
    assert info["progress_reward"] == pytest.approx(0.01)
    assert info["progress_pixels"] == 5
    for action in (0, 1, 4):
        _, reward, _, _, info = env.step(action)
        assert reward == 0 and info["progress_reward"] == 0
    env.game.player.is_grounded = True
    _, reward, _, _, info = env.step(2)
    assert reward == 0
    env.reset()
    assert env.furthest_x == start and env.progress_pixels == 0
    assert env.step(1)[1] == pytest.approx(0.01)

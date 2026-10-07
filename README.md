# ReinformentLearningGame

The Gymnasium environment uses the game's original sprite player, mask collisions,
scrolling course, gaps, patrolling enemies, fruit, and trophy. Human gameplay uses
that same engine. Both human input and agents advance the same physics engine.

Install dependencies with `python -m pip install -r requirements.txt`.
Launch human gameplay from the repository root with `python -m Game.main`.

```python
from game_env import PlatformerGameEnv

env = PlatformerGameEnv(render_mode=None, randomize_obstacles=False)
obs, info = env.reset(seed=123)
while True:
    action = env.action_space.sample()  # Replace with your policy's action.
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
```

Actions are `0` left, `1` right, `2` jump, `3` double jump, and `4` idle.
Jump actions preserve horizontal momentum. Observations contain 51 normalized
`float32` values, clipped to [-1, 1]:

| Indices | Features |
| --- | --- |
| 0–5 | Player x/y, velocity x/y, lives, remaining time |
| 6–7 | Grounded and double-jump availability, separately |
| 8–16 | Three nearest uncollected fruits: relative x/y and presence |
| 17–24 | Two nearest guards: relative x/y, patrol velocity and presence |
| 25–48 | Eight terrain samples: floor below, nearest platform height and presence |
| 49–50 | Relative goal x/y |

Terrain is sampled at -384, -192, -96, -32, +32, +96, +192 and +384 pixels
from the player's center. Platform heights are relative to the player's feet.
Object offsets are relative to the player, scaled by viewport width/height;
goal x is scaled by course width. Missing object slots are zero with presence=0.
The nearest objects are ordered by distance. These observations expose geometry
without image processing, but terrain samples remain an approximation of the map.

Models must match the current 51-value observation format.

Base event rewards remain +1 for collecting fruit, -1 for taking damage, +2 for the trophy,
and -2 for falling or timing out. Goal, falling, and zero lives terminate an
episode; timeout truncates it. Each new furthest x position also earns +0.002
per pixel; backtracking does not earn the bonus again. The default limit is 18,000 ticks (300 seconds at
60 FPS), configurable through `max_episode_steps`.

`render_mode=None` runs without initializing a display or limiting FPS.
`render_mode="human"` shows the original assets with a scrolling camera;
`render_mode="rgb_array"` returns an `(800, 1050, 3)` RGB array without a window.
Fixed courses use a deterministic layout; `randomize_obstacles=True` regenerates
the original course's gaps, floating blocks, and fruit using the reset seed.
Call `reset()` before stepping and after each episode ends.

To verify the environment, install pytest and run
`SDL_VIDEODRIVER=dummy python -m pytest -q`.

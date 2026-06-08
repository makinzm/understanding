# Meta Information

- URL: [Craftium: An Extensible Framework for Creating Reinforcement Learning Environments](https://arxiv.org/abs/2407.03969)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Malagón, M., Ceberio, J., and Lozano, J. A. (2024). Craftium: An Extensible Framework for Creating Reinforcement Learning Environments. arXiv preprint arXiv:2407.03969.

# Craftium: An Extensible Framework for Creating Reinforcement Learning Environments

## Abstract

Craftium is a framework built on top of the open-source Minetest game engine and the Gymnasium API, designed for creating and exploring rich 3D visual reinforcement learning (RL) environments. Unlike existing platforms that offer only a fixed set of predefined tasks, Craftium allows practitioners to define fully customized environments tailored to their specific research needs. The framework ships with five pre-built benchmark environments as starting points for development and evaluation.

## 1. Introduction

Most existing RL environments are derived from physics simulators or video games that were not designed with RL research in mind. As a result, they provide only a limited number of predefined tasks with minimal options for customization. Craftium addresses this limitation by exposing the full flexibility of the Minetest voxel engine—supporting arbitrary 3D worlds, custom game logic written in Lua, and a standard Python interface via Gymnasium—so that researchers can build novel environments without forking a game or patching a physics simulator.

**Problem being solved:** RL researchers who need 3D visual environments beyond the fixed task distributions of ALE, ProcGen, or Minecraft-based frameworks face steep engineering barriers. Craftium lowers those barriers by providing a clean separation between the game engine (C++/Lua) and the RL agent interface (Python/Gymnasium).

## 2. Background: Minetest

Minetest is an open-source, voxel-based game engine inspired by Minecraft. Key properties relevant to Craftium:

- Implemented in **C++**, delivering substantially higher performance than Java-based Minecraft.
- Exposes a **Lua scripting API** (called "mods") that controls all game logic, world generation, entities, and events.
- The mod system allows new games, items, terrain generators, and NPC behaviors to be created without modifying the engine itself.

Craftium requires only minimal patches to the Minetest engine—specifically adding TCP communication support and a small set of Lua API extensions—keeping the maintenance burden low.

## 3. The Craftium Framework

### 3.1 Architecture

Craftium consists of three layers:

1. **Modified Minetest engine** — the stock C++ engine with TCP socket communication and extended Lua hooks added.
2. **Craftium Python library** — implements the Gymnasium `Env` interface; handles process management, serialization of observations/actions/rewards over TCP, and episode lifecycle.
3. **Agent code** — standard Gymnasium-compatible RL code using libraries such as Stable-Baselines3, Ray RLlib, CleanRL, or skrl.

### 3.2 Observations

The observation at each timestep is a single RGB image of the agent's first-person camera view.

- Type: $o_t \in \mathbb{R}^{H \times W \times 3}$ where pixel values are in $[0, 255]$.
- Default resolution: $H = W = 64$ pixels (configurable at environment instantiation).

### 3.3 Action Space

The default action space combines discrete keyboard actions and continuous mouse movement:

- **Keyboard actions:** 21 binary keys (e.g., move forward/backward/left/right, jump, dig, place). Each key $k_i \in \{0, 1\}$, where 1 means pressed.
- **Mouse movement:** A continuous tuple $(\Delta x, \Delta y) \in [-1, 1]^2$ controlling camera rotation.
  - $\Delta x < 0$: rotate left; $\Delta x > 0$: rotate right.
  - $\Delta y < 0$: look down; $\Delta y > 0$: look up.
  - $\Delta x = \Delta y = 0$: no rotation.

Two optional **action wrappers** simplify the space:

- `BinaryActionWrapper` — restricts to the 21 keyboard actions only (no mouse).
- `DiscreteActionWrapper` — discretizes mouse movement into four directional moves (left, right, up, down) with a user-definable magnitude, producing a fully discrete action set.

### 3.4 Reward Functions and Termination

Rewards and termination conditions are defined in **Lua** inside a Minetest mod. Two helper functions are provided:

- `set_reward_once(reward_value, reset_value)` — assigns a reward for the current step and resets the internal counter to `reset_value` afterward (preventing the same event from rewarding multiple times).
- `set_termination()` — signals the Python side to terminate the current episode.

Example Lua snippet for a tree-chopping reward:

```lua
minetest.register_on_dignode(function(_pos, node)
  if string.find(node["name"], "tree") then
    set_reward_once(1.0, 0.0)
  end
end)
```

### 3.5 Creating Custom Environments

A custom Craftium environment requires two artifacts:

1. **A game world** — created via Minetest's built-in map generators (procedural or flat), by editing a saved world in-game, or via external tools such as minetestmapper.
2. **A Lua mod** — a directory containing at minimum a `mod.conf` configuration file and an `init.lua` script that registers reward callbacks and termination conditions.

The Python side instantiates the environment as:

```python
import gymnasium as gym
import craftium

env = gym.make("craftium/ChopTree-v0")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

### 3.6 Provided Benchmark Environments

| Environment | Observation | Max Steps | Reward | Termination |
|---|---|---|---|---|
| Chop Tree | 64×64 RGB | 500 | +1 per tree chopped | Truncation at 500 steps |
| Room | 64×64 RGB | 500 | −1 per timestep | Reach red block |
| Small Room | 64×64 RGB | 200 | −1 per timestep | Reach red block |
| Speleo | 64×64 RGB | 500 | Negative Y-axis position (descend deeper = higher reward) | Truncation at 500 steps |
| Spiders Attack | 64×64 RGB | 2000 | +1 per spider killed | Player death or 2000 steps |

> [!NOTE]
> The Room and Small Room environments test navigation to a goal object; the negative per-step reward encourages the agent to reach the target as quickly as possible.

> [!TIP]
> The Speleo environment is noteworthy as a dense-reward exploration task: the agent must descend in a cave to maximize reward, rewarding depth rather than any discrete event.

## 4. Related Work

Craftium is positioned among several families of RL environment frameworks:

| Category | Examples | Limitation vs. Craftium |
|---|---|---|
| Procedural generation | ProcGen | 2D only; no user-defined reward logic |
| Domain-specific languages | VizDoom, MiniHack | Narrow domain (FPS / NetHack); limited 3D flexibility |
| Python-based grid worlds | MiniGrid, MiniWorld, Griddly | Lack photorealistic visual complexity |
| Minecraft-based | Malmo, MineRL, MineDojo | Require proprietary Java Minecraft; limited custom reward API |
| Simulated driving | CARLA | Single domain (autonomous driving) |
| Robotics simulators | MuJoCo, panda-gym | Physics focus; no open-world 3D navigation |

> [!IMPORTANT]
> Craftium is distinct because it combines (a) full 3D visual richness, (b) complete customizability of tasks and worlds via Lua, and (c) open-source engine with no dependency on proprietary software.

## 5. Conclusion

Craftium fills the gap between overly simple grid-world environments and complex, hard-to-modify 3D games. The framework is fully open source with comprehensive documentation, available at https://github.com/mikelma/craftium/. Stated future directions include:

- Creating additional diverse benchmark environments.
- Implementing multi-agent support via the Petting Zoo API.

# Experiments

- **Dataset:** No external dataset. The paper evaluates the five provided environments qualitatively (visual screenshots and environment descriptions). No quantitative benchmark results (reward curves, sample efficiency numbers) are reported in the paper; Craftium is presented as an infrastructure contribution, not an algorithmic one.
- **Hardware:** Not specified.
- **Optimizer:** Not applicable (no training experiment reported).
- **Results:** The paper demonstrates that each of the five environments is functional and Gymnasium-compatible. The primary evidence is the existence of the framework itself and informal observations about environment variety (navigation, combat, exploration, tree-chopping).

# References

1. Abi Raad et al. (2024). Scaling instructable agents across many simulated worlds. arXiv:2404.10179.
2. Bamford, C., Huang, S., and Lucas, S. (2020). Griddly: A platform for AI research in games. arXiv:2011.06363.
3. Bauer, J. et al. (2023). Human-timescale adaptation in an open-ended task space. ICML 2023, PMLR vol. 202, pp. 1887–1935.
4. Beattie, C. et al. (2016). DeepMind Lab. arXiv:1612.03801.
5. Bellemare, M.G., Naddaf, Y., Veness, J., and Bowling, M. (2013). The Arcade Learning Environment: An evaluation platform for general agents. JAIR, 47:253–279.
6. Carpentier, J., Budhiraja, R., and Mansard, N. (2021). Proximal and sparse resolution of constrained dynamic equations. RSS 2021.
7. Chevalier-Boisvert, M. et al. (2023). Minigrid & Miniworld: Modular & customizable RL environments for goal-oriented tasks. arXiv:2306.13831.
8. Cobbe, K. et al. (2020). Leveraging procedural generation to benchmark reinforcement learning. arXiv:1912.01588.
9. Dosovitskiy, A. et al. (2017). CARLA: An open urban driving simulator. CoRL 2017, pp. 1–16.
10. Fan, L. et al. (2022). MineDojo: Building open-ended embodied agents with internet-scale knowledge. NeurIPS 2022, 35:18343–18362.
11. Gallouédec, Q. et al. (2021). panda-gym: Open-source goal-conditioned environments for robotic learning. NeurIPS 4th Robot Learning Workshop.
12. Guss, W.H. et al. (2019). MineRL: A large-scale dataset of Minecraft demonstrations. arXiv:1907.13440.
13. Huang, S. et al. (2022). CleanRL: High-quality single-file implementations of deep RL algorithms. JMLR, 23(274):1–18.
14. Johnson, M. et al. (2016). The Malmo platform for artificial intelligence experimentation. IJCAI 2016, pp. 4246–4247.
15. Küttler, H. et al. (2020). The NetHack Learning Environment. NeurIPS 2020, 33:7671–7684.
16. Machado, M.C. et al. (2018). Revisiting the Arcade Learning Environment: Evaluation protocols and open problems for general agents. JAIR, 61:523–562.
17. Moritz, P. et al. (2018). Ray: A distributed framework for emerging AI applications. OSDI 2018, pp. 561–577.
18. Prasanna, S. et al. (2024). Dreaming of many worlds: Learning contextual world models aids zero-shot generalization. arXiv:2403.10967.
19. Raffin, A. et al. (2021). Stable-Baselines3: Reliable reinforcement learning implementations. JMLR, 22(268):1–8.
20. Samvelyan, M. et al. (2021). MiniHack the planet: A sandbox for open-ended reinforcement learning research. NeurIPS Datasets and Benchmarks Track 2021.
21. Serrano-Muñoz, A. et al. (2023). skrl: Modular and flexible library for reinforcement learning. JMLR, 24(254):1–9.
22. Todorov, E., Erez, E., and Tassa, Y. (2012). MuJoCo: A physics engine for model-based control. IEEE/RSJ IROS 2012, pp. 5026–5033.
23. Wydmuch, M., Kempka, M., and Jaśkowski, W. (2019). ViZDoom competitions: Playing Doom from pixels. IEEE Transactions on Games, 11(3):248–259.

# Meta Information

- URL: [Reinforcement Learning Meets Visual Odometry](https://arxiv.org/abs/2407.15626)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Messikommer, N., Cioffi, G., Gehrig, M., & Scaramuzza, D. (2024). Reinforcement Learning Meets Visual Odometry. ECCV 2024. University of Zurich.

# Overview

Visual Odometry (VO) estimates camera pose from a sequence of images by tracking keypoints across frames. Classical VO systems (e.g., DSO, SVO) contain numerous heuristic design choices—keyframe selection thresholds, feature grid sizes—that require laborious hand-tuning per environment. This paper reformulates VO as a **sequential decision-making problem** and trains an RL agent to replace these hand-crafted heuristics, learning adaptive policies that generalize across diverse scenes and motion patterns without human re-tuning.

**Applicability**: Robotics, autonomous vehicles, and AR/VR systems where reliable and efficient camera pose estimation is required across varied environments (indoor rooms, outdoor aerial, low-light, high-speed motion). Particularly useful when deploying VO systems in new domains without access to domain experts for re-tuning.

# Problem Formulation as MDP

The VO pipeline is cast as a Markov Decision Process $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

- **State** $s_t \in \mathcal{S}$: The current set of tracked keypoints, their positions, descriptors, and internal VO pipeline mode (tracking, relocalization, or initialization).
- **Action** $a_t \in \mathcal{A}$: A multi-discrete action consisting of:
  1. $a_{\text{keyframe}} \in \{0, 1\}$: whether to select the current frame as a keyframe.
  2. $a_{\text{grid}} \in \{1, 2, \ldots, K\}$: grid size for new feature extraction (controls number of keypoints detected).
- **Transition** $\mathcal{P}$: Determined by the VO system executing the action on the next image frame.
- **Reward** $r_t$: A differentiable signal combining pose accuracy and runtime cost (detailed below).
- **Discount** $\gamma$: Standard exponential discounting for future rewards.

> [!NOTE]
> The agent does not replace the core VO front-end (tracking, initialization) but instead controls the *meta-decisions* that govern how the front-end operates, acting as a learned hyperparameter controller.

# Agent Architecture

## Input Representation

The agent receives $N$ keypoints as input at each timestep. Because $N$ varies with scene complexity and grid size, the agent must process variable-length inputs:

- Each keypoint $k_i$ is represented as a vector in $\mathbb{R}^{d_k}$ encoding 2D image coordinates, optical flow displacement, and descriptor features.
- Input tensor: $X \in \mathbb{R}^{N \times d_k}$ (variable $N$ per step).

## Perceiver Encoder

To handle variable-length inputs, a **Perceiver architecture** is used:

1. Initialize $M$ learnable query tokens $Q \in \mathbb{R}^{M \times d}$ (fixed size, $M \ll N$).
2. Apply cross-attention between $Q$ (queries) and $X$ (keys/values) to compress the variable-length input into fixed-size latent $Z \in \mathbb{R}^{M \times d}$.
3. Apply $L$ layers of self-attention over $Z$ to model inter-token dependencies.
4. Flatten $Z$ to a fixed vector $z \in \mathbb{R}^{M \cdot d}$.

This produces a fixed-size representation regardless of input keypoint count $N$, critical for stable RL training.

> [!TIP]
> The Perceiver architecture: [Perceiver: General Perception with Iterative Attention (arXiv:2103.03206)](https://arxiv.org/abs/2103.03206)

## Policy Head

From the latent $z \in \mathbb{R}^{M \cdot d}$, separate MLP heads produce:
- Keyframe logits $\in \mathbb{R}^{2}$ → softmax → $p(a_{\text{keyframe}})$
- Grid-size logits $\in \mathbb{R}^{K}$ → softmax → $p(a_{\text{grid}})$

The two distributions are independent (multi-discrete action space). Total network parameters: approximately 296K.

**Pseudocode: Agent forward pass**
```
Input: keypoints X ∈ R^{N × d_k}
1. Q ← learned queries ∈ R^{M × d}
2. Z ← CrossAttention(Q, X, X)    # Z ∈ R^{M × d}
3. for l in 1..L:
     Z ← SelfAttention(Z)
4. z ← Flatten(Z)                  # z ∈ R^{M·d}
5. a_kf ~ Categorical(Softmax(MLP_kf(z)))
6. a_grid ~ Categorical(Softmax(MLP_grid(z)))
Output: (a_kf, a_grid)
```

# Reward Function

The reward at step $t$ combines pose accuracy and keyframe overhead:

$$r_t = \lambda_1 \cdot \max(-1,\ 0.2 - e_{\text{tran},t}) - \lambda_2 \cdot a_{\text{keyframe},t}$$

where:
- $e_{\text{tran},t}$: translation error (meters) computed via sliding-window alignment of estimated poses against ground truth.
- $\max(-1, \cdot)$: clips minimum reward to $-1$ to prevent extreme penalty.
- $\lambda_2 \cdot a_{\text{keyframe},t}$: penalizes creating a keyframe (encourages sparsity/runtime efficiency).
- $\lambda_1, \lambda_2$: weighting hyperparameters.

> [!IMPORTANT]
> Ground-truth poses are used **only during reward computation in training**, not during inference. The agent observes only raw keypoint data during deployment.

# Training with PPO and Privileged Critic

## Algorithm: PPO with Privileged Critic

**Proximal Policy Optimization (PPO)** is used to optimize the policy:

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}_t \left[ \min\left( \rho_t \hat{A}_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where $\rho_t = \pi_\theta(a_t|s_t) / \pi_{\theta_\text{old}}(a_t|s_t)$ is the probability ratio and $\hat{A}_t$ is the advantage estimate.

**Privileged Critic**: The value function (critic) receives ground-truth pose information during training (unavailable at test time) to produce lower-variance advantage estimates. The actor policy does not receive this privileged information and uses only raw keypoints.

**Masked Rollout Buffer**: VO systems cycle through modes (tracking, relocalization, initialization). Actions are only meaningful in tracking mode. The masked buffer records transitions only during tracking mode, discarding non-tracking steps to prevent misleading gradient updates.

**Pseudocode: Training loop**
```
Initialize policy π_θ, privileged critic V_φ
For each training epoch:
  For each sequence s in training_set:
    Reset VO system
    For each frame t:
      obs_t ← get_keypoints()
      a_t ~ π_θ(obs_t)
      Apply a_t to VO system
      r_t ← compute_reward(GT_pose_t, a_t)
      if in_tracking_mode:
        buffer.add(obs_t, a_t, r_t, V_φ(obs_t, GT_pose_t))
  Update π_θ and V_φ via PPO on buffer
```

# Differences from Related Methods

| Aspect | Classical VO (DSO/SVO) | Learning-Based VO (DROID-VO) | This Work (RL VO) |
|--------|------------------------|------------------------------|-------------------|
| Pose estimation | Geometric optimization | End-to-end neural network | Classical geometric VO |
| Hyperparameter tuning | Manual, per-environment | Fixed after training | Adaptive via RL agent |
| Interpretability | High | Low | High (agent acts on keypoints) |
| Generalization | Low (requires tuning) | Moderate | High (adaptive policy) |
| Compute | Low | High | Low + lightweight agent |
| Ground truth required | No | Yes (training only) | Yes (training reward only) |

# Experiments

- **Datasets**:
  - Training: TartanAir (337 sequences, 279,987 images, synthetic with diverse environments and conditions)
  - Evaluation: EuRoC MAV Dataset (11 sequences, indoor, varying illumination/speed), TUM-RGBD (13 sequences, handheld indoor), KITTI Odometry (11 sequences, outdoor driving)
- **Hardware**: Not explicitly specified; standard GPU training inferred
- **Optimizer**: Adam (actor and critic networks); PPO from Stable Baselines3
- **VO Backends**: DSO (Direct Sparse Odometry), SVO (Semi-direct Visual Odometry), ORB-SLAM3

## Key Results

- **EuRoC (RL DSO vs DSO)**: 19% reduction in average Absolute Trajectory Error (ATE); sequence V103 completed where DSO fails
- **TUM-RGBD (RL SVO vs SVO)**: Average ATE 0.422m vs 0.471m; 2 additional sequences completed
- **Runtime (RL SVO)**: 7.40ms/frame vs 9.06ms baseline (agent adds 1.75ms but reduces keyframe count)
- **Generalization to ORB-SLAM3**: Improvements on EuRoC and TUM-RGBD without retraining the agent architecture

## Ablation Findings (Table 5)

- Removing variable-length Perceiver encoder (fixed-size input) → failure on multiple sequences
- Removing privileged critic → increased training variance, lower final performance
- Removing grid-size action → reduced accuracy, fewer completed sequences
- Removing keyframe action → increased runtime without proportional accuracy gain

# Limitations

- Sequences with motion patterns absent from TartanAir training distribution (e.g., static camera) degrade performance, as the policy has no experience with such observations
- On sequences where the baseline VO itself fails, the RL agent cannot recover (agent controls meta-decisions, not core tracking)
- Training requires synchronized ground-truth pose labels and VO system integration per new backend

# Summary

This work demonstrates that classical VO pipelines can be improved by replacing manual hyperparameter heuristics with a learned RL policy. The Perceiver-based agent efficiently compresses variable-length keypoint sets into fixed-size representations, enabling stable PPO training. The privileged critic technique and masked rollout buffer address VO-specific training challenges. The method achieves consistent improvements over DSO and SVO baselines across EuRoC, TUM-RGBD, and generalizes to ORB-SLAM3 without retraining, confirming that learned controllers can effectively augment classical geometric systems.

# Meta Information

- URL: [Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications](https://arxiv.org/abs/2510.07077)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Kawaharazuka, K., Oh, J., Yamada, J., Posner, I., & Zhu, Y. (2025). Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications. IEEE Access, 13, 162467–162504. https://doi.org/10.1109/ACCESS.2025.3609980

# Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications

This survey provides a full-stack review of Vision-Language-Action (VLA) models — systems that jointly process visual observations and natural language instructions to directly generate robot control commands. The scope explicitly excludes approaches that use vision/language only for high-level task planning that selects from a fixed set of pre-trained skills. The intended audience is robotics researchers and practitioners seeking to deploy generalizable policies to real-world robot systems across diverse embodiments, tasks, and environments.

## Challenges

### Data Requirements and Scarcity

Training VLAs requires large-scale datasets containing synchronized vision, language, and action labels. This creates two bottlenecks:
1. Web-scale vision-language datasets (e.g., COCO, LAION) lack action grounding for motor control.
2. High-quality robot demonstration datasets collected via teleoperation are expensive and narrow in task distribution.

Latent action learning from unlabeled human video (Ego4D, EPIC-KITCHENS) addresses the second bottleneck by extracting action representations without explicit action labels.

### Embodiment Transfer

Robots differ in morphology (arms only, wheeled, legged, humanoid), sensor configurations, and degrees of freedom. Transferring policies across embodiments requires mapping heterogeneous action and proprioceptive spaces. Using human motion data adds further complexity since human body kinematics differ substantially from robot-executable commands.

### Computational and Training Cost

Transformer-based VLAs scale poorly with input sequence length and modality dimensionality. Real-world deployment on resource-constrained hardware imposes latency and memory constraints. Approaches such as LoRA fine-tuning, 1.58-bit quantization (BitVLA), early exit inference (DeeR-VLA), and token caching (VLA-Cache) address this.

## Architectural Evolution

VLA design has undergone four major transitions:

| Era | Representative Models | Action Head | Key Limitation |
|---|---|---|---|
| CNN-based (pre-2022) | CLIPort | Transporter Network | Poor modality unification, poor scaling |
| Transformer + discrete tokens (2022) | Gato, VIMA, RT-1 | Autoregressive / Non-autoregressive | Coarse discretization, slow generation |
| VLM backbone + discrete tokens (2023) | RT-2, RT-X, OpenVLA | Autoregressive | Lacks smooth continuous control |
| VLM + diffusion/flow matching (2024–2025) | Octo, π₀, GR00T N1 | Diffusion / Flow matching | Current SOTA; high compute cost |

**RT-1** was the first VLA to unify a broad range of robotic tasks. It processes image sequences through EfficientNet, extracts language features via Universal Sentence Encoder, and uses TokenLearner to compress visual tokens. Actions are predicted non-autoregressively (all action dimensions in parallel), enabling real-time control.

**RT-2** fine-tunes pre-trained VLM backbones (PaLM-E, PaLI-X) to output discretized action tokens, inheriting commonsense knowledge from internet-scale pre-training. This enables generalization to novel environments without additional task-specific data.

**π₀ (pi-zero)** uses PaliGemma (SigLIP vision encoder + Gemma language model) as the backbone and a flow-matching action expert as the head. Flow matching generates smooth continuous trajectories and achieves control rates up to 50 Hz — significantly faster than diffusion-based alternatives.

**GR00T N1** represents the current state-of-the-art, integrating latent action representations (learned from video without explicit labels), a diffusion transformer low-level policy, and a flow-matching head into a unified hierarchical architecture.

## Architectures and Building Blocks

### Sensorimotor Model Variants

Seven architectural patterns are identified by how they combine the VLM backbone with an action generation mechanism:

1. **Transformer + Discrete Action Token** (RT-1, Gato, VIMA): All modalities tokenized; transformer predicts discretized actions (256 bins per dimension) via cross-entropy. RT-1 uses TokenLearner to compress image tokens before action prediction.

2. **Transformer + Diffusion Action Head** (Octo): Image and language tokens processed through a transformer; a diffusion action head conditioned on a readout token generates continuous actions. Input: $x \in \mathbb{R}^{T \times d}$ (token sequence); output: $a \in \mathbb{R}^{H \times d_a}$ (action chunk).

3. **Diffusion Transformer** (RDT-1B): Diffusion integrated into the transformer backbone via cross-attention with vision-language query tokens. Fully generative end-to-end architecture.

4. **VLM + Discrete Action Token** (RT-2, OpenVLA): Pre-trained VLM fine-tuned to append action tokens to the output vocabulary. Input: image(s) + language instruction; output: action token sequence decoded as robot commands.

5. **VLM + Diffusion Action Head** (Diffusion-VLA, ChatVLA): VLM handles perception; diffusion head generates stable continuous actions conditioned on VLM output.

6. **VLM + Flow Matching Action Head** (π₀): PaliGemma processes image $x \in \mathbb{R}^{H \times W \times 3}$ and language instruction $l$; flow-matching action expert generates action chunk $a \in \mathbb{R}^{H \times d_a}$ conditioned on VLM latent. Achieves 50 Hz control frequency.

7. **VLM + Diffusion Transformer** (GR00T N1): VLM serves as high-level policy; diffusion transformer operates as low-level policy conditioned on VLM latent actions.

### World Models

World models predict future visual observations to guide action generation without requiring explicit action labels.

- **UniPi**: Generates a future video sequence given language instruction, then uses an inverse dynamics model to extract robot actions from consecutive predicted frames.
- **LAPA (Latent Action Pre-training for Any Robot)**: Given current frame $x_t \in \mathbb{R}^{H \times W \times 3}$ and future frame $x_{t+H}$, VQ-VAE encodes the image difference into discrete latent action tokens $z_t$. This allows training on unlabeled human video (Ego4D, EPIC-KITCHENS) without explicit action labels.
- **GR-1 / GR-2**: Jointly predict both robot actions and future observation frames as an auxiliary reconstruction objective, using the prediction loss to improve representation learning.

### Affordance-Based Models

Affordance models predict where and how an action should be applied:
- **CLIPort**: Uses CLIP features to predict pick-and-place affordance maps over the workspace.
- **VoxPoser**: Uses GPT-4 and open-vocabulary detectors to generate 3D value maps; Model Predictive Control (MPC) translates these affordance maps into executable control.
- **VRB**: Extracts contact points and hand trajectories from human demonstration videos (EPIC-KITCHENS), then uses these affordances to supervise robot policies without robot-specific labels.

### Data Modalities

**Vision:**
- Encoders: ResNet → ViT → CLIP → SigLIP / DINOv2 (current dominant)
- Compression: TokenLearner (selects $k$ informative tokens from $n$ input tokens), Perceiver Resampler (fixed $m$-length output regardless of input length)
- Discretization: VQ-GAN / VQ-VAE convert continuous image patches to integer tokens

**Language:**
- Encoders: Universal Sentence Encoder (RT-1) → CLIP text encoder → LLM tokenizers (LLaMA, Gemma, Qwen2)
- VLM-based architectures integrate language through frozen or LoRA-fine-tuned LLM components

**Action — four representations:**
1. **Binned discrete tokens**: Each action dimension → 256-bin integer; loss: cross-entropy
2. **Continuous MLP output**: Direct regression; loss: L1 or L2
3. **Diffusion / flow matching**: Noise → action via iterative denoising or ODE integration; enables smooth trajectories
4. **Latent action tokens**: Learned from video (VQ-VAE on frame differences); no explicit action labels required

**3D and additional modalities:**
- Depth: encoded identically to RGB via ViT, or estimated monocularly
- Point clouds: tokenized via PointNet variants
- Tactile: encoded with ViT on sensor images
- Force-torque: integrated through MoE fusion modules
- Audio: spectrograms processed as images

### Cross-Embodiment Unification

- **CrossFormer**: Modality-specific tokenizers allow heterogeneous sensor inputs (images, proprioception) from different robots to be processed in a unified transformer.
- **UniAct**: Proposes a Universal Action Space implemented as a discrete codebook; robot-specific decoders map universal tokens to embodiment-specific commands.

### Emerging Techniques

**Hierarchical architectures** decompose control into two levels:
- High-level: Generates language or latent action representations that describe the intended behavior
- Low-level: Executes smooth continuous control conditioned on high-level tokens

Examples:
- **RT-H**: Introduces "language motion" — intermediate natural language descriptions of motion that bridge task-level instructions and low-level actions
- **π₀.5**: High-level VLM generates discrete tokens; flow-matching action expert executes
- **GR00T N1**: Latent action VLM (high-level) + diffusion transformer (low-level)

**Chain-of-Thought (CoT) reasoning:**
- **ECoT (Embodied CoT)**: Before generating actions, the model autoregressively predicts intermediate representations — task description → object locations → gripper position → motion plan — then produces final action tokens. Input: image + instruction; output: CoT reasoning tokens + action tokens.
- **CoT-VLA**: Generates subgoal images as visual reasoning steps before producing final actions.

## Training Strategy

### Supervised Learning

VLAs train on image-language-action triplets $(o_t, l, a_t)$ where $o_t \in \mathbb{R}^{H \times W \times 3}$ is the observation, $l$ is the language instruction, and $a_t \in \mathbb{R}^{d_a}$ is the action (or action chunk $a_{t:t+H}$).

**In-context learning (ICRT)**: 1–3 teleoperated demonstrations are prepended as prompt tokens, enabling zero-shot execution of new task instances without gradient updates.

### Self-Supervised Learning

- **Contrastive alignment**: Current and future state representations aligned in shared latent space
- **Visual pre-training**: MAE, CLIP, DINOv2 provide foundational visual representations
- **Latent action learning (LAPA)**: VQ-VAE applied to image differences $(x_t, x_{t+H})$ produces discrete latent codes $z_t$; the model is trained to predict $z_t$ without ground-truth action labels, enabling training on large-scale unlabeled human video

### Reinforcement Learning Integration

**Direct improvement:**
- **iRe-VLA**: Three-phase cycle — (1) supervised fine-tuning on demonstrations, (2) online RL using binary success/failure rewards, (3) additional SFT on RL-generated successes
- **DSRL**: Instead of updating VLA weights, learns a distribution over the latent noise used in flow matching. Improves π₀ task success rate from ~20% to ~100% using only 10,000 environment samples.

**Hierarchical RL + VLA:**
- **Humanoid-VLA**: VLA handles high-level decision-making; RL-trained low-level policies execute motor control
- **SLIM**: RL-trained teacher policies generate high-quality demonstrations that are distilled into VLA student models

### Training Stages and Backbone Choices

**Pre-training backbones (in historical order):**
- PaLM-E / PaLI-X (RT-2)
- Prismatic VLM based on LLaMA 2 (OpenVLA)
- PaliGemma with SigLIP vision encoder (π₀, π₀.5)
- Qwen2.5-VL (emerging)
- Gemini 2.0 (robotics variants)

**Gradient insulation**: Action head parameters are randomly initialized while backbone parameters are pre-trained. Without gradient insulation, large gradients from the action head corrupt the pre-trained representations during early training.

**Backbone adaptation trade-offs:**

| Strategy | Memory | Performance | Notes |
|---|---|---|---|
| Frozen backbone | Low (consumer GPU) | Lower task-specific | No domain adaptation |
| Full fine-tuning | High | Highest task-specific | Requires large compute |
| LoRA | Medium | Competitive | Selective rank-$r$ updates to attention matrices |
| Quantization (BitVLA) | Very low | Slight degradation | 1.58-bit compression via distillation |

### Inference Optimization

- **Real-Time Chunking**: While executing the current action chunk $a_{t:t+H}$, the model simultaneously generates the next chunk $a_{t+H:t+2H}$ in parallel, reducing latency.
- **DeeR-VLA (Early Exit)**: Monitors consecutive transformer layer predictions; if intermediate outputs converge (change below threshold), remaining layers are skipped. Saves compute on visually simple steps.
- **VLA-Cache**: Identifies static tokens (low change between timesteps) and reuses cached key-value states from the previous step, avoiding redundant attention computations for background regions.

## Datasets

### Robot Demonstration Datasets

| Dataset | Scale | Embodiments | Key Use |
|---|---|---|---|
| RT-1 Dataset | 700 tasks, 130,000 episodes | 13 RT-1 robots | Established VLA feasibility |
| Open-X Embodiment (OXE) | Multi-embodiment trajectories | Multiple robots | First multi-embodiment benchmark; trains OpenVLA |

OXE uses a standardized format: single RGB camera input, natural language task instruction, 7-DoF action vector $a \in \mathbb{R}^7$ (6-DoF end-effector delta + 1-DoF gripper).

### Human Video Datasets (for Latent Action Learning)

| Dataset | Description | Use in VLAs |
|---|---|---|
| Ego4D | Large-scale egocentric video with hand-object interactions | Affordance learning, latent action extraction (LAPA) |
| EPIC-KITCHENS | Egocentric kitchen manipulation videos | Affordance transfer (VRB), latent action learning |

### Web-Scale Vision-Language Datasets

Used for backbone pre-training (language and vision alignment only; no action grounding):
- COCO Captions, LAION, ImageNet

### Simulation

- **COSMOS World Model**: Generates realistic visual observations from synthetic trajectories, enabling scaling of training data without physical collection.

### Data Collection Hardware

- **ALOHA**: Bilateral teleoperation system pairing a WidowX 250 leader arm with a ViperX-300 follower arm. Enables precise bimanual manipulation at low cost.
- **Mobile ALOHA**: Extends ALOHA with a mobile base for whole-body navigation + manipulation tasks.

## Experiments

- Dataset: RT-1 Dataset (700 tasks, 130,000 episodes); Open-X Embodiment (multi-embodiment); Ego4D and EPIC-KITCHENS (human video); simulation benchmarks
- Hardware: Evaluated across single-arm manipulators (WidowX, Franka, ViperX), dual-arm (ALOHA), wheeled mobile robots, legged robots, and humanoids
- Key quantitative results:
  - **DSRL on π₀**: task success improves from ~20% (baseline π₀) to ~100% using 10,000 samples by learning latent noise distributions instead of updating model weights
  - **π₀ flow matching**: achieves 50 Hz real-time control, compared to slower diffusion-based alternatives
  - **BitVLA**: achieves 1.58-bit compression with minor performance degradation, enabling deployment on resource-constrained hardware

## Comparison with Related Surveys

Previous VLA reviews focused narrowly on specific architectural paradigms or excluded hardware considerations. This survey differentiates itself by:
1. **Full-stack coverage**: Includes robot platforms, sensors, actuators, and data collection hardware alongside software architecture
2. **Precise scope definition**: Explicitly excludes planning-only systems that select from pre-defined skills (e.g., SayCan)
3. **Actionable practitioner guidance**: Section VIII provides concrete recommendations for architecture selection, data collection, compute budgets, and deployment
4. **Historical coherence**: Traces each architectural decision to its predecessor, explaining why each transition occurred

## Future Directions

- Latent action learning from large-scale unlabeled human video corpora (scaling without teleoperation)
- Universal action spaces and modality-specific tokenizers for better embodiment transfer
- World models for planning and synthetic data augmentation
- Efficient on-device deployment: quantization, early exit, token caching
- Richer multimodal integration: tactile, audio, force-torque, 3D point clouds
- Chain-of-Thought and hierarchical planning as scalable paths to complex task generalization

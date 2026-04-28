# Meta Information

- URL: [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952)
- LICENSE: [arXiv.org - Non-exclusive license to distribute](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html)
- Reference: Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Muller, J., Penna, J., & Rombach, R. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. arXiv:2307.01952.

# SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis

SDXL is a latent diffusion model (LDM) for text-to-image synthesis that addresses the visual quality shortcomings of earlier Stable Diffusion (SD 1.x, SD 2.x) models. It introduces a substantially enlarged UNet backbone, dual text encoders, three novel micro-conditioning schemes, and an optional refinement stage model. The result is an open-weights, state-of-the-art text-to-image generator that outperforms prior SD versions by a significant margin and is competitive with proprietary commercial systems.

**Who uses this**: Researchers, artists, and developers building text-to-image applications who require high-resolution (1024×1024) outputs with good composition, aesthetics, and fine detail.

**When**: When generating single or multi-aspect-ratio images from text prompts at approximately $1024^2$ pixel area.

**Where**: Any environment supporting PyTorch + diffusers; pre-trained weights are publicly available on Hugging Face.

## Architecture Overview

SDXL maintains the latent diffusion model (LDM) framework from Rombach et al. (2021): a VAE encodes images into a compressed latent space, and a conditional UNet denoises latents guided by text. The major architectural changes relative to SD 2.x are:

| Component | SD 1.x / 2.x | SDXL |
|---|---|---|
| UNet parameters | 860M | 2.6B |
| Text encoder | CLIP ViT-L (768-d) | CLIP ViT-L + OpenCLIP ViT-bigG (2048-d) |
| Transformer block distribution | Uniform | Heterogeneous: `[0, 2, 10]` |
| Pooled embedding pathway | No | Yes (via timestep embedding) |

### UNet Input/Output

- **Input**: Noisy latent $\mathbf{z}_t \in \mathbb{R}^{b \times 4 \times h/8 \times w/8}$ (VAE spatial compression factor 8), noise level $\sigma_t$, conditioning vector $\mathbf{c}$.
- **Output**: Denoised latent estimate $D_{\boldsymbol{\theta}}(\mathbf{z}_t; \sigma_t, \mathbf{c}) \in \mathbb{R}^{b \times 4 \times h/8 \times w/8}$.

### Dual Text Encoders

Two text encoders are used in combination:

1. **CLIP ViT-L**: produces token embeddings of dimension 768 per token.
2. **OpenCLIP ViT-bigG**: produces token embeddings of dimension 1280 per token; its pooled output (1280-d) is additionally fed into the UNet via the timestep embedding pathway.

The two sets of token embeddings are concatenated along the channel axis, yielding a joint context tensor $\mathbf{c}_\text{text} \in \mathbb{R}^{b \times T \times 2048}$ where $T$ is sequence length. This doubles the text embedding dimensionality relative to SD 2.x (1024-d) and nearly triples it relative to SD 1.x (768-d), enabling richer text conditioning.

### Heterogeneous Transformer Block Distribution

Prior SD versions distributed transformer blocks uniformly across UNet levels. SDXL concentrates them at the lowest-resolution (most semantically rich) level:

| UNet Level | Spatial Resolution | Transformer Blocks |
|---|---|---|
| Level 1 (highest) | $h/8 \times w/8$ | 0 |
| Level 2 | $h/16 \times w/16$ | 2 |
| Level 3 (lowest) | $h/32 \times w/32$ | 10 |

Concentrating attention blocks at the lowest resolution is computationally efficient (attention complexity scales quadratically with spatial size) while maximizing semantic processing capacity.

## Diffusion Formulation

SDXL uses denoising score matching in continuous time. The denoiser $D_{\boldsymbol{\theta}}$ is trained to minimize:

```math
\begin{align}
  \mathcal{L} = \mathbb{E}_{(\mathbf{x}_0,\mathbf{c})\sim p_\text{data},\,(\sigma,\mathbf{n})\sim p(\sigma,\mathbf{n})}
  \left[\lambda_\sigma \|D_{\boldsymbol{\theta}}(\mathbf{x}_0+\mathbf{n};\,\sigma,\mathbf{c}) - \mathbf{x}_0\|_2^2\right]
\end{align}
```

where $\mathbf{x}_0$ is the clean latent, $\mathbf{n} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$ is Gaussian noise, $\sigma$ is the noise level drawn from $p(\sigma)$, and $\lambda_\sigma$ is a noise-level weighting function.

**Classifier-Free Guidance (CFG)** interpolates between conditional and unconditional denoising at inference time:

```math
\begin{align}
  D^w(\mathbf{x};\sigma,\mathbf{c}) = (1+w)\,D_{\boldsymbol{\theta}}(\mathbf{x};\sigma,\mathbf{c}) - w\,D_{\boldsymbol{\theta}}(\mathbf{x};\sigma)
\end{align}
```

where $w \geq 0$ is the guidance scale and $D_{\boldsymbol{\theta}}(\mathbf{x};\sigma)$ is the unconditional denoiser (trained by randomly dropping $\mathbf{c}$ during training). Larger $w$ increases adherence to the text prompt at the cost of diversity.

## Micro-Conditioning Schemes

Three new conditioning variables are embedded via Fourier feature embeddings and injected into the UNet alongside the timestep embedding $t$.

### Size Conditioning

Without a resolution filter, 39% of training images (those below 256px on the short side) would normally be discarded. Instead, the model is conditioned on each image's original resolution:

```math
\begin{align}
  \mathbf{c}_\text{size} = (h_\text{original},\, w_\text{original})
\end{align}
```

During training, $\mathbf{c}_\text{size}$ equals the actual image resolution. At inference, the user sets $\mathbf{c}_\text{size} = (1024, 1024)$ to signal that the model should behave as though generating from a high-quality $1024 \times 1024$ image, regardless of whether the actual output resolution matches.

> [!NOTE]
> "Training with size conditioning improves FID from 39.76 → 36.53 and IS from 211.50 → 215.34 on ImageNet class-conditional generation."

### Crop Conditioning

Random cropping during training causes objects to appear truncated or off-center in generated images. The model is conditioned on the crop offset applied during training:

```math
\begin{align}
  \mathbf{c}_\text{crop} = (c_\text{top},\, c_\text{left})
\end{align}
```

Crop coordinates $(c_\text{top}, c_\text{left})$ are sampled uniformly from the valid crop range for each training image. At inference, setting $\mathbf{c}_\text{crop} = (0, 0)$ produces well-composed, centered outputs.

### Multi-Aspect Ratio Training

Rather than training on a single square resolution, SDXL uses a set of target resolutions with varying aspect ratios, all maintaining approximately $1024^2$ total pixels. Heights and widths are constrained to multiples of 64. Example buckets:

| Aspect Ratio | Height | Width |
|---|---|---|
| 0.25 (portrait) | 512 | 2048 |
| 0.57 | 768 | 1344 |
| 1.0 (square) | 1024 | 1024 |
| 1.75 | 1344 | 768 |
| 4.0 (landscape) | 2048 | 512 |

Training alternates between buckets, with each batch drawing from a single bucket. The model is additionally conditioned on the target resolution:

```math
\begin{align}
  \mathbf{c}_\text{ar} = (h_\text{tgt},\, w_\text{tgt})
\end{align}
```

**Combined conditioning**: All three micro-conditioning vectors $(\mathbf{c}_\text{size}, \mathbf{c}_\text{crop}, \mathbf{c}_\text{ar})$ are embedded via Fourier feature embeddings and concatenated with the timestep embedding before being added into the UNet's time-embedding pathway.

### Conditioning Pipeline (Pseudocode)

```
Input: training dataset D, target size s = (h_tgt, w_tgt)

while not converged:
    x      <- sample from D
    h_orig, w_orig <- height(x), width(x)
    c_size <- (h_orig, w_orig)

    x <- resize(x, s)    // scale shortest side to match target

    if h_orig <= w_orig:
        c_left ~ Uniform(0, width(x) - w_tgt)
        c_top  = 0
    else:
        c_top  ~ Uniform(0, height(x) - h_tgt)
        c_left = 0

    c_crop <- (c_top, c_left)
    x <- crop(x, s, c_crop)

    update theta using loss L(x, c_size, c_crop, c_ar=s, c_text)
```

## Improved Autoencoder (VAE)

The VAE encoder/decoder architecture is identical to prior Stable Diffusion VAEs (spatial compression factor 8, 4 latent channels), but the model was retrained from scratch with:
- **Batch size**: 256 (vs 9 in original SD)
- **EMA**: exponential moving average of weights tracked during training

**Reconstruction quality on MS-COCO 2017 validation (256×256):**

| Model | PSNR ↑ | SSIM ↑ | LPIPS ↓ | rFID ↓ |
|---|---|---|---|---|
| SD-VAE 1.x | 23.4 | 0.69 | 0.96 | 5.0 |
| SD-VAE 2.x | 24.5 | 0.71 | 0.92 | 4.7 |
| SDXL-VAE | **24.7** | **0.73** | **0.88** | **4.4** |

> [!NOTE]
> The SDXL-VAE is compatible as a drop-in replacement for SD 1.x and SD 2.x VAEs.

## Refinement Stage

A separate UNet-based diffusion model is trained to refine the outputs of the base SDXL model. The refinement model:
- Operates in the same VAE latent space ($\mathbb{R}^{b \times 4 \times h/8 \times w/8}$)
- Specializes in high-frequency details (faces, textures, fine-grained backgrounds)
- Uses SDEdit (Meng et al., 2021): adds noise to the base model's denoised latent up to a timestep $t_\text{start}$, then denoises with the refinement model

**Two-stage inference pipeline:**

```
1. Base model:
   z_T ~ N(0, I)
   z_0_base = denoise(z_T; text_cond, c_size, c_crop, c_ar)

2. Refinement model (SDEdit):
   t_start = 200  // add noise partway through schedule
   z_{t_start} = add_noise(z_0_base, sigma_{t_start})
   z_0_refined = denoise(z_{t_start}; text_cond)   // from t_start to 0

Output: decode(z_0_refined) via VAE decoder
```

**Human preference study (4-way comparison):**

| Model | Preference (%) |
|---|---|
| SDXL + refinement | **48.44** |
| SDXL base only | 36.93 |
| Stable Diffusion 1.5 | 7.91 |
| Stable Diffusion 2.1 | 6.71 |

## Training Procedure

Multi-stage training pipeline:

| Stage | Resolution | Steps | Batch Size | Notes |
|---|---|---|---|---|
| 1 (pretraining) | 256×256 | 600,000 | 2048 | — |
| 2 (finetuning) | 512×512 | 200,000 | — | Continue from Stage 1 |
| 3 (multi-aspect) | ~1024² | — | — | Offset noise $\sigma=0.05$, multi-aspect buckets |

The offset noise addition in Stage 3 helps the model generate images with correct global brightness levels, addressing a known artifact in standard diffusion training.

## Comparison with Similar Methods

| Method | Text Encoder | Latent Space | Resolution | Key Feature |
|---|---|---|---|---|
| SD 1.x | CLIP ViT-L (768-d) | VAE (4ch, f8) | 512×512 | Original LDM |
| SD 2.x | OpenCLIP ViT-H (1024-d) | VAE (4ch, f8) | 768×768 | Improved CLIP |
| **SDXL** | CLIP ViT-L + OpenCLIP ViT-bigG (2048-d) | VAE (4ch, f8) | ~1024² | Larger UNet, micro-conds, refinement |
| Imagen (Google) | T5-XXL (text-only) | Pixel-space cascade | 1024×1024 | No VAE; text-focused encoder |
| DALL-E 2 (OpenAI) | CLIP (image+text) | CLIP-guided prior | 1024×1024 | Closed; CLIP prior + LDM |
| Midjourney v5.1 | Proprietary | Proprietary | Variable | Closed; strong aesthetic quality |

> [!IMPORTANT]
> SDXL achieves worse FID scores than SD 2.x, yet human evaluators strongly prefer SDXL outputs. The authors argue FID is negatively correlated with visual aesthetics at this scale, and recommend human preference studies as the primary evaluation criterion for text-to-image foundation models.

**Midjourney v5.1 comparison (17,153 pairwise comparisons on PartiPrompts):**
- SDXL preferred: 54.9%
- Midjourney v5.1 preferred: 45.1%

## Experiments

- **Datasets**:
  - ImageNet (Deng et al., 2009) — ablation study for size conditioning
  - MS-COCO 2017 validation set — VAE reconstruction evaluation (256×256 crops)
  - PartiPrompts (P2) benchmark — 17,153-comparison user study vs. Midjourney v5.1
  - Internal large-scale image dataset (training data; details not disclosed)
- **Hardware**: Not specified
- **Optimizer**: Not specified for main model (AdamW typical for LDMs)
- **Key results**:
  - Size conditioning improves FID: 39.76 → 36.53 on ImageNet class-conditional
  - SDXL-VAE achieves best reconstruction across all metrics vs. prior SD VAEs
  - SDXL + refinement preferred by 48.44% of users in 4-way comparison (vs. 36.93% for base alone)
  - SDXL preferred over Midjourney v5.1 in 54.9% of comparisons

## Limitations

- Difficulty synthesizing intricate structures (e.g., human hands)
- Does not achieve perfect photorealism
- Contains social and racial biases inherited from training data
- "Concept bleeding" — multiple objects in a scene may unintentionally share visual attributes
- Text rendering quality is inconsistent, especially for long or complex strings

## Future Directions

- **Single-stage pipeline**: Eliminate the two-model base+refinement requirement
- **Byte-level tokenizer**: Integrate ByT5 for improved legible text rendering in images
- **Transformer-based UNet**: Explore UViT or DiT architectures in place of the convolutional UNet
- **Model distillation**: Progressive or consistency distillation for faster inference
- **Continuous-time formulation**: Adopt the EDM framework (Karras et al., 2022)

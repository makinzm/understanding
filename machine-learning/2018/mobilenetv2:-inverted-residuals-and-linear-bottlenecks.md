# Meta Information

- [ [1801.04381] MobileNetV2: Inverted Residuals and Linear Bottlenecks ]( https://arxiv.org/abs/1801.04381 )
- LICENSE: [ arXiv.org - Non-exclusive license to distribute ]( https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html )

# 1. Introduction

In this paper, we present a new mobile tailored CV models architecutre by inverted residuals and linear bottlenecks to decrease the computation cost.

# 2. Related work

MobileNetV2 is based on MobileNetV1.

> [!NOTE]
>
> [ [1704.04861] MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications ]( https://arxiv.org/abs/1704.04861 )
>
> There is depthwise and pointwise convolution to reduce the computation cost.
>
> Normal computation cost is $K^2 \cdot C_{in} \cdot C_{out} \cdot H \cdot W$　　（K: kernel size, C: channel size, H: height, W: width）
>
> MobileNetV1 computation cost is $K^2 \cdot C_{in} \cdot H \cdot W + C_{in} \cdot C_{out} \cdot H \cdot W$

> [!CAUTION]
> In original paper, $H$ and $W$ is the same $D_f$, but in this note, I use $H$ and $W$ for height and width respectively.

# 3. Preliminaries, discussion and intuition

## 3.1. Depthwise separable convolutions

>[!NOTE]
> It is the same as MobileNetV1.

## 3.2. Linear bottlenecks

ReLU causes information loss in low-dimensional manifolds. So, we should use linear transformation in low-dimensional space.

So, we firstly expand linearly to high-dimensional space, then apply non-linear transformation, and finally reduce linearly to low-dimensional space.

## 3.3. Inverted residuals

In inverted residuals, shortcut connections are between the bottleneck layers.

> [!NOTE]
> In ResNet, shortcut connections are between the high-dimensional layers not bottleneck layers.
>
> [ [1512.03385] Deep Residual Learning for Image Recognition ]( https://arxiv.org/abs/1512.03385 )

## 3.4. Information flow interpretation

This model separates capacity and expressiveness, linear transformation is for capacity, and non-linear transformation is for expressiveness.

# 4. Model architecture

Input -> Conv2d with 32 channels -> Bottleneck x 19 -> Conv2d 1 x 1 with 1280 channels -> AvgPool -> Conv2d with number of classes channels

# 5. Implementation Notes

## 5.1. Memory efficient inference

TODO

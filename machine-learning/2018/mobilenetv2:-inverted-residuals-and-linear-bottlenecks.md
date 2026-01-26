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

Inverted residual blocks allow memory-efficient inference by treating expanded intermediate tensors as disposable. 

Since depthwise convolution is per-channel, we can split the large intermediate tensor (e.g., 384ch) into chunks (e.g., 64ch), compute sequentially, and only keep small bottleneck tensors (e.g., 64ch input/output) in memory.

Memory: O(bottleneck size) instead of O(expanded size)

# 6. Experiments

# MobileNetV2 Experimental Setup and Results

## Training Setup

| Component | Value |
|-----------|-------|
| Optimizer | RMSProp (decay=0.9, momentum=0.9) |
| Weight Decay | 0.00004 |
| Learning Rate | 0.045 (decay: 0.98/epoch) |
| Batch Size | 96 (16 async GPUs) |

## Datasets

| Task | Dataset |
|------|---------|
| Classification | ImageNet |
| Object Detection | COCO (trainval35k/test-dev) |
| Segmentation | PASCAL VOC 2012 |

## Key Results

### ImageNet Classification

| Model | Top-1 | Params | MAdds | Latency |
|-------|-------|--------|-------|---------|
| MobileNetV1 | 70.6% | 4.2M | 575M | 113ms |
| MobileNetV2 | 72.0% | 3.4M | 300M | 75ms |
| MobileNetV2 (1.4) | 74.7% | 6.9M | 585M | 143ms |

> [!NOTE]
> MAdds: Multiply-Adds i.e. calculation cost

### Object Detection (COCO)

| Model | mAP | Params | MAdds |
|-------|-----|--------|-------|
| YOLOv2 | 21.6 | 50.7M | 17.5B |
| MobileNetV2+SSDLite | 22.1 | 4.3M | 0.8B |

20× more efficient, 10× smaller than YOLOv2

> [!NOTE]
> [ SSDlite — Torchvision main documentation ]( https://docs.pytorch.org/vision/main/models/ssdlite.html )
>
> mAP: mean Average Precision [ mAP (mean Average Precision) for Object Detection | by Jonathan Hui | Medium ]( https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173 )


### Semantic Segmentation (PASCAL VOC)

| Model | mIOU | Params | MAdds |
|-------|------|--------|-------|
| ResNet-101 | 80.49% | 58.16M | 81.0B |
| MobileNetV2 | 75.32% | 2.11M | 2.75B |

~5× fewer MAdds than ResNet-101

# 7. Conclusion

We proposed a new mobile architecture by inverted residuals and linear bottlenecks, which outperforms previous models on classification, detection, and segmentation tasks.

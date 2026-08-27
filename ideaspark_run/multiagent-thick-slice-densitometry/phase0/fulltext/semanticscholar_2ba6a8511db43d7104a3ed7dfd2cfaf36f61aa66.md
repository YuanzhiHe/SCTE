# Unrolled Richardson-Lucy deconvolution network with partially connected layers in computational microscopy.

paper_id: semanticscholar:2ba6a8511db43d7104a3ed7dfd2cfaf36f61aa66
tier: T2
source_used: failed
warning: fetch failed across all paths; intro filled with abstract; method empty

## Intro

Computational microscopy systems often employ optical encoders with non-localized point spread functions (PSFs) to enable advanced imaging capabilities. A notable example is the multi-cluster PSF, commonly used in light-field microscopes and mask-based integrated microscopes for single-shot 3D imaging. These systems rely on computational decoders to reconstruct high-quality images from complex optical measurements. Traditional iterative optimization methods, while physically interpretable, are computationally intensive. In contrast, deep neural networks offer faster inference but often lack physical interpretability. Physics-informed neural networks aim to bridge this gap by integrating model-based insights into data-driven architectures. Here, we propose an unrolled neural network architecture [partially connected unrolled Richardson-Lucy (RL) network (PC-RLN)], which is inspired by the RL deconvolution algorithm, and tailored for microscopy with non-localized, multi-cluster, and spatial variant PSFs. Each stage of the network mimics an iteration of the RL algorithm, and incorporates a learnable, partially connected layer to model both the forward imaging process and the back-projection. This design enables efficient and interpretable image reconstruction with substantially reduced computational cost compared to the standard RL deconvolution algorithm. We demonstrate the effectiveness of our approach in both Fourier light-field microscopy and mask-based integrated microscopy with microlens array.

## Method



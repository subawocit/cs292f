# Image Reconstruction from fMRI

*Fall 2023 CS292F (Machine Learning on Graphs) course project*

In this project, we introduced a new deep-learning framework to reconstruct images from **human brain fMRI data** using Latent Diffusion Models (LDM). 

Our contributions are:

1. We proposed four brain-to-image decoding neural network modules; 
2. We implemented a novel GCN-based module for brain decoding tasks;
2. We adapted our architecture to two distinct datasets (NSD and THINGS-fMRI) and established new benchmarks for future studies. 


![plot](/figures/model_overview.png)
**Figure 1.** Proposed framework overview, image adapted from [Takagi & Nishimoto, 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Takagi_High-Resolution_Image_Reconstruction_With_Latent_Diffusion_Models_From_Human_Brain_CVPR_2023_paper.pdf) and [Lu et al., 2023](https://dl.acm.org/doi/10.1145/3581783.3613832).

:link: For environment setup, please see: [Implementation Details Section](#implementation-details)

:bookmark_tabs: For further reading, please see: [final_report.pdf](/final_report.pdf)

# Introduction
Reconstructing images from brain activity can provide valuable insights into neural coding mechanisms. 

Recent works in brain-to-image tasks sometimes relied on having linear projections from fMRI features to pre-trained latent spaces, which may not fully capture the brain's nonlinear neural coding.

To address these gaps:
- We explored nonlinear architectures (CNN, VAE, GCN) for brain-to-image decoding.
- We incorporated LDM (Stable Diffusion) to reconstruct high-fidelity images from neural activity.

# Methods
## Pipeline
Inspired by [Takagi & Nishimoto, 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Takagi_High-Resolution_Image_Reconstruction_With_Latent_Diffusion_Models_From_Human_Brain_CVPR_2023_paper.pdf) and the understanding that the lower visual cortex is more relevant to low-level image features (e.g., edges and colors) while the higher visual cortex is associated with high-level semantic information, our reconstruction pipeline consists of two stages:

Stage 1:
- Map *higher* visual cortex fMRI activity to CLIP latent text embeddings.
- Map *lower* visual cortex fMRI activity to VQ-VAE latent image embeddings.

Stage 2:
- Generate images using LDM (Stable Diffusion) conditioned on mapped latent text and image features

## Architectures
- fMRI-to-text module
    1. CNN-based: residual Conv1D layers followed by 3 fully connected layers (inspired by [Lin et al, 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/bee5125b773414d3d6eeb4334fbc5453-Paper-Conference.pdf))
- fMRI-to-image modules 
    1. CNN-based: Residual Conv1D layers followed by 3 fully connected layers
    2. VAE-based: Variational Autoencoder with two fully connected layers as the encoder and three FC-BN-LeakyReLU blocks as the decoder
    3. GCN-based: 
        - Two ChebConv layers with BatchNorm and ReLU
        - To construct fMRI graph representation from raw fMRI signals:
            - Neurons from visual cortex V1 and V2 were treated as two separate nodes, and V3 and V4 were combined into another node.
            - Graph node's *features* were the corresponding ROI's normalized voxel activity.
            - Graph *edges* were computed by the functional connectivity across nodes using Pearson's correlation coefficient.

## Datasets
- [The Natural Scenes Dataset (NSD)](https://naturalscenesdataset.org/)
- [THINGS-fMRI](https://openneuro.org/datasets/ds004192/versions/1.0.7)

# Results

Baseline: [Takagi & Nishimoto, 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Takagi_High-Resolution_Image_Reconstruction_With_Latent_Diffusion_Models_From_Human_Brain_CVPR_2023_paper.pdf)

## Stage 1: Feature Decoding
*Nonlinear* models significantly outperformed linear baselines in decoding fMRI to image and text latent spaces.

![plot](/figures/Table3.jpg)

## Stage 2: Image Reconstruction
Our proposed *CNN-based fMRI-to-text* module and *GCN-based fMRI-to-image* module yielded the best reconstruction results, both qualitatively and quantitatively, in the NSD and THINGS-fMRI datasets.

![plot](/figures/Table4.jpg)

Sample reconstructed images from NSD dataset:

![plot](/figures/NSD.png)

Sample reconstructed images from THINGS-fMRI dataset:

<img src="figures/things.png" alt="plot" width="400"/>

All reconstructed images are available: [Google Drive](https://drive.google.com/drive/folders/13K7H1X_cuCKwYBZGEG3xxEtBNYyuUJcM?usp=drive_link)


# Conclusion
Our work introduced four brain-to-stimuli decoding methods and showed the capability of nonlinear brain-inspired architectures in reconstructing images from fMRI data, providing potential insights into visual reconstructions for Brain-Computer Interface applications.


# Implementation Details
## Environment Setup
Create and activate conda environment named ```ldm``` from ```environment_cs292.yml```
```sh
cd cs292f
conda env create -f environment_cs292.yml
conda activate ldm
```

Install Stable Diffusion v1.4 (under the diffusion_sd1/ directory), download checkpoint (sd-v1-4.ckpt), and place it under the ```codes/diffusion_sd1/stable-diffusion/models/ldm/stable-diffusion-v1/``` directory.


Note: I hard-coded some file paths, please do 
```sh
grep -r '/hdd/yuchen'
```
and change file paths accordingly to make sure everything is stored in the intended location

## File Descriptions

```generate_files.ipynb```: generating fMRI and image data files

```roi_image_encoder.ipynb```: mapping low-level fMRI to image CLIP space using GCN

```roi_text_encoder.ipynb```: mapping high-level fMRI to text CLIP space using GCN

```evaluation.ipynb```: evaluation

# Let's Connect!

:e-mail: **Yuchen Hou** |  [GitHub](https://github.com/subawocit) | [LinkedIn](https://www.linkedin.com/in/yuchen-hou-b95083205/) | [Webpage](https://bionicvisionlab.org/people/hou_yuchen/)

:rocket: I'm always happy to chat about research ideas, potential collaborations, or anything you're passionate about!

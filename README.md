# Image Reconstruction from fMRI

*Fall 2023 CS292F (Machine Learning on Graphs) course project*

In this project, we proposed a new framework to reconstract images from human fMRI data. Our contributions are:

1. We proposed **four nonlinear neural network modules** (including a GCN-based architecture) to map brain fMRI data into the corresponding image and text latent features in response to the visual stimuli. 

2. We adapted our architecture to **two distinct datasets (NSD and THINGS-fMRI)**, demonstrating its potential for Brain-Computer Interface applications and establishing new benchmarks for future studies. 


![plot](/figures/model_overview.png)
Proposed framework overview. Figure adapted from [(Takagi & Nishimoto, 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Takagi_High-Resolution_Image_Reconstruction_With_Latent_Diffusion_Models_From_Human_Brain_CVPR_2023_paper.pdf) and [(Lu et al., 2023)](https://dl.acm.org/doi/10.1145/3581783.3613832).

:link: For environment setup and required weights and datasets, please see: [Implementation Details Section](#implementation-details)

:bookmark_tabs: For further reading, please see: [final_report.pdf](/final_report.pdf)

# Implementation Details
## Environment Setup
Create and activate conda environment named ```ldm``` from ```environment_cs292.yml```
```sh
cd cs292f
conda env create -f environment_cs292.yml
conda activate ldm
```

Install Stable Diffusion v1.4 (under the diffusion_sd1/ directory), download checkpoint (sd-v1-4.ckpt), and place it under the ```codes/diffusion_sd1/stable-diffusion/models/ldm/stable-diffusion-v1/``` directory.

All reconstructed images are available: [Google Drive](https://drive.google.com/drive/folders/13K7H1X_cuCKwYBZGEG3xxEtBNYyuUJcM?usp=drive_link)

note: I hard-coded some file paths, please do 
```sh
grep -r '/hdd/yuchen'
```
and change file paths accordingly to make sure everything is stored in the intended location

File description:

```generate_files.ipynb```: generating fmri and image data files

```roi_image_encoder.ipynb```: mapping low-level fmri to image CLIP space using GCN

```roi_text_encoder.ipynb```: mapping high-level fmri to text CLIP space using GCN

```evaluation.ipynb```: evaluation




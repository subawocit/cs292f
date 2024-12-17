# CS292F_Project

## Environment Setup
Create and activate conda environment named ```ldm``` from ```environment_cs292.yml```
```sh
cd CS292F_Project
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




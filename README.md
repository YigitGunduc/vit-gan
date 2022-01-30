# VIT-GAN 
[Arxiv](https://arxiv.org/abs/2110.09305)

## Abstract
In this paper, we have developed a general-purpose architecture, 
Vit-Gan, capable of performing most of the image-to-image translation 
tasks from semantic image segmentation to single image depth 
perception. This paper is a follow-up paper, an extension of
generator-based model [1] in which the obtained results were very promising. 
This opened the possibility of further improvements with adversarial architecture. 
We used a unique vision transformers-based generator architecture and 
Conditional GANs(cGANs) with a Markovian Discriminator (PatchGAN) (this https URL).
In the present work, we use images as conditioning arguments.
It is observed that the obtained results are more realistic than the commonly 
used architectures.

## Setup

Clone the repo
```bash
git clone https://github.com/yigitgunduc/vit-gan/
```

Install requirements
```bash
pip3 install -r requirements.txt
```

> For GPU support setup `TensorFlow >= 2.4.0` with `CUDA v11.0 or above` 
> - you can ignore this step if you are going to train on the CPU

## Training

Train the model
```bash
python3 src/train.py
```
Weights are saved after every epoch and can be found in `./weights/`

## Evaluating

After you have trained the model you can test it against 3 different criteria 
(FID, Structural similarity, Inceptoin score). 

```bash
python3 src/evaluate.py path/to/weights
```

## Datasets

Implementation support 8 datasets for various tasks. 6 pix2pix datasets and two additional ones.
6 of the pix2pix dataset can be used by changing the `DATASET` variable on the `src/train.py`
for the additional datasets please see `notebooks/object-segmentation.ipynb` and 
`notebooks/depth.ipynb`

Dataset available thought the `src/train.py` 

- `cityscapes` 99 MB
- `edges2handbags` 8.0 GB
- `edges2shoes` 2.0 GB
- `facades`	29 MB
- `maps` 239 MB
- `night2day` 1.9 GB

Dataset available though the notebooks

- `Oxford-IIIT Pets` 
- `RGB+D DATABASE`

## Cite
If you use this code for your research, please cite our paper link 
```           
@article{gunducc2021vit,
  title={Vit-GAN: Image-to-image Translation with Vision Transformes and Conditional GANS},
  author={G{\"u}nd{\"u}{\c{c}}, Yi{\u{g}}it},
  journal={arXiv preprint arXiv:2110.09305},
  year={2021}
}
```

This repository contains the implementation, trained models, and an excerpt from our labeled IC scanning electron microscope (SEM) image dataset from our paper _"Towards Unsupervised SEM Image Segmentation for IC Layout Extraction"_. We will make the full metal-2 layer dataset available upon publication.

The SEM images and labels are licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png

# Setup

Tested on Python 3.8.10 with `torch` 1.12.1 on CUDA 11.3.

Next to the dependencies from `requirements.txt` (install using `pip install -r requirements.txt`),
this project requires [`torch`](https://pytorch.org/) with a matching CUDA installation, as well as OpenCL at least in version 2.0.

# Training

Run `python train.py -h` for details on the script's parameters.
Note that the `--batch_split` parameter must evenly divide the number of image patches.
Our 4096x3536 pixel SEM images are split into 864 128x128 pixel patches.
Splitting these images into 8, 16, or 32 batches would thus work, while splitting them into 10 batches, for example, would cause an error.

To supply training data, copy your SEM image dataset into the `dataset/sems` folder and their SVG labels into `dataset/labels`. When using images of a different size, adapt the scripts accordingly.

# Evaluation

Run `python eval.py` with corresponding parameters to obtain the Electrically Significant Difference results of a trained model on the whole dataset.
We supply the models we trained for our evaluation in the `models` directory.

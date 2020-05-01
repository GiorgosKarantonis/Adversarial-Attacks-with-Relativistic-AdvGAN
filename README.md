# Adversarial Attacks with AdvGAN and AdvRaLSGAN


## Introduction

Adversarial examples are a very exciting ascpect of Deep Learning! This repo contains a PyTorch implementation of the 
[AdvGAN](https://arxiv.org/abs/1801.02610) model for MNIST, CIFAR-10 and the 
[NIPS 2017 Adversarial Learning challenges dataset](https://www.kaggle.com/google-brain/nips-2017-adversarial-learning-development-set). 

I've also adapted the [Relativistic Average LSGAN (RaLSGAN)](https://arxiv.org/abs/1807.00734) and shown that it is able to 
increase the performance of the original AdvGAN both in terms of accuracy and perceptual similarity of the adversarial 
examples to the original ones. 

| Dataset                          | Target Model         | AdvGAN (paper) | AdvGAN (implementation) | AdvRaLSGAN |
|:--------------------------------:|:--------------------:|:--------------:|:-----------------------:|:----------:|
| MNIST *(black-box)*              | Naturally Trained    | -              | **85.49%**              | 89.76%     |
| MNIST *(black-box)*              | Adversarially Trained| -              | 98.12%                  | **97.97%** |
| MNIST *(black-box)*              | Secret Model         | **92.76%**     | 98.12%                  | **97.96%** |
| CIFAR-10 *(black-box)*           | Naturally Trained    | -              | 84.14%                  | **81.77%** |
| CIFAR-10 *(black-box)*           | Adversarially Trained| -              | 86.80%                  | **86.64%** |
| CIFAR-10 *(black-box)*           | Secret Model         | -              | 86.60%                  | **86.33%** |
| HighResolution *(semi white-box)*| Inception v3         | **0%**         | 70%                     | 70%        |

*Note 1 : I haven't implemented the distillation techniques for the black-box attacks and I believe that this is a big part of the reason for the performance difference between the original paper and my implementation.*

*Note 2: The score of Inception v3 on pristine data was ~95%.*

*Note 3: My guess for the scores for HighResolution being suboptimal is that the authors of the AdvGAN paper either
trained the AdvGAN also on data from ImageNet or they trained the target Inception v3 model from scratch only on the NIPS 2017 
Adversarial Learning challenges dataset; I plan to work on that in the future.*

![mnist](https://github.com/GiorgosKarantonis/Adversarial-Attacks/blob/master/img/mnist.png)

![high_res](https://github.com/GiorgosKarantonis/Adversarial-Attacks/blob/master/img/high_res.png)

## Overview

*The following modules are required:*

* `cuda/10.1` *(if you want to run the code on GPU, which is highly recommended)*

* `python3/3.6.5`

* `pytorch/1.3`

* `tensorflow/1.13.1` *(only for the MadryLab Challenge)*

Also, all the dependencies can be installed using the requirements.txt file. 

Finally, all the hyperparameters are defined in `hyperparameters.json`. If you want to simply experiment with the model 
this is the only file you'll need to modify 🙂. 

**The available hyperparameters are the following:**

* `target_dataset` : The dataset used to train the target model. Choose between `MNIST`, `CIFAR10` and 
[`HighResolution`](https://www.kaggle.com/google-brain/nips-2017-adversarial-learning-development-set). 
*	`target_learning_rate` : The learning rate for training the target model. 
*	`target_model_epochs` : The number of epochs for the target's training. 

*	`AdvGAN_epochs` : The number of epochs for AdvGAN's training. 
*	`AdvGAN_learning_rate` : The learning rate for training AdvGAN. 
*	`maximum_perturbation_allowed` : The maximum change allowed in the value of a single pixel. If set to `"Auto"`, it will be 
`0.3` on a scale of `0-1` for `MNIST`, `8` on a scale of `0-255` for `CIFAR10` and `0.01` on a scale of `0-1` for `HighResolution`. 
*	`alpha` : The weight of the GAN Loss in the total loss. 
*	`beta` : The weight of the hinge loss in the total loss. 
*	`gamma` : The weight of the adversarial loss in the total loss. 
*	`kappa` : The constant used in the CW loss. The CW loss is used for capturing the adversarial loss. 
*	`c` : The constant used in the hinge loss. 
*	`D_number_of_steps_per_batch` : The number of updates for the Discriminator before the Generator gets updated. 
*	`G_number_of_steps_per_batch` : The number of updates for the Generator before the Discriminator gets updated. 
*	`is_relativistic` : If set to `True`, the cost function for the AdvGAN will be the Relativistic Average Least Squares 
loss. Otherwise, the training will be on the Least Squares objective as in the original paper. 


## Instructions

### Run the code for MNIST and CIFAR-10

To run the code simply specify the hyperparameters you want and run `python3 main.py`. 

Since I haven't uploaded the checkpoints for the target models, this will first train the target you specified in the 
`hyperparameters.json` and then the AdvGAN (or the AdvRaLSGAN if you set `is_relativistic=True`). 

All the losses and the produced adversarial examples are now saved in the new folder `src/results`. 

A new folder called `src/npy` has also been created. The four numpy files are saved in it have the original images, the adversarial images, the true labels and the labels predicted by the target. Keep in mind that in some cases, like for the `HighResolution` dataset, you have need to denormalize the images if you want them to look to look as in their original form. You can use the `NormalizeInverse` class from the `custom_data.py` script for this task. 

### Run the code for High Resolution Images
First you have to do the following: 

* Download the dataset from 
[NIPS 2017 Adversarial Learning challenges](https://www.kaggle.com/google-brain/nips-2017-adversarial-learning-development-set). 
* Copy everything in `datasets/high_resolution`
* Rename `datasets/high_resolution/images` to `datasets/high_resolution/img`

When you're done, follow the same steps as for MNIST and CIFAR-10. 


### How to test on the MadryLab Challenges
MadryLab has created challenges for Adversarial Attacks for both the [MNIST](https://github.com/MadryLab/mnist_challenge) 
and the [CIFAR-10](https://github.com/MadryLab/cifar10_challenge) dataset. Because their implementation is in Tensorflow, 
I've modified the `run_attack.py` file to also work with the outputs of this PyTorch repo. 

#### In order to setup the challenges

* Fork or download the MadryLab Challenges' repos (probably we don't need the full repos but just to be sure). 
* Get their pretrained checkpoints using `python3 fetch_model.py` following the instructions in the respective repos. 
* Copy all the files of the MNIST repo, **apart from the `run_attack.py`**, in `src/MadryLab_Challenge/MNIST/`
* Copy all the files from the CIFAR-10 repo, **apart from the `run_attack.py`**, in `src/MadryLab_Challenge/CIFAR10/`

#### In order to test our model in the challenges

* If you just trained AdvGAN on the MNIST dataset, copy `src/npy` to `src/MadryLab_Challenge/MNIST/`. Otherwise, if you just 
trained AdvGAN on the CIFAR-10 dataset, copy `src/npy` to `src/MadryLab_Challenge/CIFAR10/`. 
* Open `src/MadryLab_Challenge/{Dataset}/config.json` and specify in `model_dir` if you want to test against the Naturally 
trained, the Adversarially trained or the Secret model. You can check instructions in the challenges' repos for the appropriate values. 
* Run the `run_attack.py` script. 

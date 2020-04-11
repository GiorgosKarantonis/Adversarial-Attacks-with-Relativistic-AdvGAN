# Increasing Privacy with Adversarial Examples

Introduction

Large Datasets and powerful computers have made Neural Networks the go to architecture for AI applications to a point that even people without scientific knowledge can utilize them. One highly contributing factor to the rise of Neural Networks is their ability to act as impressively good classifiers, meaning that after proper training a Neural Network is able to classify unseen data points so well that some models can beat even human performance. But does this mean that they actually understand the underlying concepts of the data or do they just learn to memorize the training set?

## What are Adversarial Examples and Adversarial Attacks
An adversarial example is a perturbed version of an actual data point that looks similar or even identical to the original data point but at the same time it can trick a neural network into misclassifying it. It is straightforward to also define an adversarial attack as the act of crafting adversarial examples and feeding them to a Neural Network. The two major ways to create adversarial examples are to either directly perform  perturbations to the original example or to craft a suitable mask and apply it to the original example. One impressive aspect of the creation of adversarial examples is exposed by the "One Pixel Attack" which showed that in some cases even the perturbation of a single pixel can generate an adversarial example. 

![pig](https://github.com/GiorgosKarantonis/Adversarial-Attacks/blob/master/img/pig.png)
![macaw](https://github.com/GiorgosKarantonis/Adversarial-Attacks/blob/master/img/macaw.png)

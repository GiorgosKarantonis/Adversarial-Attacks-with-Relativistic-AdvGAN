# Adversarial Attacks on Image Classifiers

## Introduction
Large Datasets and powerful computers have made Neural Networks the go to architecture for AI applications to a point that even people without scientific knowledge can utilize them. One highly contributing factor to the rise of Neural Networks is their ability to act as impressively good classifiers, meaning that after proper training a Neural Network is able to classify unseen data points so well that some models can beat even human performance. 


## What are Adversarial Examples and Adversarial Attacks
An adversarial example is a perturbed version of an actual data point that although it seems similar or even identical to the original data point, it can trick a neural network into misclassifying it. The two major ways to create adversarial examples are to either perform proper perturbations to the original example or to craft a suitable mask and apply it to the original example. 

The act of feeding an adversarial example to a Neural Network is called adversarial attack. Adversarial Attacks are divided on white box and black box depending on whether or not we know the exact model we are trying to attack and on targeted and untargeted depending on whether or not we want the adversarial example to be classified as a specific class. 


## Why should we care
The search for adversarial examples allows for better evaluation of deep models and gives us a better understanding of how they actually learn; do they tend to simple memorize patterns or do they understand, at least to some extent, high level concepts?

From a theoretical point of view the utter goal of Neural Networks is not to achieve high accuracy on the training set, but to be able to be able generalize well on unseen data. This implies that the training data can be seen as a sample from a larger distribution and thus the goal is to approximate this distribution as efficiently as possible, meaning that the classes' boundaries need to be be as accurate as possible. The robustness against adversarial attacks can thus give further insight on how accurate the learned function actually is. 

In a more practical side, the authors of the paper "Robust Physical-World Attacks on Deep Learning Visual Classification" showed that they were able to trick neural networks to misclassify stop signs during driving by performing slight physical perturbation to actual stop signs. The physical perturbations were performed using graffity or by applying proper stickers to the signs but in a way that a human could still correctly identify the stop signs. Considering that we are in a time that self driving cars are getting more and more into our lives makes it obvious how that adversatial attacks can impose an actual and life threatening danger. 

/* STOP SIGNS IMAGE */


## Performing Adversarial Attacks


### Dealing with Black Box Settings


# Generative Adversarial Networks
The basic assumption of all generative models is that the observed data are just a sample of a larger distribution. Such models try to either calculate or approximate this distribution and assuming a generative model has managed to approximate efficiently this distribution we could be able to sample new data points from it. This could allow to perform style transfer (examples), to create new realities in to better train reinforcement learning models and much more. 

Generative Adversarial Networks or GANs are one the hottest models in generative modelling right now because the have achieved revolutionary results in this field. 

-Description of GANs
-Problems with GANs (convergence and mode collapse)


# Performing Adversarial Attacks with GANs



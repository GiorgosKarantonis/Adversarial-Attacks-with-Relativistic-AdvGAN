# Adversarial Attacks on Image Classifiers

## Introduction
Large Datasets and powerful computers have made Neural Networks the go to architecture for AI applications to a point that even people without scientific knowledge can utilize them. One highly contributing factor to the rise of Neural Networks is their ability to act as impressively good classifiers, meaning that after proper training a Neural Network is able to classify unseen data points so well that some models can beat even human performance. 


## What are Adversarial Examples and Adversarial Attacks
An adversarial example is a perturbed version of an actual data point that although it looks similar or even identical to the original data point, it can trick a neural network into misclassifying it while the act of crafting adversarial examples and feeding them to a Neural Network is called adversarial attack. The two major ways to create adversarial examples are to either directly perform  perturbations to the original example or to craft a suitable mask and apply it to the original example. 

Adversarial Attacks are divided on white box and black box depending on whether or not we know the exact model we are trying to attack and on targeted and untargeted depending on whether or not we want the adversarial example to be classified as a specific class. 


## Why should we care
The search for adversarial examples allows for better evaluation of deep models and gives us a better understanding of how they actually learn; do they tend to simple memorize patterns or do they understand, at least to some extent, high level concepts?

From a theoretical point of view the utter goal of Neural Networks is not to achieve high accuracy on the training set, but to be able to be able generalize well on unseen data. This implies that the training data can be seen as a sample from a larger distribution and thus the goal is to approximate this distribution as efficiently as possible, meaning that the classes' boundaries need to be be as accurate as possible. The robustness against adversarial attacks can thus give further insight on how accurate the learned function actually is. 

In a more practical side, the authors of the paper "Robust Physical-World Attacks on Deep Learning Visual Classification" showed that they were able to trick neural networks to misclassify stop signs during driving by performing slight physical perturbation to actual stop signs. The physical perturbations were performed using graffity or by applying proper stickers to the signs but in a way that a human could still correctly identify the stop signs. Considering that we are in a time that self driving cars are getting more and more into our lives makes it obvious how that adversatial attacks can impose an actual and life threatening danger. 

/* STOP SIGNS IMAGE */


## Performing Adversarial Attacks
One of the simplest ways to perform adversarial attacks is the Fast Gradient Sign Method which is a non-iterative method proposed by Ian Goodfellow et al.. The idea behind this method is to take a step defined by a hyperparameter, epsilon, towards the direction that is defined by the gradient of the loss function with respect to the example. . Due to the fact that this method is a non iterative one and its function relies solely on the hyperparamer epsilon and the direction that is obtained by the gradient, its main contribution is the exposure of the existance of adversarial examples and cannot be considered a proper way to perform adversarial attacks. 


### Dealing with Black Box Settings


## Defending Against Adversarial Attacks
Since the discovery of adversarial examples many algorithms have been proposed but it wasn't until Madry et al. proposed a variation of Gradient Descent called Projected Gradient Descent (PGD) . In their paper, called "Towards Deep Learning Models Resistant to Adversarial Attacks", they provided a mathematical proof showing that PGD can detect all the extrema that can be found than any first order method. It is derived from their discovery that PGD can create the most powerful adversarial examples and thus a system trained on a dataset augmented with such adversarial examples can successfully block most of the adversarial attacks. 


# Generative Adversarial Networks
The basic assumption of all generative models is that the observed data are just a sample of a larger distribution. Such models try to either calculate or approximate this distribution so that we could be able to sample new data points from it. This could allow to perform style transfer (examples), to create new realities in to better train reinforcement learning models and much more. 

Generative Adversarial Networks or GANs are one the hottest models in generative modelling right now because the have achieved revolutionary results in this field. While other popular methods like Variational Autoencoders try to assign training examples into hyperspaces, GANs take a different approach and try to learn a function capable of fooling a classifier. 

Although GANs have become extremely popular and a huge amount of variations over the past few years, their training can still be very challenging. One major problem of GANs is the so called mode collapse, which refers to their inability to create a variaty of realistic looking examples. If the generator realizes that a specific 


-Description of GANs
-Problems with GANs (convergence and mode collapse)


# Performing Adversarial Attacks with GANs



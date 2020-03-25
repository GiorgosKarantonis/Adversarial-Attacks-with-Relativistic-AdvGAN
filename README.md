# Adversarial Attacks on Image Classifiers

## Introduction
Large Datasets and powerful computers have made Neural Networks the go to architecture for AI applications to a point that even people without scientific knowledge can utilize them. One highly contributing factor to the rise of Neural Networks is their ability to act as impressively good classifiers, meaning that after proper training a Neural Network is able to classify unseen data points so well that some models can beat even human performance. But does this mean that they actually understand the underlying concepts of the data or do they just learn to memorize the training set? 


## What are Adversarial Examples and Adversarial Attacks
An adversarial example is a perturbed version of an actual data point that looks similar or even identical to the original data point but at the same time it can trick a neural network into misclassifying it. It is straightforward to also define an adversarial attack as the act of crafting adversarial examples and feeding them to a Neural Network. The two major ways to create adversarial examples are to either directly perform  perturbations to the original example or to craft a suitable mask and apply it to the original example. One impressive aspect of the creation of adversarial examples is exposed by the "One Pixel Attack" which showed that in some cases even the perturbation of a single pixel can generate an adversarial example. 

/* EXAMPLE IMAGES OF ADVERSARIAL ATTACKS */


## Why should we care
The search for adversarial examples allows for better evaluation of deep models and gives us a better understanding of how they actually learn; do they tend to simple memorize patterns or do they understand, at least to some extent, high level concepts?

From a theoretical point of view the utter goal of Neural Networks is not to achieve high accuracy on the training set, but to be able to be able generalize well on unseen data. This implies that the training data can be seen as a sample from a larger distribution and thus the goal is to approximate this distribution as efficiently as possible, meaning that the classes' boundaries need to be be as accurate as possible. The robustness against adversarial attacks can thus give further insight on how accurate the learned function actually is. 

In a more practical side, the authors of the paper "Robust Physical-World Attacks on Deep Learning Visual Classification" showed that they were able to trick neural networks to misclassify stop signs during driving by performing slight physical perturbation to actual stop signs. The physical perturbations were performed using graffity or by applying proper stickers to the signs but in a way that a human could still correctly identify the stop signs. Considering that we are in a time that self driving cars are getting more and more into our lives makes it obvious how that adversatial attacks can impose an actual and life threatening danger. 

![stop sign](https://github.com/GiorgosKarantonis/Adversarial-Attacks/blob/master/img/stop_sign.png)


## Performing Adversarial Attacks on Image Classifiers
In general, adversarial attacks are divided into white box and black box depending on whether or not we have knowledge about the exact model we are trying to attack and into targeted and untargeted depending on whether or not we want the adversarial example to be classified as a specific class. 

Targeted attacks are deployed by detecting the second most probable class of an actual example and trying to perturb the original example enought such that it gets assigned to this class. The idea behind this approach is that this way we make sure that the adversarial example has is constructed by applying the minimum perturbation to the original image. This way the difference between the original and the adversarial datapoint is highly probable that will be unnoticed even by humans. 

Although it would be ideal to know the exact model that we want to attack, in real case applications this is almost impossible. Thus most of the attacks performed in the real world fall into the black box category, but this doesn't mean that we should simply discard the white box setting. In research, we may have knowledge about some models and try to attack them in order to evaluate the performance of a adversarial attack model. Apart from that even in the real world we may know that a target model is based on an existing model, for which we have some knowledge, and leverage this knowledge to better approximate the target model architecture. In the case where we want to attack a completely unknown model, meaning we want to perform a black box attack, the workaround is to first try to approximate the target model by feeding it various examples and observing the results. Since a trained Neural Network is nothing more than a learned function, we can approximate this function to some extent with good results in some cases. 

### Fast Gradient Sign Method
One of the simplest ways to perform adversarial attacks is the Fast Gradient Sign Method(FGSM) which is a non-iterative method proposed by Ian Goodfellow et al.. The idea behind this method is to take a step the size of which is defined by a hyperparameter, epsilon, towards the direction that is defined by the gradient of the loss function with respect to the example. . Due to the fact that this method is a non iterative one and its function relies solely on the hyperparamer epsilon and the direction that is obtained by the gradient, its main contribution is the exposure of the existance of adversarial examples and cannot be considered a proper way to perform adversarial attacks. 

![panda-gibbon](https://github.com/GiorgosKarantonis/Adversarial-Attacks/blob/master/img/panda_gibbon.png)

### AdvGAN
The basic assumption of all generative models is that the observed data are just a sample of a larger distribution. Such models try to either calculate or approximate this distribution so that we could be able to sample new data points from it. This could allow to perform style transfer (examples), to create new realities in to better train reinforcement learning models and much more. 

Generative Adversarial Networks or GANs are one the hottest models in generative modelling right now because the have achieved unprecedented results. While other popular methods like Variational Autoencoders try to assign training examples into hyperspaces, GANs take a different approach and try to learn a function capable of fooling a classifier. 

Although GANs have become extremely popular and a huge amount of variations have been proposed over the past few years, their training can still be very challenging. One major problem of GANs is the so called mode collapse, which refers to their inability to create a variaty of realistic looking examples. If the generator realizes that a specific 
-Problems with GANs (convergence and mode collapse)

![strawberry](https://github.com/GiorgosKarantonis/Adversarial-Attacks/blob/master/img/strawberry.png)
![buckeye](https://github.com/GiorgosKarantonis/Adversarial-Attacks/blob/master/img/buckeye.png)

### Spatially Transformed Adversarial Examples

### Other Methods


## Defending Against Adversarial Attacks
Since the discovery of adversarial examples many algorithms have been proposed but it wasn't until Madry et al. proposed a variation of Gradient Descent called Projected Gradient Descent (PGD) . In their paper, called "Towards Deep Learning Models Resistant to Adversarial Attacks", they provided a mathematical proof showing that PGD can detect all the extrema that can be found than any first order method. It is derived from their discovery that PGD can create the most powerful adversarial examples and thus a system trained on a dataset augmented with such adversarial examples can successfully block most of the adversarial attacks. 





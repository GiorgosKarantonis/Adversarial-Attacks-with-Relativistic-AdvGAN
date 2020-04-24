import os
import json

import numpy as np

import torch
import torchvision.datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import models
from advGAN import AdvGAN_Attack



def create_required_directories():
    if not os.path.exists('./results/examples/MNIST/train/'):
        os.makedirs('./results/examples/MNIST/train/')

    if not os.path.exists('./results/examples/MNIST/test/'):
        os.makedirs('./results/examples/MNIST/test/')

    if not os.path.exists('./results/examples/CIFAR10/train/'):
        os.makedirs('./results/examples/CIFAR10/train/')

    if not os.path.exists('./results/examples/CIFAR10/test/'):
        os.makedirs('./results/examples/CIFAR10/test/')

    if not os.path.exists('./checkpoints/target/'):
        os.makedirs('./checkpoints/target/')

    if not os.path.exists('./npy/MNIST/'):
        os.makedirs('./npy/MNIST/')

    if not os.path.exists('./npy/CIFAR10/'):
        os.makedirs('./npy/CIFAR10/')


def test_attack_performance(target, dataloader, mode, adv_GAN, target_model, batch_size, l_inf_bound, dataset_size):
    n_correct = 0

    true_labels, pred_labels = [], []
    img_np, adv_img_np = [], []
    for i, data in enumerate(dataloader, 0):
        img, true_label = data
        img, true_label = img.to(device), true_label.to(device)
        
        perturbation = adv_GAN(img)
        perturbation = torch.clamp(perturbation, -l_inf_bound, l_inf_bound)
        
        adv_img = perturbation + img
        adv_img = torch.clamp(adv_img, 0, 1)
        
        pred_label = torch.argmax(target_model(adv_img), 1)
        n_correct += torch.sum(pred_label == true_label, 0)

        true_labels.append(true_label.cpu().numpy())
        pred_labels.append(pred_label.cpu().numpy())
        img_np.append(img.detach().permute(0, 2, 3, 1).cpu().numpy())
        adv_img_np.append(adv_img.detach().permute(0, 2, 3, 1).cpu().numpy())

        print('Saving images for batch {} out of {}'.format(i+1, len(dataloader)))
        for j in range(adv_img.shape[0]):
            save_image(adv_img[j], './results/examples/{}/{}/example_{}_{}.png'.format(target, mode, i, j))


    true_labels = np.concatenate(true_labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)
    img_np = np.concatenate(img_np, axis=0)
    adv_img_np = np.concatenate(adv_img_np, axis=0)

    np.save('./npy/{}/true_labels'.format(target), true_labels)
    np.save('./npy/{}/pred_labels'.format(target), pred_labels)
    np.save('./npy/{}/img_np'.format(target), img_np)
    np.save('./npy/{}/adv_img_np'.format(target), adv_img_np)

    print(target)
    print('Correctly Classified: ', n_correct.item())
    print('Accuracy under attacks in {} {} set: {}\n'.format(target, mode, n_correct.item()/dataset_size))


def train_target_model(target, target_model, epochs, train_dataloader, test_dataloader, dataset_size):
    target_model.train()
    optimizer = torch.optim.Adam(target_model.parameters(), lr=LR_TARGET_MODEL)
    
    for epoch in range(epochs):
        loss_epoch = 0

        for i, data in enumerate(train_dataloader, 0):
            train_imgs, train_labels = data
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            logits_model = target_model(train_imgs)
            criterion = F.cross_entropy(logits_model, train_labels)
            loss_epoch += criterion
            
            optimizer.zero_grad()
            criterion.backward()
            optimizer.step()

        print('Loss in epoch {}: {}'.format(epoch, loss_epoch.item()))

    # save model
    targeted_model_file_name = './checkpoints/target/{}_bs_{}_lbound_{}.pth'.format(target, batch_size, l_inf_bound)
    torch.save(target_model.state_dict(), targeted_model_file_name)
    target_model.eval()

    n_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        
        pred_lab = torch.argmax(target_model(test_img), 1)
        n_correct += torch.sum(pred_lab == test_label,0)

    print('{} test set:'.format(target))
    print('Correctly Classified: ', n_correct.item())
    print('Accuracy in {} test set: {}\n'.format(target, n_correct.item()/dataset_size))


def init_params(target):
    if target == 'MNIST':
        train_dataset = torchvision.datasets.MNIST('./datasets', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = torchvision.datasets.MNIST('./datasets', train=False, transform=transforms.ToTensor(), download=True)

        target_model = models.MNIST_target_net().to(device)

        batch_size = 128
        l_inf_bound = .3 if L_INF_BOUND == 'Auto' else L_INF_BOUND

        n_labels = 10
        n_channels = 1
    elif target == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10('./datasets', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = torchvision.datasets.CIFAR10('./datasets', train=False, transform=transforms.ToTensor(), download=True)

        target_model = models.resnet32().to(device)

        batch_size = 400
        l_inf_bound = 8 if L_INF_BOUND == 'Auto' else L_INF_BOUND

        n_labels = 10
        n_channels = 3
    else:
        raise NotImplementedError('Unknown Dataset')

    return train_dataset, test_dataset, target_model, batch_size, l_inf_bound, n_labels, n_channels



print('\nLOADING CONFIGURATIONS\n')
with open('hyperparams.json') as hp_file:
    hyperparams = json.load(hp_file)

TARGET = hyperparams['target_dataset']
LR_TARGET_MODEL = hyperparams['target_learning_rate']
EPOCHS_TARGET_MODEL = hyperparams['target_model_epochs']
L_INF_BOUND = hyperparams['maximum_perturbation_allowed']

FORCE_TRAIN = hyperparams['force_train']
EPOCHS = hyperparams['AdvGAN_epochs']
LR = hyperparams['AdvGAN_learning_rate']
ALPHA = hyperparams['alpha']
BETA = hyperparams['beta']
GAMMA = hyperparams['gamma']
KAPPA = hyperparams['kappa']
C = hyperparams['c']
N_STEPS_D=hyperparams['D_number_of_steps_per_batch']
N_STEPS_G = hyperparams['G_number_of_steps_per_batch']
BOX_MIN = hyperparams['box_min']
BOX_MAX = hyperparams['box_max']
CLIPPING_TRICK = hyperparams['use_clipping_trick']

if hyperparams['is_relativistic'] == "True":
    IS_RELATIVISTIC = True
else:
    IS_RELATIVISTIC = False


create_required_directories()


# Define what device we are using
print('\nWARMING UP\n')
use_cuda = True
print('CUDA Available: ',torch.cuda.is_available())
device = torch.device('cuda' if (use_cuda and torch.cuda.is_available()) else 'cpu')


train_dataset, test_dataset, target_model, batch_size, l_inf_bound, n_labels, n_channels = init_params(TARGET)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1)



print('\nCHECKING FOR PRETRAINED TARGET MODEL')
try:
    pretrained_target = './checkpoints/target/{}_bs_{}_lbound_{}.pth'.format(TARGET, batch_size, l_inf_bound)
    target_model.load_state_dict(torch.load(pretrained_target))
    target_model.eval()
except FileNotFoundError:
    print('\tNO PRETRAINED MODEL FOUND... TRAINING TARGET FROM SCRATCH\n')
    train_target_model(
        target=TARGET, 
        target_model=target_model, 
        epochs=EPOCHS_TARGET_MODEL, 
        train_dataloader=train_dataloader, 
        test_dataloader=test_dataloader, 
        dataset_size=len(test_dataset)
    )
print('FINISHED TRAINING TARGET\n')


# train AdvGAN
print('\nTRAINING ADVGAN\n')
advGAN = AdvGAN_Attack(
    device, 
    target_model, 
    n_labels, 
    n_channels, 
    target=TARGET, 
    lr=LR, 
    box_min=BOX_MIN,
    box_max=BOX_MAX, 
    l_inf_bound=l_inf_bound, 
    alpha=ALPHA, 
    beta=BETA, 
    gamma=GAMMA, 
    kappa=KAPPA, 
    c=C, 
    n_steps_D=N_STEPS_D, 
    n_steps_G=N_STEPS_G, 
    clipping_trick=CLIPPING_TRICK, 
    is_relativistic=IS_RELATIVISTIC
)
# advGAN.train(train_dataloader, EPOCHS)


# load the trained AdvGAN
print('\nLOADING TRAINED ADVGAN\n')
adv_GAN_path = './checkpoints/AdvGAN/G_epoch_{}.pth'.format(EPOCHS)
adv_GAN = models.Generator(n_channels, n_channels).to(device)
adv_GAN.load_state_dict(torch.load(adv_GAN_path))
adv_GAN.eval()


print('\nTESTING PERFORMANCE OF ADVGAN\n')
test_attack_performance(target=TARGET, dataloader=test_dataloader, mode='test', adv_GAN=adv_GAN, target_model=target_model, batch_size=batch_size, l_inf_bound=l_inf_bound, dataset_size=len(test_dataset))











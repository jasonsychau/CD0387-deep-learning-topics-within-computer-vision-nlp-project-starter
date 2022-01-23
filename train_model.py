#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import logging

import argparse

from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook
import json
import os
import time
import smdebug.pytorch as smd

#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    running_loss=0
    running_corrects=0
#     hook.register_loss(criterion)
#     hook.set_mode(modes.EVAL)

    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    print(f"Testing Accuracy: {100*total_acc}")
    print(f"Testing Loss: {total_loss}")
    pass

def train(model, train_loader, valid_loader, loss_optim, optimizer, epochs, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    #TODO: Set Hook to track the loss
    hook = get_hook(create_if_not_exists=True)
    if hook:
        hook.register_loss(loss_optim)    
    
    for i in range(epochs):
        print("START TRAINING")
        # TODO: Set hook to train mode
        if hook:
            hook.set_mode(modes.TRAIN)
        model.train()
        train_loss = 0
        running_samples = 0
        for image_no, (inputs, targets) in enumerate(train_loader):
            if (image_no % 100 == 0):
                print('fitting {}'.format(image_no))
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_optim(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            running_samples+=len(inputs)
            #NOTE: train on smaller sample since time is running low
#             if running_samples>(0.2*len(train_loader.dataset)):
#                 break

        print("START VALIDATING")
        #TODO: Set hook to eval mode
        if hook:
            hook.set_mode(modes.TRAIN)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(valid_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_optim(outputs, targets)
                val_loss += loss.item()
        print(
            "Epoch %d: train loss %.3f, val loss %.3f"
            % (i, train_loss, val_loss)
        )
    return model

def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 10))
    return model

def create_data_loaders(data_dir, batch_size, test_batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    ''' 
    
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    testing_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
        download=True, transform=training_transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
            shuffle=True)

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
        download=True, transform=testing_transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
            shuffle=False)
    
    return (trainloader, testloader)

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
#     hook = get_hook(create_if_not_exists=True)
#     if hook:
#         hook.register_hook(model)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    (train_loader, test_loader) = create_data_loaders(args.data_dir, args.batch_size, args.test_batch_size)
    model=train(model, train_loader, test_loader, loss_criterion, optimizer, args.epochs, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    print("Saving the model.")
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for training (default: 100)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for testing (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args=parser.parse_args()
    
    main(args)


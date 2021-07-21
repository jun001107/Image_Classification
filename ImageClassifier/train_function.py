import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
from utils import load_checkpoint, load_cat_names


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = 
    traning_data_transforms = transforms.Compose([transforms.RandomRotation(15), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    data_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    train_datasets = datasets.ImageFolder(train_dir, transform=training_data_transforms_only)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=data_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=data_transforms)
    
    dataloader_train = torch.utils.data.DataLoader(image_datasets_train, batch_size=64, shuffle=True)
    dataloader_valid = torch.utils.data.DataLoader(image_datasets_valid, batch_size=32)
    
     return dataloader_train, dataloader_valid, image_datasets_train.class_to_idx

def check_validation_set(model, data_loader,criterion, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    loss=0
    
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if criterion:
                loss += criterion(outputs, labels).item()
    return 100*correct / total
def network(arch, gpu, hidden_layers = 4096, learning_rate = 0.001,  classifier_input_size = 25088, output_size = 102 ):
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained = True)
    elif arch == 'densenet161':
        model = models.densenet161(pretrained=True)
    else:
        model = models.vgg19(pretranined=True)

    classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(classifier_input_size, hidden_layers)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(hidden_layers, output_size)),
            ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    return model, criterion, optimizer


def training(model, train_loader, valid_loader, epochs, print_every, criterion, optimizer, device = 'cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0
    
    #set model to cuda
    model.to(device)
    
    for epoch in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_accuracy = check_validation_set(valid_loader, device)

                print("Epoch: {}/{} ".format(epoch+1, epochs),
                      "Training Loss: {:.3f} ".format(running_loss/print_every),
                      "Validation Accuracy: {:.3f}".format(valid_accuracy))
                
                running_loss = 0
                model.train()
                
                print("Epoch {} completed".format(e+1)

def train(args):
    input_size = 25088
    output_size = 102
    data_dir = args.data_dir
    arch = args.arch
    hidden_units = int(args.hidden_units)
    learning_rate = float(args.learning_rate)
    gpu = args.gpu
    epochs = int(args.epochs)
    save_dir = args.save_dir
    
    dataloader_train, dataloader_valid, class_to_idx = load_data(data_dir)
    training(model, train_loader, valid_loader, epochs, print_every, criterion, optimizer, device = 'cpu')

    checkpoint = {
    'input_size': classifier_input_size,
    'learning rate': learning_rate,
    'hidden_layers': hidden_layers,
    'output_size': output_size,
    'epochs': epochs,
    'features': model.features,
    'optimizer': model.state_dict(),
    'class_to_idx': model.class_to_idx,
    'state_dict': model.state_dict()
}
    torch.save(checkpoint, 'checkpoint.pth')
    print("saved successfully")                  
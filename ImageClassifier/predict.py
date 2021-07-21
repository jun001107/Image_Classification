import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, models
import argparse
import numpy as np
import pandas as pd
import json
import os
import random
from PIL import Image
from utils import load_checkpoint, load_cat_names
from collections import OrderedDict

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type = str, help = 'Path to the image that should be predicted.')
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='5')
    parser.add_argument('--filepath', dest='filepath', default=None)
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', default=True)
    return parser.parse_args()

def process_image(image):
    image = Image.open(image)
    image.thumbnail((256,256))
    image = image.resize((224,224))
    np_image = np.array(image)
    np_image = (np_image - np_image.mean()) / np_image.std()
    np_image = np_image.transpose((2,0,1))
    return np_image

#load checkpoint
def load_checkpoint(path):
    model_info = torch.load(path)
    model = models.vgg19(pretrained = True)
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 4096)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(4096, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier
    model.load_state_dict(model_info['state_dict'])
    return model, model_info

def predict(image_path, model, topk=5):
    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(image)
        image.unsqueeze_(0)
        image = image.float()
        model, _ = load_checkpoint(model)
        outputs = model(image)
        probs, classes = torch.exp(outputs).topk(topk)
        return probs[0].tolist(), classes[0].add(1).tolist()
    
def run(args):
    flower_names = []
    category_names = args.category_names
    checkpoint = args.checkpoint
    path_to_image_file = args.path_to_image_file
    top_k = int(args.top_k)
    gpu = args.gpu
    
    model, load_success = load_checkpoint(checkpoint, gpu)
    
    if load_success:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        
        probs, classes = predict(path_to_image_file, model, top_k, gpu)
        for class_num in classes:
            flower_names.append(cat_to_name[class_num])

        return probs, flower_names

    return [], []

if __name__ == '__main__':
    main()
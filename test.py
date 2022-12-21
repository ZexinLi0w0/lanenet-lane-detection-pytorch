import time
import os
import sys

import torch
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from model.utils.cli_helper_test import parse_args
import numpy as np
from PIL import Image
import pandas as pd
import cv2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

debug = True
def time_it(output=''):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            elapsed = end - start
            debug and print(output + '{:.6f}s'.format(elapsed))
            return result
        return wrapper
    return decorator

def load_test_data(img_path, transform):
    img = Image.open(img_path)
    img = transform(img)
    return img

@time_it('[Metrics][TIME] Running: ')
def inference(model, img):
    return model(img)

@time_it('[Metrics][TIME] Initializing: ')
def load_model(model_path, model_type):
    model = LaneNet(arch=model_type)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    return model
@time_it('[Metrics][TIME] Preprocessing: ')
def preprocess(img_path, data_transform):
    img = Image.open(img_path)
    img = data_transform(img)
    img = img.unsqueeze(0)
    return img

@time_it('[Metrics][TIME] Data Transfer: ')
def data_transfer(img):
    img = img.to(DEVICE)
    return img

def test():
    # check directory
    '''
        if os.path.exists('test_output') == False:
            os.mkdir('test_output')
    '''
    print("LaneNet Testing...")

    args = parse_args()
    img_path = args.img
    resize_height = args.height
    resize_width = args.width

    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = load_model(model_path=args.model, model_type=args.model_type)
    img = preprocess(img_path, data_transform)
    img = data_transfer(img)

    # forward
    outputs = inference(model, img)
    # begin = time.time()
    # outputs = model(img)
    # end = time.time()
    # print('Inference time: ', end-begin)

    # save output
    '''
        instance_pred = torch.squeeze(outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
        binary_pred = torch.squeeze(outputs['binary_seg_pred']).to('cpu').numpy() * 255
        input = Image.open(img_path)
        input = input.resize((resize_width, resize_height))
        input = np.array(input)
        cv2.imwrite(os.path.join('test_output', 'input.jpg'), input)
        cv2.imwrite(os.path.join('test_output', 'instance_output.jpg'), instance_pred.transpose((1, 2, 0)))
        cv2.imwrite(os.path.join('test_output', 'binary_output.jpg'), binary_pred)
    '''


if __name__ == "__main__":
    test()
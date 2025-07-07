'''
 * The Recognize Anything Plus Model (RAM++) inference on unseen classes
 * Written by Xinyu Huang
'''
import argparse
import numpy as np
import random

import torch

from PIL import Image
from ram.models import ram_plus
from ram import inference_ram_openset as inference
from ram import get_transform

from ram.utils import build_openset_llm_label_embedding
from torch import nn
import json

def slicing(img, window_height=512, window_width=512, stride=0.5):
    img_height, img_width, num_channels = img.shape
    #print(f"Image shape: {img.shape}")
    
    slices = []
    for i in range(0, img_height - window_height + 1, int(stride * window_height)):
        for j in range(0, img_width - window_width + 1, int(stride * window_width)):
            img_segment = img[i:i+window_height, j:j+window_width,:]
            slices.append(Image.fromarray(img_segment))
    
    return slices

def get_labels(model, transform, device, image):
    
    #.unsqueeze(0).to(device)
    image = np.array(Image.open(image))
    sliced_images = slicing(image)
    res_list = []
    for sliced_image in sliced_images:
        image = transform(sliced_image).unsqueeze(0).to(device)
        res = inference(image, model)
        res_list.append(res)
    
    return res_list




import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import os
import torch
from utils import read_image
import cv2

def load_as_float(path):
    return imread(path).astype(np.float32)

def read_depth(depth_name):
    depth = cv2.imread(depth_name,-1).astype(np.float32)
    depth = depth/255.0
    return depth

def read_depth2(depth_name):
    depth = cv2.imread(depth_name,-1).astype(np.float32)
    depth = depth/65535.0
    return depth

def depth2disp(depth):
    depth_tmp = depth
    # depth_tmp = (depth-depth.min()) / (depth.max()-depth.min()) # 深度图归一化，去除尺度信息
    disp = 1/(depth_tmp+1e-3)
    max_disp, min_disp = disp.max(), disp.min()
    return (disp-min_disp)/(max_disp-min_disp)

def depth2disp2(depth):
    return depth
def depth2disp_norm(depth):
    return (depth-depth.min())/(depth.max()-depth.min())

class UCL_Dataset(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, input_path, transform, train):
        np.random.seed(1)
        random.seed(1)
        self.is_train = train
        self.root = Path(input_path)
        self.transform = transform
        
        self.generateSample()
    
    def generateSample(self):
        self.sample_input, self.sample_gt = [], []

        scenes = []
        filename = self.root/'train.txt' if self.is_train else self.root/'val.txt'
        with open(filename, 'r') as f:
            lines = f.readlines()
        scenes = [self.root/line[:-1] for line in lines]
        
        for scene in scenes:
            rgb_files, depth_files = sorted(scene.listdir("FrameBuffer_*.png")), sorted(scene.listdir("Depth_*.png"))
            self.sample_input += rgb_files
            self.sample_gt += depth_files
        
        

    def __getitem__(self, index):
        image_name, depth_name = self.sample_input[index], self.sample_gt[index]
        
        original_image_rgb = read_image(image_name)  # in [0, 1] [H,W,3]
        image = self.transform({"image": original_image_rgb})["image"] # jh in [-1, 1]
        
        original_depth = read_depth(depth_name)
        # original_depth = read_depth2(depth_name)
        # disp = depth2disp(original_depth)
        # disp = depth2disp2(original_depth)
        disp = depth2disp_norm(original_depth)
        
        image,disp = torch.from_numpy(image), torch.from_numpy(disp)
        
        if self.is_train:
            return image, disp
        else:
            return image,disp,original_image_rgb.transpose((2,0,1))

    def __len__(self):
        return len(self.sample_input)


class SimCol3D_Dataset(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, input_path, transform, train):
        np.random.seed(1)
        random.seed(1)
        self.is_train = train
        self.root = Path(input_path)
        self.transform = transform
        
        self.generateSample()
    
    def generateSample(self):
        self.sample_input, self.sample_gt = [], []

        scenes = []
        filename = self.root/'train.txt' if self.is_train else self.root/'val.txt'
        with open(filename, 'r') as f:
            lines = f.readlines()
        scenes = [self.root/line[:-1] for line in lines]
        
        for scene in scenes:
            rgb_files, depth_files = sorted(scene.listdir("FrameBuffer_*.png")), sorted(scene.listdir("Depth_*.png"))
            self.sample_input += rgb_files
            self.sample_gt += depth_files
        
        

    def __getitem__(self, index):
        image_name, depth_name = self.sample_input[index], self.sample_gt[index]
        
        original_image_rgb = read_image(image_name)  # in [0, 1] [H,W,3]
        image = self.transform({"image": original_image_rgb})["image"] # jh in [-1, 1]
        
        original_depth = read_depth2(depth_name)
        # original_depth = read_depth2(depth_name)
        # disp = depth2disp(original_depth)
        # disp = depth2disp2(original_depth)
        disp = depth2disp_norm(original_depth)
        
        image,disp = torch.from_numpy(image), torch.from_numpy(disp)
        
        if self.is_train:
            return image, disp
        else:
            return image,disp,original_image_rgb.transpose((2,0,1))

    def __len__(self):
        return len(self.sample_input)

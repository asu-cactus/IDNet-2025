#!/usr/bin/env python
# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import numpy as np
import cv2
import random
import math
import torch                                                                                                                                                                                            
from torch.utils.data import Dataset, DataLoader                                                                                                                                                        
from torchvision import transforms                                                                                                                                                                      
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class ImageCSVDataset(Dataset):                                                                                                                                                                         
    def __init__(self, input_list, transform=None):                                                                                                                                                     
        self.input = input_list                                                                                                                                                                         
        self.transform = transform                                                                                                                                                                      
                                                                                                                                                                                                        
    def __len__(self):                                                                                                                                                                                  
        return len(self.input)                                                                                                                                                                          
                                                                                                                                                                                                        
    def __getitem__(self, idx):                                                                                                                                                                         
        img_path = self.input[idx][0]                                                                                                                                                                   
        image = Image.open(img_path).convert('RGB')                                                                                                                                                     
        label = self.input[idx][1]                                                                                                                                                                      
                                                                                                                                                                                                        
        if self.transform:                                                                                                                                                                              
            image = self.transform(image)                                                                                                                                                               
                                                                                                                                                                                                        
        return image, label, img_path                                                                                                                                                                   

def eval_models(test_paths, confs, testing, candidate_models):                                                                                                                                                                                                                                                                
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    since = time.time()                                                                                                                                                                                                                                                                                                   
    for M in confs['models']:
        name = M['name']
        if not testing:
            if name not in candidate_models:
                continue
        model_path = M['path']
        im_size = M['im_size']
        transform = transforms.Compose([                                                                                                                                                                        
            transforms.Resize((im_size, im_size)),                                                                                                                                                                      
            transforms.ToTensor(),                                                                                                                                                                              
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])                                                                                                                         
        ])                                                                                                                                                                                                      
        test_dataset = ImageCSVDataset(test_paths, transform=transform)                                                                                                                                        
        test_loader = DataLoader(test_dataset, batch_size= 32, num_workers = 8, shuffle=False) 
        acc_history = []                                                                                                                                                                                                                                                                                                      
        best_acc = 0.0                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                              
        model = torch.load(model_path, weights_only = False, map_location = device)                                                                                                                                                                                                                                                                                       
        model.eval()                                                                                                                                                                                                                                                                                                          
        model.to(device)                                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                              
        running_corrects = 0                                                                                                                                                                                                                                                                                                  
        All_labels = []
        All_preds = []
                                                                                                                                                                                                                                                                                                                              
        for inputs, labels, filenames in tqdm(test_loader):                                                                                                                                                                                                                                                                   
            inputs = inputs.to(device)                                                                                                                                                                                                                                                                                        
            labels = labels.to(device)                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                              
            with torch.no_grad():                                                                                                                                                                                                                                                                                             
                outputs = model(inputs)                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                                                              
            _, preds = torch.max(outputs, 1)                                                                                                                                                                                                                                                                                  
            All_labels.extend(labels.data.cpu().numpy().tolist())
            All_preds.extend(preds.cpu().numpy().tolist())
            values, indices = torch.sort(outputs, dim=1, descending=True)                                                                                                                                                                                                                                                     
            running_corrects += torch.sum(preds == labels.data)                                                                                                                                                                                                                                                               
        epoch_acc = running_corrects.double() / len(test_loader.dataset)                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                          
        print('Model: {}, Acc: {:.4f}'.format(name, epoch_acc))                                                                                                                                                                                                                                                                                
        results[name]= [All_preds, All_labels]
                                                                                                                                                                                                                                                                                                                          
    time_elapsed = time.time() - since                                                                                                                                                                                                                                                                                    
    print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                          
    return results                                                                                                                                                                                                                                                                                                    

def oval_mask_in_corner(shape1, shape2, corner='top-left'):
    height, width = shape1
    mask = np.zeros((height, width), dtype=np.uint8)

    # Define the size of the ellipse (can adjust as needed)
    ellipse_height, ellipse_width = shape2

    # Create coordinate grid
    Y, X = np.ogrid[:ellipse_height, :ellipse_width]

    # Ellipse equation: (x/a)^2 + (y/b)^2 <= 1
    a = ellipse_width // 2
    b = ellipse_height // 2
    ellipse = ((X - a)**2 / (a**2 + 0.0001) + (Y - b)**2 / (b**2 + 0.0001)) >= 1
    # Place the ellipse in the specified corner
    if corner == 'top-left':
        mask[:b, :a] = np.logical_or(mask[:b, :a], ellipse[:b, :a])
    elif corner == 'top-right':
        mask[:b, -a:] = np.logical_or(mask[:b, -a:], ellipse[:b, -a:])
    elif corner == 'bottom-left':
        mask[-b:, :a] = np.logical_or(mask[-b:, :a], ellipse[-b:, :a])
    elif corner == 'bottom-right':
        mask[-b:, -a:] = np.logical_or(mask[-b:, -a:], ellipse[-b:, -a:])
    else:
        raise ValueError("corner must be one of: 'top-left', 'top-right', 'bottom-left', 'bottom-right'")

    return mask
    #return ellipse

def get_mask(img, param):
    img_np = np.array(img)
    shape1 = img.size
    mask = np.zeros(shape1, dtype=np.uint8)
    mask = np.logical_or(mask, oval_mask_in_corner(shape1, param['bottom_left'], corner='bottom-left'))
    mask = np.logical_or(mask, oval_mask_in_corner(shape1, param['bottom_right'], corner='bottom-right'))
    mask = np.logical_or(mask, oval_mask_in_corner(shape1, param['top_right'], corner='top-right'))
    mask = np.logical_or(mask, oval_mask_in_corner(shape1, param['top_left'], corner='top-left'))

    mask_img = Image.fromarray(((~mask).astype(np.uint8) * 255).transpose()).convert("L")
    return mask_img          # <- ready for Image.paste



def create_shadow(ori_img, id_img, paper_img, param):

    bbox = get_mask(id_img, param)
    shadow = Image.new("RGBA", id_img.size, color=param['shadow_color'])
    w1, h1 = id_img.size
    w2, h2 = paper_img.size
    left_w = int((w2 - w1)/2)
    left_h = int((h2 - h1) /2)
    (o1, o2) = param['shadow_offset']
    id_coords = (left_w + o1, left_h + o2)
    shadow_coords = (left_w, left_h)
    
    paper_img.paste(shadow, shadow_coords, mask=bbox)
    paper_img = paper_img.filter(ImageFilter.GaussianBlur(radius=param['shadow_blur_radius']))
    
    paper_img.paste(id_img, id_coords, mask=bbox)
    return paper_img

def simulate_scan(params):
    img_pil = Image.open(params['input_file_path']).convert("RGBA") # Keep alpha for resizing and pasting
    img_np = np.array(img_pil)
    original_width, original_height = img_pil.size

    ori_img = img_pil
    # Adjust brightness and contrast
    enhancer_brightness = ImageEnhance.Brightness(img_pil)
    img_pil = enhancer_brightness.enhance(params['brightness'])
    enhancer_contrast = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer_contrast.enhance(params['contrast'])

    # Sharpen
    if params['sharpness_factor'] != 1.0:
        enhancer_sharpness = ImageEnhance.Sharpness(img_pil)
        img_pil = enhancer_sharpness.enhance(params['sharpness_factor'])

    img_np = np.array(img_pil)

    # Add noise
    if params['noise_std'] > 0:
        noise = np.zeros_like(img_np, np.uint8)
        cv2.randn(noise, 0, params['noise_std'])
        img_np = cv2.add(img_np, noise).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

    # Blur
    if params['blur_radius'] > 0:
        img_np = cv2.GaussianBlur(img_np, (0, 0), sigmaX=params['blur_radius'], sigmaY=params['blur_radius'])
        img_pil = Image.fromarray(img_np)

    paper = Image.open(params['paper_texture_path']).convert("RGBA")

    paper_width, paper_height = paper.size
    id_width, id_height = img_pil.size
    if params.get('id_resized_shape'):
        (target_width, target_height) = params.get('id_resized_shape') 

    resized_id = img_pil.resize((target_width, target_height))
    length = math.sqrt( target_width **2 + target_height ** 2)
    start1 = params['position1']
    start2 = params['position2']
    if start1 + length > paper_width:
        start1 = paper_width - length
        params['position1'] = start1
    if start2 + length > paper_height:
        start2 = paper_height - length
        params['position2'] = start2
    paper_tmp = paper.crop((start1, start2, start1 + length, start2 + length))

    id_img = ori_img.resize((target_width, target_height))
    resized_id = create_shadow(ori_img, id_img, paper_tmp, params)

    resized_id_width, resized_id_height = resized_id.size

    resized_id = resized_id.rotate(params['rotate'])
    # Composite resized ID onto paper
    paper.paste(resized_id, (start1, start2), resized_id.split()[3])
    final_image = paper.convert("RGB")
    return paper


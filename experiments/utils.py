#!/usr/bin/env python
# coding=utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
#from matplotlib import pyplot as plt
import json
from tqdm import tqdm
import pandas as pd

import time                                                                                                                                                                                             
import copy                                                                                                                                                                                             
import glob                                                                                                                                                                                             
import random                                                                                                                                                                                           
import timm                                                                                                                                                                                             
                                                                                                                                                                                                        
import torch                                                                                                                                                                                            
import torchvision                                                                                                                                                                                      
import torchvision.transforms as transforms                                                                                                                                                             
from efficientnet_pytorch import EfficientNet                                                                                                                                                           
import torchvision.models as models                                                                                                                                                                     
import torch.nn as nn                                                                                                                                                                                   
import torch.optim as optim 

from PIL import Image                                                                                                                                                                                   
from torch.utils.data import Dataset, DataLoader                                                                                                                                                        
from torchvision import transforms                                                                                                                                                                      
import shutil
from custom_noise import add_gaussian_noise, add_salt_and_pepper_noise, add_poisson_noise
                                                                                                                                                                                                        

def get_optimal_font_scale(text, width):
    fontsize = 1  # starting font size
    sel_font =  get_font_scale()  # "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    stop  = False  # portion of image width you want text width to be
    img_fraction = 1
    try:
        font = ImageFont.truetype(font=sel_font, size=fontsize ,encoding="unic")
    except:
        sel_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font=sel_font, size=fontsize ,encoding="unic")

    while (font.getsize(text)[0] < img_fraction*width) and (stop == False):
        # iterate until the text size is just larger than the criteria
        if font.getsize(text)[0] == 0:
            sel_font =  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

            if font.getsize(text)[1] == 0:
                stop = True
                break

        fontsize += 1
        font = ImageFont.truetype(sel_font, fontsize ,encoding="unic")

    # optionally de-increment to be sure it is less than criteria
    fontsize -= 1
    font = ImageFont.truetype(sel_font, fontsize ,encoding="unic")

    return font


def get_font_scale(inner_path: str = os.path.join(os.getcwd(), "datas", "fake_fonts", "TTF")):

    ## TODO solve
    try:
        deja = [i for i in os.listdir(inner_path) if "DejaVu" in i]

    except FileNotFoundError:
        for root, dirs, files in os.walk(os.getcwd()):
            for name in dirs:
                if "TTF" == name:
                    inner_path = os.path.join(root, name)
                    break

        deja = [i for i in os.listdir(inner_path) if "DejaVu" in i]

    selected = random.choice(deja)

    return os.path.join(inner_path, selected)

def coord_to_coord1(bbox):
    """This function convert the kin of the shape from bbox rectangle x0,y0 + heigh and weight to the polygon coordenades.

    Returns:
        _type_: _description_
    """

    x, y, x_f, y_f = bbox

    return [x, y, x_f - x, y_f - y]

def coord_to_shape(bbox):
    """This function convert the kin of the shape from bbox rectangle x0,y0 + heigh and weight to the polygon coordenades.

    Returns:
        _type_: _description_
    """

    x, y, x_f, y_f = bbox
    c1, c2, c3, c4 = [x, y], [x_f, y], [x_f, y_f], [x, y_f]

    return [c1, c2, c3, c4]

def mask_from_info(img:np.ndarray, shape:np.ndarray):

    """"
        This f(x) extract the  ROI that will be inpainted

    """
    def midpoint(x1, y1, x2, y2):
        x_mid = int((x1 + x2) / 2)
        y_mid = int((y1 + y2) / 2)
        return (x_mid, y_mid)

    x0, x1, x2, x3 = shape[0][0], shape[1][0], shape[2][0], shape[3][0]
    y0, y1, y2, y3 = shape[0][1], shape[1][1], shape[2][1], shape[3][1]


    xmid0, ymid0 = midpoint(x1, y1, x2, y2)
    xmid1, ymid1 = midpoint(x0, y0, x3, y3)

    thickness = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.line(mask, (xmid0, ymid0), (xmid1, ymid1), 255, thickness)

    masked = cv2.bitwise_and(img, img, mask=mask)

    return mask, masked

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
                                                                                                                                                                                                        
                                                                                                                                                                                                        
                                                                                                                                                                                                        


def get_font_styles(base_walk_path):
    results = []
    for r, d, f in os.walk(base_walk_path):
        f.sort()
        for file in f:
            if file.endswith('.ttf'):
                results.append(base_walk_path + file)
    return results

#def draw_images(im1, im2, output_name): 
#    plt.figure(figsize=(10, 5))                                                                                                                                         
#    plt.subplot(211), plt.imshow(im1)                                                                                                                                                                       
#    plt.title('Original Image'), plt.xticks([]), plt.yticks([])                                                                                                                                             
#    plt.subplot(212), plt.imshow(im2)                                                                                                                                                                       
#    plt.title('Generated Image'), plt.xticks([]), plt.yticks([])  
#    plt.savefig(output_name)
#    plt.show()  
#    plt.close()

def write_parameters1(xx, yy, font_size, stroke_width, xc, yc, zc,  
                    font_style_idx, content, font_styles, template):
    font_file = font_styles[int(font_style_idx)]
    imp = ImageDraw.Draw(template)
    xx = int(xx)	
    yy = int(yy)
    font_size = int(font_size)
    stroke_width = int(stroke_width)
    xc = int(xc)
    yc = int(yc)
    zc = int(zc)
    #xc = 0
    #yc = 0
    #zc = 0
    test_mf = ImageFont.truetype(font_file, int(font_size))                                                                                                                                             
    text_width = imp.textlength(content, font=test_mf)                                                                                                                                               
    text_width = int(math.ceil(text_width))                                                                                                                                                             
    coord = (xx, yy, xx + text_width, yy + int(font_size))                                                                                                                                                                                                                                                                                                                
    im2 = template.crop(coord)                                                                                                                                                                      
    im = ImageDraw.Draw(im2)                                                                                                                                                                                               
    textcolor = (int(xc), int(yc), int(zc))                                                                                                                                                         
    imp.text((xx, yy), content, textcolor, font=test_mf, stroke_width=int(stroke_width))                                                                                                            
    #im2 = im2.filter(ImageFilter.GaussianBlur(radius=blur_ratio))  
    return imp

def write_parameters(xx, yy, font_size, stroke_width, xc, yc, zc, 
                    font_style_idx, content, font_styles, template):
    font_file = font_styles[int(font_style_idx)]
    imp = ImageDraw.Draw(template)
    xx = int(xx)	
    yy = int(yy)
    font_size = int(font_size)
    stroke_width = int(stroke_width)
    xc = int(xc)
    yc = int(yc)
    zc = int(zc)
    test_mf = ImageFont.truetype(font_file, int(font_size))                                                                                                                                             
    text_width = imp.textlength(content, font=test_mf)                                                                                                                                               
    text_width = int(math.ceil(text_width))                                                                                                                                                             
    coord = (xx, yy, xx + text_width, yy + int(font_size))                                                                                                                                                                                                                                                                                                                
    im2 = template.crop(coord)                                                                                                                                                                      
    im = ImageDraw.Draw(im2)                                                                                                                                                                                               
    textcolor = (int(xc), int(yc), int(zc))                                                                                                                                                         
    imp.text((xx, yy), content, textcolor, font=test_mf, stroke_width=int(stroke_width))                                                                                                            
    im2 = im2.filter(ImageFilter.GaussianBlur(radius=blur_ratio))  
    return im2

'''
mode = [1, #:'Using whole Image',
        2, #: Using segment Image',
        3, #: Using segment edge"
       ]
'''
def load_all_templates(input_folder, area):                                                                                                                                                                                                                                                                                             
    input_paths = {}                                                                                                                                                                                                                                                                                                      
    for image_name in os.listdir(input_folder):                                                                                                                                                                                                                                                                           
        if image_name[:3] == area.lower():                                                                                                                                                                                                                                                                                
            input_path = os.path.join(input_folder, image_name)                                                                                                                                                                                                                                                           
            input_paths[image_name] = input_path                                                                                                                                                                                                                                                                                
    return input_paths 

def load_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    return annotations

def inpaint_image(img: np.ndarray, coord:np.ndarray, mask: np.ndarray, text_str: str):                                                                                                                                                                                                                                    
    """                                                                                                                                                                                                                                                                                                                   
    Inpaints the masked region in the input image using the TELEA algorithm and adds text to it.                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                          
    Args:                                                                                                                                                                                                                                                                                                                 
        img (np.ndarray): Input image.                                                                                                                                                                                                                                                                                    
        coord (np.ndarray[int, ...]): An array of integers representing the (x,y) coordinates of the top-left corner,                                                                                                                                                                                                     
            as well as the width and height of the region where the text will be added.                                                                                                                                                                                                                                   
        mask (np.ndarray): A binary mask with the same shape as `img`, where the masked pixels have value 0.                                                                                                                                                                                                              
        text_str (str): The text to be added to the inpainted region.                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                                          
    Returns:                                                                                                                                                                                                                                                                                                              
        np.ndarray: A numpy array representing the inpainted image with the text added to it.                                                                                                                                                                                                                             
    """                                                                                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                                          
    inpaint = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)                                                                                                                                                                                                                                                                
    fake_text_image = copy.deepcopy(inpaint)                                                                                                                                                                                                                                                                              
    x0, y0, w, h = coord                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                          
    color = (0, 0, 0)                                                                                                                                                                                                                                                                                                     
    font  = get_optimal_font_scale(text_str, w)                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                                          
    img_pil = Image.fromarray(fake_text_image)                                                                                                                                                                                                                                                                            
    draw = ImageDraw.Draw(img_pil)                                                                                                                                                                                                                                                                                        
    draw.text(((x0, y0)), text_str, font=font, fill=color)                                                                                                                                                                                                                                                                
    fake_text_image = np.array(img_pil)                                                                                                                                                                                                                                                                                   
                                                                                                                                                                                                                                                                                                                          
    return fake_text_image

from sklearn.metrics import confusion_matrix, accuracy_score
def get_cm(true_labels, predicted_labels):
    #true_labels = [0, 1, 2, 1, 0, 2, 1, 0]  # Replace with your true labels
    #predicted_labels = [0, 2, 2, 1, 0, 0, 1, 2]  # Replace with your predicted labels
    
    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    #print(accuracy_score(true_labels, predicted_labels))
    num_classes = cm.shape[0]
    #print(num_classes)
    
    # Calculate the False Negative Rate (FNR) for each class
    fnr_per_class = []
    fpr_per_class = []
    prevalence = []
    fpn = []
    Fp = []
    results = []
    for i in range(1, num_classes):

        FN = np.sum(cm[i, :]) - cm[i, i]  # False Negatives: Sum of row i excluding the diagonal element
        FP = np.sum(cm[:, i]) - cm[i, i]  # False Positives: Sum of column i excluding the diagonal element
        TP = cm[i, i]  # True Positives for class i
        TN = np.sum(cm) - (FP + FN + TP)  # True Negatives for class i

        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0  # False Negative Rate
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate
        ACC = (TP + TN) / (TP + TN + FP + FN)
        results.extend([ACC, FNR, FPR, TP, TN, FP, FN])
    print(results)
    return results

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
                                                                                                                                                                                                                                                                                                                              
        # Iterate over data.                                                                                                                                                                                                                                                                                                  
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
        get_cm(All_labels, All_preds)
        epoch_acc = running_corrects.double() / len(test_loader.dataset)                                                                                                                                                                                                                                                      
        #cm = ConfusionMatrix(actual_vector=All_labels, predict_vector=All_preds)                                                                                                                                                                                                                                             
        #print(cm)                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                          
        print('Model: {}, Acc: {:.4f}'.format(name, epoch_acc))                                                                                                                                                                                                                                                                                
        results[name]= [All_preds, All_labels]
                                                                                                                                                                                                                                                                                                                          
    time_elapsed = time.time() - since                                                                                                                                                                                                                                                                                    
    print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                          
    #return epoch_acc.cpu()                                                                                                                                                                                                                                                                                                    
    return results                                                                                                                                                                                                                                                                                                    

def eval_model(test_paths, confs):                                                                                                                                                                                                                                                                

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = confs['model_path']
    im_size = confs['im_size']
    transform = transforms.Compose([                                                                                                                                                                        
        transforms.Resize((im_size, im_size)),                                                                                                                                                                      
        transforms.ToTensor(),                                                                                                                                                                              
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])                                                                                                                         
    ])                                                                                                                                                                                                      
    test_dataset = ImageCSVDataset(test_paths, transform=transform)                                                                                                                                        
    test_loader = DataLoader(test_dataset, batch_size= 32, num_workers = 8, shuffle=False) 
    since = time.time()                                                                                                                                                                                                                                                                                                   
    acc_history = []                                                                                                                                                                                                                                                                                                      
    best_acc = 0.0                                                                                                                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                          
    model = torch.load(model_path, weights_only = False, map_location = device)                                                                                                                                                                                                                                                                                       
    model.eval()                                                                                                                                                                                                                                                                                                          
    model.to(device)                                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                          
    running_corrects = 0                                                                                                                                                                                                                                                                                                  
    All_labels = []
    All_preds = []
                                                                                                                                                                                                                                                                                                                          
    # Iterate over data.                                                                                                                                                                                                                                                                                                  
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
    get_cm(All_labels, All_preds)
    epoch_acc = running_corrects.double() / len(test_loader.dataset)                                                                                                                                                                                                                                                      
    #cm = ConfusionMatrix(actual_vector=All_labels, predict_vector=All_preds)                                                                                                                                                                                                                                             
    #print(cm)                                                                                                                                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                          
    print('Acc: {:.4f}'.format(epoch_acc))                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                          
    time_elapsed = time.time() - since                                                                                                                                                                                                                                                                                    
    print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                          
    #return epoch_acc.cpu()                                                                                                                                                                                                                                                                                                    
    return All_preds, All_labels                                                                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                          
def evaluate_parameters_custom(xx, yy, font_size, stroke_width, xc, yc, zc,  
                        font_style_idx, save_quality1, save_quality2, segment, confs, testing):

# add file path to file
    base_save_path = "/scratch/luluxie/tmp/"
    quality1 = int(save_quality1)
    quality2 = int(save_quality2)
    annotation_path = confs['annotation_path']
    template1 = confs['template_path']
    oppath = confs['generated_path']
    font_path = confs["fonts_path"]
    real_path = f"{oppath}/reals"
    fake_path = f"{oppath}/fakes"
    try:
        shutil.rmtree(real_path)
        shutil.rmtree(fake_path)
        os.mkdir(real_path)
        os.mkdir(fake_path)
    except:
        pass
    
    if 'font_files' in confs:
        font_styles = confs['font_files']
    else:
        font_styles = get_font_styles(font_path)
    font_file = font_styles[int(font_style_idx)]

    samples_path = confs['sample_path']
    template1_image = Image.open(template1).convert("RGB")
    area, segment_key = segment.split('_', 1)
    bbox = confs[segment_key]['bbox']
    if testing:
        val_datas = confs['test_data']
    else:
        val_datas = confs['val_data']

    templates = load_all_templates(samples_path, area)
    annotations = load_annotations(annotation_path)
    test_paths = []
    fake_paths = []
    ssims = []
    sample_paths = []

    df = pd.read_csv("Popular_Baby_Names.csv")
    names = df["Child's First Name"]

    print(df['Gender'].describe())
    print(df['Ethnicity'].describe())

    for filename, values in tqdm(annotations.items()):
        if filename in templates and filename in val_datas:
            for n_index, name in enumerate(tqdm(names)):
                template2 = templates[filename]
                template2_image = Image.open(template2).convert("RGB")
                sample_np = np.array(template2_image.crop(bbox))
                region = template1_image.crop(bbox)
                template2_image.paste(region, bbox)

                content = name #values[segment_key]['value']

                imp = ImageDraw.Draw(template2_image)

                test_mf = ImageFont.truetype(font_file, int(font_size))
                text_width = imp.textlength(content, font=test_mf)
                text_width = int(math.ceil(text_width))
                xx = int(xx)
                yy = int(yy)
                coord = (xx, yy, xx + text_width, yy + int(font_size))
                coord1 = (xx, yy, text_width, int(font_size))
                textcolor = (int(xc), int(yc), int(zc))

                imp.text((xx, yy), content, textcolor, font=test_mf, stroke_width=int(stroke_width))
                generated_np = np.array(template2_image.crop(bbox))
                sv, _ = ssim(sample_np, generated_np, full=True, multichannel=True, channel_axis=-1)
                ssims.append(sv)
                #pv = psnr(sample_np, generated_np)
                #ssims.append(pv)

                real_name = f"{base_save_path}/real_{filename}_{n_index}"
                fake_name = f"{base_save_path}/fake_{filename}_{n_index}"
                template2_image.save(real_name, format='JPEG', subsampling=0, quality=quality1)

                shape = coord_to_shape(coord)
                img = np.array(template2_image)
                mask, _ = mask_from_info(img, shape)
                fake_text_image =  inpaint_image(img=img, coord=coord1, mask=mask, text_str=content)

                Image.fromarray(fake_text_image).save(fake_name, format='JPEG', subsampling=0, quality=quality2)
                test_paths.append([real_name, 0])
                fake_paths.append([fake_name, 1])

            break
    real_gau_paths_5 = add_gaussian_noise(test_paths, 0, 5)
    fake_gau_paths_5 = add_gaussian_noise(fake_paths, 0, 5)
    real_gau_paths_10 = add_gaussian_noise(test_paths, 0, 10)
    fake_gau_paths_10 = add_gaussian_noise(fake_paths, 0, 10)
    real_gau_paths_15 = add_gaussian_noise(test_paths, 0, 15)
    fake_gau_paths_15 = add_gaussian_noise(fake_paths, 0, 15)

    real_sp_paths_01 = add_salt_and_pepper_noise(test_paths, 0.01)
    fake_sp_paths_01 = add_salt_and_pepper_noise(fake_paths, 0.01)
    real_sp_paths_05 = add_salt_and_pepper_noise(test_paths, 0.05)
    fake_sp_paths_05 = add_salt_and_pepper_noise(fake_paths, 0.05)

    real_poisson_paths = add_poisson_noise(test_paths)
    fake_poisson_paths = add_poisson_noise(fake_paths)



    real_pred, real_labels = eval_model(test_paths, confs)                                                                                                                                                                                                                                                                
    fake_pred, fake_labels = eval_model(fake_paths, confs)                                                                                                                                                                                                                                                                
    df['real_results'] = get_binary_column(real_labels, real_pred)
    df['fake_results'] = get_binary_column(fake_labels, fake_pred)

    real_pred, real_labels = eval_model(real_gau_paths_5, confs)                                                                                                                                                                                                                                                                
    fake_pred, fake_labels = eval_model(fake_gau_paths_5, confs)                                                                                                                                                                                                                                                                
    df['real_gau_paths_5'] = get_binary_column(real_labels, real_pred)
    df['fake_gau_paths_5'] = get_binary_column(fake_labels, fake_pred)

    real_pred, real_labels = eval_model(real_gau_paths_10, confs)                                                                                                                                                                                                                                                                
    fake_pred, fake_labels = eval_model(fake_gau_paths_10, confs)                                                                                                                                                                                                                                                                
    df['real_gau_paths_10'] = get_binary_column(real_labels, real_pred)
    df['fake_gau_paths_10'] = get_binary_column(fake_labels, fake_pred)

    real_pred, real_labels = eval_model(real_gau_paths_15, confs)                                                                                                                                                                                                                                                                
    fake_pred, fake_labels = eval_model(fake_gau_paths_15, confs)                                                                                                                                                                                                                                                                
    df['real_gau_paths_15'] = get_binary_column(real_labels, real_pred)
    df['fake_gau_paths_15'] = get_binary_column(fake_labels, fake_pred)

    real_pred, real_labels = eval_model(real_sp_paths_01, confs)                                                                                                                                                                                                                                                                
    fake_pred, fake_labels = eval_model(fake_sp_paths_01, confs)                                                                                                                                                                                                                                                                
    df['real_sp_paths_01'] = get_binary_column(real_labels, real_pred)
    df['fake_sp_paths_01'] = get_binary_column(fake_labels, fake_pred)

    real_pred, real_labels = eval_model(real_sp_paths_05, confs)                                                                                                                                                                                                                                                                
    fake_pred, fake_labels = eval_model(fake_sp_paths_05, confs)                                                                                                                                                                                                                                                                
    df['real_sp_paths_05'] = get_binary_column(real_labels, real_pred)
    df['fake_sp_paths_05'] = get_binary_column(fake_labels, fake_pred)

    real_pred, real_labels = eval_model(real_poisson_paths, confs)                                                                                                                                                                                                                                                                
    fake_pred, fake_labels = eval_model(fake_poisson_paths, confs)                                                                                                                                                                                                                                                                
    df['real_poisson_paths'] = get_binary_column(real_labels, real_pred)
    df['fake_poisson_paths'] = get_binary_column(fake_labels, fake_pred)

    df.to_csv(confs['save_path'], index=False)
    #acc = accuracy_score(all_samples, all_test)
    #score = acc + 0.1 * (sum(ssims) / len(ssims))
    score = sum(ssims) / len(ssims)
    #print(f"Model acc: {acc}, Evaluation score: {score}")
    return score

def get_binary_column(label, predict):
    results = []
    for i, j in zip(label, predict):
        if i == j:
            results.append(1)
        else:
            results.append(0)
    return results

def evaluate_cyclegan1(xx, yy, font_size, stroke_width, xc, yc, zc,  
                        font_style_idx, save_quality1, save_quality2, segment, confs, testing, candidate_models, with_model, dataset):

# add file path to file
    quality1 = int(save_quality1)
    quality2 = int(save_quality2)
    annotation_path = confs['annotation_path']
    template1 = confs['template_path']
    oppath = confs['generated_path']
    font_path = confs["fonts_path"]
    real_path = f"{oppath}/reals"
    fake_path = f"{oppath}/fakes"
    #try:
    #    shutil.rmtree(real_path)
    #    shutil.rmtree(fake_path)
    #    os.mkdir(real_path)
    #    os.mkdir(fake_path)
    #except:
    #    pass
    
    if 'font_files' in confs:
        font_styles = confs['font_files']
    else:
        font_styles = get_font_styles(font_path)
    font_file = font_styles[int(font_style_idx)]

    samples_path = confs['sample_path']
    template1_image = Image.open(template1).convert("RGB")
    area, segment_key = segment.split('_', 1)
    bbox = confs[segment_key]['bbox']
    if testing:
        val_datas = confs['test_data']
    else:
        val_datas = confs['val_data']

    def load_all_images(input_folder):                                                                                                                                                                                                                                                                                             
        input_paths = {}                                                                                                                                                                                                                                                                                                      
        for image_name in os.listdir(input_folder):                                                                                                                                                                                                                                                                           
            input_path = os.path.join(input_folder, image_name)                                                                                                                                                                                                                                                           
            index = image_name.split('_')[0]
            key = f'alb_id_{index}.jpg'
            if key in input_paths:
                input_paths[key].append(input_path)                                                                                                                                                                                                                                                                                
            else:
                input_paths[key] = [input_path]                                                                                                                                                                                                                                                                                
        return input_paths 

    templates = load_all_images(f'/scratch/luluxie/GAN/pytorch-CycleGAN-and-pix2pix/results/{dataset}/test_latest/images/')
    real_paths = "/scratch/luluxie/GAN/pytorch-CycleGAN-and-pix2pix/datasets/idnet2sidtdtest/testB/"
    annotations = load_annotations(annotation_path)
    test_paths = []
    ssims = []
    sample_paths = []
    filename_set = set()
    for filename, values in tqdm(annotations.items()):
        if filename in templates :
            filename_set.add(filename)
            t2 = templates[filename]
            for template2 in t2:
                #template2_image = Image.open(template2).convert("RGB")
                #sample_np = np.array(template2_image.crop(bbox))
                #region = template1_image.crop(bbox)
                #template2_image.paste(region, bbox)

                content = values[segment_key]['value']

                #imp = ImageDraw.Draw(template2_image)

                #test_mf = ImageFont.truetype(font_file, int(font_size))
                #text_width = imp.textlength(content, font=test_mf)
                #text_width = int(math.ceil(text_width))
                #xx = int(xx)
                #yy = int(yy)
                #coord = (xx, yy, xx + text_width, yy + int(font_size))
                #coord1 = (xx, yy, text_width, int(font_size))
                #textcolor = (int(xc), int(yc), int(zc))

                #imp.text((xx, yy), content, textcolor, font=test_mf, stroke_width=int(stroke_width))
                #generated_np = np.array(template2_image.crop(bbox))
                #sv, _ = ssim(sample_np, generated_np, full=True, multichannel=True, channel_axis=-1)
                #ssims.append(sv)
                ##pv = psnr(sample_np, generated_np)
                ##ssims.append(pv)

                #real_name = f"{real_path}/IDNetreal_{filename}"
                #fake_name = f"{fake_path}/IDNetfake_{filename}"
                #template2_image.save(real_name, format='JPEG') #, subsampling=0, quality=quality1)

                #shape = coord_to_shape(coord)
                #img = np.array(template2_image)
                #mask, _ = mask_from_info(img, shape)
                #fake_text_image =  inpaint_image(img=img, coord=coord1, mask=mask, text_str=content)

                ##Image.fromarray(fake_text_image).save(fake_name, format='JPEG', subsampling=0, quality=quality2)
                #Image.fromarray(fake_text_image).save(fake_name, format='JPEG') #, subsampling=0, quality=quality2)
                #test_paths.append([real_name, 0])
                #test_paths.append([fake_name, 1])

                itype = template2.split('/')[-1].split('.')[0][-4:]
                if itype == 'real':
                    template2 = os.path.join(real_paths, template2.split('/')[-1][:2] + ".jpg")

                fake_real_name = f"{fake_path}/SIDTDfake_{template2.split('/')[-1]}"
                real_img = np.array(Image.open(template2).convert("RGB"))
                real_coord = values[segment_key]['bbox']
                real_shape = coord_to_shape(real_coord)
                real_mask, _ = mask_from_info(real_img, real_shape)
                real_coord1 = coord_to_coord1(real_coord)
                fake_real_image =  inpaint_image(img=real_img, coord=real_coord1, mask=real_mask, text_str=content)
                Image.fromarray(fake_real_image).save(fake_real_name)
                
                #itype = template2.split('/')[-1].split('.')[0][-4:]
                if itype == 'real':
                    sample_paths.append([template2, 0])
                    sample_paths.append([fake_real_name, 1])
                elif itype == 'fake':
                    test_paths.append([template2, 0])
                    test_paths.append([fake_real_name, 1])
                else:
                    print(f"warning: Need to check {template2}")

    with open("SIDTD_paths_test.json", 'w') as file:
        json.dump(sample_paths, file, indent=4)
    with open("IDNet_paths_test.json", 'w') as file:
        json.dump(test_paths, file, indent=4)



    all_tests = eval_models(test_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
    all_samples = eval_models(sample_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
    accs = [accuracy_score(all_samples[key][0], all_tests[key][0]) for key in all_tests.keys()]
    print(f"Models:{all_tests.keys()}, Model accs: {accs}")
    #if with_model:
    #    score = sum(accs)/len(accs) + (sum(ssims) / len(ssims))
    #else:
    #    score = sum(ssims) / len(ssims)
    #print(f"Models:{all_tests.keys()}, Model accs: {accs}, Evaluation score: {score}")
    #return score
def evaluate_cyclegan(xx, yy, font_size, stroke_width, xc, yc, zc,  
                        font_style_idx, save_quality1, save_quality2, segment, confs, testing, candidate_models, with_model, dataset):

# add file path to file
    quality1 = int(save_quality1)
    quality2 = int(save_quality2)
    annotation_path = confs['annotation_path']
    template1 = confs['template_path']
    oppath = confs['generated_path']
    font_path = confs["fonts_path"]
    real_path = f"{oppath}/reals"
    fake_path = f"{oppath}/fakes"
    #try:
    #    shutil.rmtree(real_path)
    #    shutil.rmtree(fake_path)
    #    os.mkdir(real_path)
    #    os.mkdir(fake_path)
    #except:
    #    pass
    
    if 'font_files' in confs:
        font_styles = confs['font_files']
    else:
        font_styles = get_font_styles(font_path)
    font_file = font_styles[int(font_style_idx)]

    samples_path = confs['sample_path']
    template1_image = Image.open(template1).convert("RGB")
    area, segment_key = segment.split('_', 1)
    bbox = confs[segment_key]['bbox']
    if testing:
        val_datas = confs['test_data']
    else:
        val_datas = confs['val_data']

    templates = load_all_templates(samples_path, area)
    annotations = load_annotations(annotation_path)
    test_paths = []
    ssims = []
    sample_paths = []
    filename_set = set()
    for filename, values in tqdm(annotations.items()):
        if filename in templates and filename in val_datas and filename not in filename_set:
            filename_set.add(filename)
            template2 = templates[filename]
            template2_image = Image.open(template2).convert("RGB")
            sample_np = np.array(template2_image.crop(bbox))
            region = template1_image.crop(bbox)
            template2_image.paste(region, bbox)

            content = values[segment_key]['value']

            imp = ImageDraw.Draw(template2_image)

            test_mf = ImageFont.truetype(font_file, int(font_size))
            text_width = imp.textlength(content, font=test_mf)
            text_width = int(math.ceil(text_width))
            xx = int(xx)
            yy = int(yy)
            coord = (xx, yy, xx + text_width, yy + int(font_size))
            coord1 = (xx, yy, text_width, int(font_size))
            textcolor = (int(xc), int(yc), int(zc))

            imp.text((xx, yy), content, textcolor, font=test_mf, stroke_width=int(stroke_width))
            generated_np = np.array(template2_image.crop(bbox))
            sv, _ = ssim(sample_np, generated_np, full=True, multichannel=True, channel_axis=-1)
            ssims.append(sv)
            #pv = psnr(sample_np, generated_np)
            #ssims.append(pv)

            real_name = f"{real_path}/IDNetreal_{filename}"
            fake_name = f"{fake_path}/IDNetfake_{filename}"
            fake_real_name = f"{fake_path}/SIDTDfake_{filename}"
            template2_image.save(real_name, format='JPEG', subsampling=0, quality=quality1)

            shape = coord_to_shape(coord)
            img = np.array(template2_image)
            mask, _ = mask_from_info(img, shape)
            fake_text_image =  inpaint_image(img=img, coord=coord1, mask=mask, text_str=content)

            #Image.fromarray(fake_text_image).save(fake_name, format='JPEG', subsampling=0, quality=quality2)
            Image.fromarray(fake_text_image).save(fake_name, format='JPEG', subsampling=0, quality=quality2)
            test_paths.append([real_name, 0])
            test_paths.append([fake_name, 1])

            real_img = np.array(Image.open(template2).convert("RGB"))
            real_coord = values[segment_key]['bbox']
            real_shape = coord_to_shape(real_coord)
            real_mask, _ = mask_from_info(real_img, real_shape)
            real_coord1 = coord_to_coord1(real_coord)
            fake_real_image =  inpaint_image(img=real_img, coord=real_coord1, mask=real_mask, text_str=content)
            Image.fromarray(fake_real_image).save(fake_real_name)
            sample_paths.append([template2, 0])
            sample_paths.append([fake_real_name, 1])

    #with open("SIDTD_paths_test.json", 'w') as file:
    #    json.dump(sample_paths, file, indent=4)
    #with open("IDNet_paths_test.json", 'w') as file:
    #    json.dump(test_paths, file, indent=4)



    #all_test, _ = eval_model(test_paths, confs)                                                                                                                                                                                                                                                                
    #all_samples, _ = eval_model(sample_paths, confs)                                                                                                                                                                                                                                                                
    #acc = accuracy_score(all_samples, all_test)
    #score = acc + (sum(ssims) / len(ssims))
    ##score = sum(ssims) / len(ssims)
    #print(f"Model acc: {acc}, Evaluation score: {score}")
    #return score

    all_tests = eval_models(test_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
    all_samples = eval_models(sample_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
    accs = [accuracy_score(all_samples[key][0], all_tests[key][0]) for key in all_tests.keys()]
    if with_model:
        score = sum(accs)/len(accs) + (sum(ssims) / len(ssims))
    else:
        score = sum(ssims) / len(ssims)
    print(f"Models:{all_tests.keys()}, Model accs: {accs}, Evaluation score: {score}")
    return score

def evaluate_parameters(xx, yy, font_size, stroke_width, xc, yc, zc,  
                        font_style_idx, save_quality1, save_quality2, segment, confs, testing, candidate_models, with_model, l0, l1):

# add file path to file
    quality1 = int(save_quality1)
    quality2 = int(save_quality2)
    annotation_path = confs['annotation_path']
    template1 = confs['template_path']
    oppath = confs['generated_path']
    font_path = confs["fonts_path"]
    real_path = f"{oppath}/reals"
    fake_path = f"{oppath}/fakes"
    #try:
    #    shutil.rmtree(real_path)
    #    shutil.rmtree(fake_path)
    #    os.mkdir(real_path)
    #    os.mkdir(fake_path)
    #except:
    #    pass
    
    if 'font_files' in confs:
        font_styles = confs['font_files']
    else:
        font_styles = get_font_styles(font_path)
    font_file = font_styles[int(font_style_idx)]

    samples_path = confs['sample_path']
    template1_image = Image.open(template1).convert("RGB")
    area, segment_key = segment.split('_', 1)
    bbox = confs[segment_key]['bbox']
    if testing:
        val_datas = confs['test_data']
    else:
        val_datas = confs['val_data']

    templates = load_all_templates(samples_path, area)
    annotations = load_annotations(annotation_path)
    test_paths = []
    ssims = []
    sample_paths = []
    filename_set = set()
    for filename, values in tqdm(annotations.items()):
        if filename in templates and filename in val_datas and filename not in filename_set:
            filename_set.add(filename)
            template2 = templates[filename]
            template2_image = Image.open(template2).convert("RGB")
            sample_np = np.array(template2_image.crop(bbox))
            region = template1_image.crop(bbox)
            template2_image.paste(region, bbox)

            content = values[segment_key]['value']

            imp = ImageDraw.Draw(template2_image)

            test_mf = ImageFont.truetype(font_file, int(font_size))
            text_width = imp.textlength(content, font=test_mf)
            text_width = int(math.ceil(text_width))
            xx = int(xx)
            yy = int(yy)
            coord = (xx, yy, xx + text_width, yy + int(font_size))
            coord1 = (xx, yy, text_width, int(font_size))
            textcolor = (int(xc), int(yc), int(zc))

            imp.text((xx, yy), content, textcolor, font=test_mf, stroke_width=int(stroke_width))
            generated_np = np.array(template2_image.crop(bbox))
            sv, _ = ssim(sample_np, generated_np, full=True, multichannel=True, channel_axis=-1)
            ssims.append(sv)
            #pv = psnr(sample_np, generated_np)
            #ssims.append(pv)

            real_name = f"{real_path}/newBO_{filename}"
            fake_name = f"{fake_path}/newBO_{filename}"
            fake_real_name = f"{fake_path}/real_{filename}"
            template2_image.save(real_name, format='JPEG', subsampling=0, quality=quality1)

            shape = coord_to_shape(coord)
            img = np.array(template2_image)
            mask, _ = mask_from_info(img, shape)
            fake_text_image =  inpaint_image(img=img, coord=coord1, mask=mask, text_str=content)

            Image.fromarray(fake_text_image).save(fake_name, format='JPEG', subsampling=0, quality=quality2)
            test_paths.append([real_name, 0])
            test_paths.append([fake_name, 1])

            real_img = np.array(Image.open(template2).convert("RGB"))
            real_coord = values[segment_key]['bbox']
            real_shape = coord_to_shape(real_coord)
            real_mask, _ = mask_from_info(real_img, real_shape)
            real_coord1 = coord_to_coord1(real_coord)
            fake_real_image =  inpaint_image(img=real_img, coord=real_coord1, mask=real_mask, text_str=content)
            Image.fromarray(fake_real_image).save(fake_real_name)
            sample_paths.append([template2, 0])
            sample_paths.append([fake_real_name, 1])

    with open("sample_paths_0511.json", 'w') as file:
        json.dump(sample_paths, file, indent=4)
    with open("new_BO_paths_0511.json", 'w') as file:
        json.dump(test_paths, file, indent=4)

    #assert 0

    #real_gau_paths_5 = add_gaussian_noise(test_paths, 0, 5)
    #real_gau_paths_10 = add_gaussian_noise(test_paths, 0, 10)
    #real_gau_paths_15 = add_gaussian_noise(test_paths, 0, 15)
    #real_gau_paths_20 = add_gaussian_noise(test_paths, 0, 20)
    #real_gau_paths_25 = add_gaussian_noise(test_paths, 0, 25)
    #real_gau_paths_30 = add_gaussian_noise(test_paths, 0, 30)
    #real_sp_paths_01 = add_salt_and_pepper_noise(test_paths, 0.01)
    #real_sp_paths_05 = add_salt_and_pepper_noise(test_paths, 0.05)
    #real_sp_paths_10 = add_salt_and_pepper_noise(test_paths, 0.1)
    #real_sp_paths_15 = add_salt_and_pepper_noise(test_paths, 0.15)
    #real_poisson_paths = add_poisson_noise(test_paths)

    #real_pred, real_labels = eval_model(test_paths, confs)                                                                                                                                                                                                                                                                
    #real_pred, real_labels = eval_model(real_gau_paths_5, confs)                                                                                                                                                                                                                                                                
    #real_pred, real_labels = eval_model(real_gau_paths_10, confs)                                                                                                                                                                                                                                                                
    #real_pred, real_labels = eval_model(real_gau_paths_15, confs)                                                                                                                                                                                                                                                                
    #real_pred, real_labels = eval_model(real_gau_paths_20, confs)                                                                                                                                                                                                                                                                
    #real_pred, real_labels = eval_model(real_gau_paths_25, confs)                                                                                                                                                                                                                                                                
    #real_pred, real_labels = eval_model(real_gau_paths_30, confs)                                                                                                                                                                                                                                                                
    #real_pred, real_labels = eval_model(real_sp_paths_01, confs)                                                                                                                                                                                                                                                                
    #real_pred, real_labels = eval_model(real_sp_paths_05, confs)                                                                                                                                                                                                                                                                
    #real_pred, real_labels = eval_model(real_sp_paths_10, confs)                                                                                                                                                                                                                                                                
    #real_pred, real_labels = eval_model(real_sp_paths_15, confs)                                                                                                                                                                                                                                                                
    #real_pred, real_labels = eval_model(real_poisson_paths, confs)                                                                                                                                                                                                                                                                

    #assert 0
    #sample_gau_paths_5 = add_gaussian_noise(sample_paths, 0, 5)
    #sample_gau_paths_10 = add_gaussian_noise(sample_paths, 0, 10)
    #sample_gau_paths_15 = add_gaussian_noise(sample_paths, 0, 15)
    #sample_gau_paths_20 = add_gaussian_noise(sample_paths, 0, 20)
    #sample_gau_paths_25 = add_gaussian_noise(sample_paths, 0, 25)
    #sample_gau_paths_30 = add_gaussian_noise(sample_paths, 0, 30)
    #sample_sp_paths_01 = add_salt_and_pepper_noise(sample_paths, 0.01)
    #sample_sp_paths_05 = add_salt_and_pepper_noise(sample_paths, 0.05)
    #sample_sp_paths_10 = add_salt_and_pepper_noise(sample_paths, 0.1)
    #sample_sp_paths_15 = add_salt_and_pepper_noise(sample_paths, 0.15)
    #sample_poisson_paths = add_poisson_noise(sample_paths)

    #sample_pred, sample_labels = eval_model(sample_paths, confs)                                                                                                                                                                                                                                                                
    #sample_pred, sample_labels = eval_model(sample_gau_paths_5, confs)                                                                                                                                                                                                                                                                
    #sample_pred, sample_labels = eval_model(sample_gau_paths_10, confs)                                                                                                                                                                                                                                                                
    #sample_pred, sample_labels = eval_model(sample_gau_paths_15, confs)                                                                                                                                                                                                                                                                
    #sample_pred, sample_labels = eval_model(sample_gau_paths_20, confs)                                                                                                                                                                                                                                                                
    #sample_pred, sample_labels = eval_model(sample_gau_paths_25, confs)                                                                                                                                                                                                                                                                
    #sample_pred, sample_labels = eval_model(sample_gau_paths_30, confs)                                                                                                                                                                                                                                                                
    #sample_pred, sample_labels = eval_model(sample_sp_paths_01, confs)                                                                                                                                                                                                                                                                
    #sample_pred, sample_labels = eval_model(sample_sp_paths_05, confs)                                                                                                                                                                                                                                                                
    #sample_pred, sample_labels = eval_model(sample_sp_paths_10, confs)                                                                                                                                                                                                                                                                
    #sample_pred, sample_labels = eval_model(sample_sp_paths_15, confs)                                                                                                                                                                                                                                                                
    #sample_pred, sample_labels = eval_model(sample_poisson_paths, confs)                                                                                                                                                                                                                                                                

    #all_test, _ = eval_model(test_paths, confs)                                                                                                                                                                                                                                                                
    #all_samples, _ = eval_model(sample_paths, confs)                                                                                                                                                                                                                                                                
    #acc = accuracy_score(all_samples, all_test)
    #score = acc + (sum(ssims) / len(ssims))
    ##score = sum(ssims) / len(ssims)
    #print(f"Model acc: {acc}, Evaluation score: {score}")
    #return score

    if with_model:
        all_tests = eval_models(test_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
        all_samples = eval_models(sample_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
        accs = [accuracy_score(all_samples[key][0], all_tests[key][0]) for key in all_tests.keys()]
        score = l0 * sum(accs)/len(accs) + l1 * (sum(ssims) / len(ssims))
    else:
        score = sum(ssims) / len(ssims)
    if testing:
        all_tests = eval_models(test_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
        all_samples = eval_models(sample_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
        accs = [accuracy_score(all_samples[key][0], all_tests[key][0]) for key in all_tests.keys()]
        print(f"Models:{all_tests.keys()}, Model consistency: {accs}, Evaluation score: {score}")
    return score
                                                                                                                                                                                                                                                                                                                          


            


"""
def evaluate_parameters(xx, yy, font_size, stroke_width, xc, yc, zc,  
                        font_style_idx, segment_key, confs):
    
    font_file = font_styles[int(font_style_idx)]
    with Image.open(template_path) as template:
        imp = ImageDraw.Draw(template)

        test_mf = ImageFont.truetype(font_file, int(font_size))
        text_width = imp.textlength(content, font=test_mf)
        text_width = int(math.ceil(text_width))
        xx = int(xx)
        yy = int(yy)
        coord = (xx, yy, xx + text_width, yy + int(font_size))
        textcolor = (int(xc), int(yc), int(zc))
        #im2 = template.crop(coord)
        #im2d = ImageDraw.Draw(im2)
        imp.text((xx, yy), content, textcolor, font=test_mf, stroke_width=int(stroke_width))
        im2 = template.crop(coord)
        #im2 = im2.filter(ImageFilter.GaussianBlur(radius=blur_ratio))
        #Image.Image.paste(template, im2, (xx, yy))
        with Image.open(sample_image_path) as sample_image:
            if mode == 1:
                template_np = np.array(template)
                sample_np = np.array(sample_image)

            elif mode == 2:
                im1 = sample_image.crop(coord)
                im2 = template.crop(coord)
                template_np = np.array(im2)
                sample_np = np.array(im1)

            elif mode == 3:
                im1 = sample_image.crop(coord)
                im2 = template.crop(coord)
                im1 = cv2.cvtColor(np.array(im1), cv2.COLOR_RGB2GRAY)
                im2 = cv2.cvtColor(np.array(im2), cv2.COLOR_RGB2GRAY)
                sobelx1 = cv2.Sobel(im1, cv2.CV_64F, 1, 0, ksize=5)  # X direction
                sobely1 = cv2.Sobel(im1, cv2.CV_64F, 0, 1, ksize=5)  # Y direction
                sobel_combined1 = cv2.sqrt(sobelx1**2 + sobely1**2)
                sobelx2 = cv2.Sobel(im2, cv2.CV_64F, 1, 0, ksize=5)  # X direction
                sobely2 = cv2.Sobel(im2, cv2.CV_64F, 0, 1, ksize=5)  # Y direction
                sobel_combined2 = cv2.sqrt(sobelx2**2 + sobely2**2)
                sobel_combined1 = cv2.normalize(
                        sobel_combined1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
                )
                sobel_combined2 = cv2.normalize(
                        sobel_combined2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
                )
                template_np = np.array(sobel_combined2)
                sample_np = np.array(sobel_combined1)

            else:
                print("Mode didnot defined, Please check again!")
                assert 0
            #pv = psnr(sample_np, template_np, data_range=1)
            #print("######################")
            #print(sample_np.shape)
            #print(template_np.shape)
            try:
                sv, _ = ssim(sample_np, template_np, full=True, multichannel=True, channel_axis=-1)
                print(sv)
            except:
                print("SSIM compute error!")
                sv = 0
            return sv
"""

#!/usr/bin/env python
# coding=utf-8
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
import json
from tqdm import tqdm
import time                                                                                                                                                                                             
import torch                                                                                                                                                                                            
from torch.utils.data import Dataset, DataLoader                                                                                                                                                        
from torchvision import transforms                                                                                                                                                                      
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

def get_font_styles(base_walk_path):
    results = []
    for r, d, f in os.walk(base_walk_path):
        f.sort()
        for file in f:
            if file.endswith('.ttf'):
                results.append(base_walk_path + file)
    return results

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

def evaluate_parameters(xx, yy, font_size, stroke_width, xc, yc, zc,  
                        font_style_idx, save_quality, segment, confs, testing, candidate_models, with_model):

# add file path to file
    quality = int(save_quality)
    annotation_path = confs['annotation_path']
    template1 = confs['template_path']
    oppath = confs['generated_path']
    font_path = confs["fonts_path"]
    real_path = f"{oppath}/reals"
    os.makedirs(real_path, exist_ok=True)
    
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
            textcolor = (int(xc), int(yc), int(zc))
            imp.text((xx, yy), content, textcolor, font=test_mf, stroke_width=int(stroke_width))
            generated_np = np.array(template2_image.crop(bbox))
            sv, _ = ssim(sample_np, generated_np, full=True, multichannel=True, channel_axis=-1)
            ssims.append(sv)
            real_name = f"{real_path}/newBO_{filename}"
            template2_image.save(real_name, format='JPEG', subsampling=0, quality=quality)
            test_paths.append([real_name, 0])
            sample_paths.append([template2, 0])
    if with_model:
        all_tests = eval_models(test_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
        all_samples = eval_models(sample_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
        accs = [accuracy_score(all_samples[key][0], all_tests[key][0]) for key in all_tests.keys()]
        score = sum(accs)/len(accs) + (sum(ssims) / len(ssims))
    else:
        score = sum(ssims) / len(ssims)
    if testing:
        all_tests = eval_models(test_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
        all_samples = eval_models(sample_paths, confs, testing, candidate_models)                                                                                                                                                                                                                                                                
        accs = [accuracy_score(all_samples[key][0], all_tests[key][0]) for key in all_tests.keys()]
        print(f"Models:{all_tests.keys()}, Model consistency: {accs}, Evaluation score: {score}")
    return score
                                                                                                                                                                                                                                                                                                                          



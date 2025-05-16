#!/usr/bin/env python
# coding=utf-8
from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageOps, ImageFilter
import math
import json
import os

def get_font_styles(directory):
    files_and_dirs = os.listdir(directory)
    files_and_dirs.sort()

    full_paths = []
    for item in files_and_dirs:
        if item.endswith('.ttf'):
            full_path = os.path.join(directory, item)
            full_paths.append(full_path)
    return full_paths



def write_parameters(xx, yy, font_size, stroke_width, xc, yc, zc,                                                                                                                                                                                                 
                    font_style_idx, content, template, font_styles):                                                                                                                                                                                                          
    font_file = font_styles[font_style_idx]                                                                                                                                                                                                                              
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
    return im2, coord

def region_attributes(value = None, font_style = None, font_size = None, font_color = None, bbox = None):
    attributes = {}
    attributes['value'] = value
    attributes['font_style'] = font_style
    attributes['font_size'] = font_size
    attributes['font_color'] = font_color
    attributes['bbox'] = bbox
    return attributes

def font_choices(fontofchoice, signaturetext, base_x, base_y, fontsize):
    if(fontofchoice=='1_AdamSamuel.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(base_x - (len(signaturetext)/2)) , base_y)
        fontsize = fontsize
 
    elif(fontofchoice=='2_Amanda.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(base_x - (len(signaturetext)/2)) , base_y)
        fontsize = round(0.9 * fontsize)
 
    elif(fontofchoice=='3_Balisa.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(base_x  - (len(signaturetext)/2)) , base_y)
        fontsize = round(0.9 * fontsize)
 
    elif(fontofchoice=='4_daniel.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(base_x - (len(signaturetext)/2)) , base_y)
        fontsize = round(0.85 * fontsize)
 
    elif(fontofchoice=='5_MarioEmma.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(base_x - (len(signaturetext)/2)) , base_y)
        fontsize = round(0.85 * fontsize)
 
    elif(fontofchoice=='6_ReeyRegular.otf'):
        signaturetext = signaturetext[:6]
        textpos = (round(base_x - (len(signaturetext)/2)) , round(1.05*base_y))
        fontsize = round(0.70 * fontsize)
 
    elif(fontofchoice=='7_RememberNight.ttf'):
        signaturetext = signaturetext[:4]
        textpos = (round(round(1*base_x) - (len(signaturetext)/2)) , base_y)
        if signaturetext[0] == "L":
            textpos = (round(round(1*base_x) - (len(signaturetext)/2)) , base_y)
        fontsize = round(0.70 * fontsize)
 
    elif(fontofchoice=='8_Sansilk.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(base_x - (len(signaturetext)/2)) , base_y)
        fontsize = round(1.1 * fontsize)
 
    elif(fontofchoice=='9_Sristian.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(base_x - (len(signaturetext)/2)) , base_y)
        fontsize = round(1.1 * fontsize)
 
    elif(fontofchoice=='10_Stigmature.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(base_x - (len(signaturetext)/2)) , base_y)
        fontsize = round(0.85 * fontsize)
        
    elif(fontofchoice=='11_SweetLovelyRainbowOne.otf'):
        signaturetext = signaturetext[:4]
        textpos = (round(round(0.97*base_x) - (len(signaturetext)/2)) , round(1.01*base_y))
        fontsize = round(0.75 * fontsize)
        
    elif(fontofchoice=='12_TravelNovember.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(base_x - (len(signaturetext)/2)) , base_y)
        fontsize = round(0.9 * fontsize)
 
    elif(fontofchoice=='13_UniqueSurfer.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(base_x - (len(signaturetext)/2)) , base_y)
        fontsize = round(1.15 * fontsize)
 
    elif(fontofchoice=='14_Windsong.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(base_x - (len(signaturetext)/2)) , base_y)
        fontsize = round(1.2 * fontsize)
    return signaturetext, textpos, fontsize

def read_json(path: str):
    with open(path) as f:
        return json.load(f)
def write_json(data:dict, path:str):
    with open(path, "w", encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

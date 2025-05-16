import sys
from utils import *

from PIL import Image, ImageFont, ImageDraw, ImageEnhance, ImageOps
from rembg import remove
import json
import os
import random
import math
import tqdm
import argparse
import os

parser = argparse.ArgumentParser(description="Configurable paths for document generation")

parser.add_argument('--base_path', type=str, required=True, help='Base directory')
parser.add_argument('--area', type=str, required=True, help='Area name (e.g., ALB)')

# Optional overrides (defaults are constructed from base_path + area)
parser.add_argument('--parameters_path', type=str, help='Path to parameters JSON')
parser.add_argument('--jsonpath', type=str, help='Path to the JSON file of searched parameter for each segment')
parser.add_argument('--rmbgpath', type=str, help='Path to background-removed images, for portrait photo')
parser.add_argument('--oppath', type=str, help='Path to the output of the generated images')
parser.add_argument('--templatepath', type=str, help='Path to template image that we want fill the infomation in')
parser.add_argument('--bluebgpath', type=str, help='Path to background image of portrait photo')
parser.add_argument('--fontpath', type=str, help='Path to fonts directory used for generate the signature text')
parser.add_argument('--annotationfile', type=str, help='Path to the output annotation JSON')
parser.add_argument('--font_gen_path', type=str, help='Path to fonts used in PII filling')

args = parser.parse_args()

parameters_path = args.parameters_path 
jsonpath = args.jsonpath 
rmbgpath = args.rmbgpath 
oppath = args.oppath 
templatepath = args.templatepath 
bluebgpath = args.bluebgpath 
fontpath = args.fontpath 
annotationfile = args.annotationfile 
pth = args.font_gen_path 

id_number = 1
jsonfiles = os.listdir(jsonpath)
opfiles = os.listdir(oppath)
fontfiles = os.listdir(fontpath)
font_styles = get_font_styles(pth)
print(font_styles)
with open( parameters_path, 'r') as f:
    parameters = json.load(f)


outerdict = {}


ctr = 1
# print(jsonfiles)
for eachfile in tqdm.tqdm(jsonfiles):

    innerdict = {}
    basenamefile = os.path.basename(eachfile)
    basenamefile = basenamefile[:len(basenamefile) - 5]
    tempfilename = basenamefile + ".png"
    print(f"{ctr} - {tempfilename}")
    ctr += 1

    # is_donor = 'False'
    # is_veteran = 'False'
    data = ""
    tempaddress = jsonpath + eachfile
    with open(tempaddress, "r") as jsonfile:
        data = json.load(jsonfile)
        
    template = Image.open(templatepath)
    bluebg = Image.open(bluebgpath)
    overlay_rmbg = Image.open(rmbgpath + tempfilename)
    width, height = overlay_rmbg.size
    scaling = 1.8
    ghostscaling = 0.8
    newsize = (431, 561)
    ghostsize = (217, 264)
    new_bgsize = (470, 640)
    #newsize = new_bgsize
    r1 = newsize[0]/width
    r2 = newsize[1]/height
    r = max(r1, r2)
    w1 = round(width * r)
    h1 = round(height * r)
    if r1 > r2:
        crop_box = (0,(h1-newsize[1])//2, w1, (h1 + newsize[1])//2 )
    else:
        crop_box = ((w1-newsize[0])//2, 0, (w1 + newsize[0])//2, h1 )
    tmpsize = (round(width * r), round(height * r))
    overlay_rmbg = overlay_rmbg.resize(tmpsize)

    overlay_rmbg = overlay_rmbg.crop(crop_box)
    #overlay_rmbg = overlay_rmbg.resize(newsize)

    white_bg = Image.new("RGBA", overlay_rmbg.size, (255,255,255))
    white_bg.putalpha(200)
    overlay_rmbg = Image.alpha_composite(white_bg,overlay_rmbg)
    overlay_rmbg = overlay_rmbg.convert("L")
    overlay_rmbg = remove(overlay_rmbg)

    # ghostimg = overlay_rmbg.convert("L")
    ghostimg = overlay_rmbg.resize(ghostsize)
    # ghostimg = remove(ghostimg)

    ghostimg2 = ghostimg.copy()
    ghostimg2 = ghostimg2.convert("RGBA")
    r,g,b,a = ghostimg2.split()
    bluetint = a.point(lambda x: x * 0.5)
    ghostimg = Image.merge("RGBA", (r,g,b,bluetint))


    bluebg = bluebg.convert("RGBA")
    bluebg = bluebg.resize(new_bgsize)
    enhancer = ImageEnhance.Contrast(bluebg)
    img_low_contrast = enhancer.enhance(0.7)
    img_with_alpha = img_low_contrast.convert("RGBA")
    alpha_value = 100  # 0 is fully transparent, 255 is fully opaque
    alpha = Image.new("L", img_with_alpha.size, alpha_value)
    img_with_alpha.putalpha(alpha)
    bluebg = img_with_alpha
    trial = bluebg.resize(ghostsize)


    im = ImageDraw.Draw(template)
    mfsizespecial = 6
    mfsize = 25
    mf1size = 64
    mf1 = ImageFont.truetype(f'{base_path}/arial.ttf', mf1size)
    mf2size = 70
    mf2 = ImageFont.truetype(f'{base_path}/arial.ttf', mf2size)

    textcolor = (25,25,25)
    textcolorred = (128,0,0)
    textcolorbday = (129,115,103)
    textcolordln = (0,50,224)
    
    innerdict["fraud"] = "False"

    for key in parameters:                                                                                                                                                                                                                                                
        tmpdict = {}
        if key == 'place_of_birth':
            content = data[key] + "ALB"
        else:
            content = data[key]                                                                                                                                                                                                                                                   
        parameters[key].pop('blur_ratio', None)
        im1, coord = write_parameters(**parameters[key], content=content, template = template, font_styles = font_styles)                                                                                                                                                             
        tmpdict = {}
        tmpdict['value'] = content
        tmpdict['font_style'] = font_styles[parameters[key]['font_style_idx']].split('/')[-1]
        tmpdict['font_size'] = parameters[key]['font_size']
        tmpdict['font_color'] = [parameters[key]['xc'], parameters[key]['yc'], parameters[key]['zc']]
        tmpdict['bbox'] = list(coord)
        innerdict[key] = tmpdict

    template.paste(bluebg, (164,222), mask=bluebg)
    template.paste(overlay_rmbg, (183,242), mask=overlay_rmbg)
    #template.paste(overlay_rmbg, (164,232), mask=white_bg)
    width, height = overlay_rmbg.size                                                                                                                                                                                                                                                                                 
    innerdict['face'] = region_attributes('','','','',[183,242, 164 + width, 232 + height])                                                                                                                                                                                                                           
    template.paste(ghostimg, (1863,510), mask=ghostimg)
    width, height = ghostimg.size                                                                                                                                                                                                                                                                                     
    innerdict['ghostimg'] = region_attributes('','','','',[1863, 510, 1863 + width, 510 + height])                                                                                                                                                                                                                     
    

    id_number += 1
    signaturetext = ""
    extrachar = ['L', 'l', 'I', '']
    random.shuffle(extrachar)
    signaturetext = signaturetext + random.choice(extrachar)
    signaturetext = signaturetext + data["given_name"][0].lower()
    signaturetext = signaturetext + data["surname"].lower()
    random.shuffle(fontfiles)
    fontofchoice = random.choice(fontfiles)
    if(fontofchoice=='1_AdamSamuel.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(290 - (len(signaturetext)/2)) , 970)
        fontsize = 106

    elif(fontofchoice=='2_Amanda.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(290 - (len(signaturetext)/2)) , 970)
        fontsize = 100

    elif(fontofchoice=='3_Balisa.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(290  - (len(signaturetext)/2)) , 970)
        fontsize = 104

    elif(fontofchoice=='4_daniel.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(280 - (len(signaturetext)/2)) , 970)
        fontsize = 98

    elif(fontofchoice=='5_MarioEmma.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(290 - (len(signaturetext)/2)) , 970)
        fontsize = 104

    elif(fontofchoice=='6_ReeyRegular.otf'):
        signaturetext = signaturetext[:6]
        textpos = (round(290 - (len(signaturetext)/2)) , 970)
        fontsize = 100

    elif(fontofchoice=='7_RememberNight.ttf'):
        signaturetext = signaturetext[:4]
        textpos = (round(280 - (len(signaturetext)/2)) , 970)
        if signaturetext[0] == "L":
            textpos = (round(270 - (len(signaturetext)/2)) , 970)
        fontsize = 88

    elif(fontofchoice=='8_Sansilk.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(290 - (len(signaturetext)/2)) , 970)
        fontsize = 108

    elif(fontofchoice=='9_Sristian.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(290 - (len(signaturetext)/2)) , 970)
        fontsize = 104

    elif(fontofchoice=='10_Stigmature.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(290 - (len(signaturetext)/2)) , 970)
        fontsize = 92
        
    elif(fontofchoice=='11_SweetLovelyRainbowOne.otf'):
        signaturetext = signaturetext[:4]
        textpos = (round(290 - (len(signaturetext)/2)) , 990)
        fontsize = 65
        
    elif(fontofchoice=='12_TravelNovember.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(290 - (len(signaturetext)/2)) , 970)
        fontsize = 100

    elif(fontofchoice=='13_UniqueSurfer.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(290 - (len(signaturetext)/2)) , 970)
        fontsize = 108

    elif(fontofchoice=='14_Windsong.ttf'):
        signaturetext = signaturetext[:6]
        textpos = (round(290 - (len(signaturetext)/2)) , 970)
        fontsize = 108

    chosenfontpath = fontpath + fontofchoice
    mf3 = ImageFont.truetype(chosenfontpath,fontsize)
    im.text(textpos, signaturetext, textcolor, font=mf3)
    template.save((oppath + tempfilename), "JPEG", quality=int(random.randint(60, 100)))
    outerdict[tempfilename] = innerdict


write_json(outerdict, annotationfile)


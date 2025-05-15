#!/usr/bin/env python
# coding=utf-8

import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import numpy as np
import math
import json
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from utils import *
import sys
import time
import random


bps = {}
bps['brightness'] = random.uniform(0.5, 1.5) #brightness in ID area
bps['contrast'] = random.uniform(0.5, 1.5) # contrast in ID area
bps['sharpness_factor'] = random.uniform(0.5, 1.5) #shaprness factor
bps['noise_std'] = random.uniform(0.5, 1.5) # noised added on ID
bps['blur_radius'] = random.uniform(0.5, 1.5) # blur added on ID
bps['shadow_offset1'] = int(random.uniform(-3, 3)) # shadow shift in horizoton
bps['shadow_offset2'] = int(random.uniform(-3, 3)) # shadow shift in vertical
bps['shadow_color'] = 128 # shadow darkness
bps['shadow_blur_radius'] = 2 #shadow blurness

#alb
bps['id_resized_shape1'] = 1013 #resized id width
bps['id_resized_shape2'] = 641 #resized in height
bps['top_left1'] = 60 #top left corner length horizone
bps['top_left2'] = 60 #top left corner lenght vertical

bps['top_right1'] = 60   # bottom left corner length horizone
bps['top_right2'] = 65 # bottom left corner length vertical

bps['bottom_left1'] = 60 #top right corner length horizone
bps['bottom_left2'] = 60 # top right corner length vertical

bps['bottom_right1'] = 60 #bottom right corner length horizone
bps['bottom_right2'] = 60 #bottom right corner length vertical
bps['rotate'] = random.uniform(-90, 90) # rotate angle for ID card
bps['position1'] = 1200 # ID position on the paper horizone
bps['position2'] = 1200 # ID position on the paper vertical

# grc
#bps['id_resized_shape1'] = 1417
#bps['id_resized_shape2'] = 1006
#
#bps['top_left1'] = 10
#bps['top_left2'] = 10
#
#bps['top_right1'] = 60   # bottom left
#bps['top_right2'] = 65
#
#bps['bottom_left1'] = 10 #top right
#bps['bottom_left2'] = 10
#
#bps['bottom_right1'] = 60
#bps['bottom_right2'] = 60


if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser(description="ID card processing parameters")

    # Randomizable parameters with default random values
    parser.add_argument('--brightness', type=float, default=random.uniform(1.2, 1.5), help='Brightness in ID area')
    parser.add_argument('--contrast', type=float, default=random.uniform(1.2, 1.5), help='Contrast in ID area')
    parser.add_argument('--sharpness_factor', type=float, default=random.uniform(1.2, 1.5), help='Sharpness factor')
    parser.add_argument('--noise_std', type=float, default=random.uniform(0.5, 1.5), help='Noise added to ID')
    parser.add_argument('--blur_radius', type=float, default=random.uniform(0.5, 1.5), help='Blur added to ID')
    parser.add_argument('--shadow_offset', type=tuple, default=(int(random.uniform(-3, 3)), int(random.uniform(-3, 3))), help='Shadow shift')
    parser.add_argument('--shadow_blur_radius', type=int, default=2, help='Shadow blur radius')
    parser.add_argument('--rotate', type=float, default=random.uniform(-90, 90), help='Rotate angle for ID card')
    parser.add_argument('--position1', type=int, default=int(random.uniform(600, 1300)), help='ID position on the paper horizontally')
    parser.add_argument('--position2', type=int, default=int(random.uniform(600, 1300)), help='ID position on the paper vertically')
    parser.add_argument('--save_quality', type=int, default=int(random.uniform(60, 95)), help='Image quality for saving')

    # Fixed default parameters
    parser.add_argument('--id_resized_shape', type=tuple, default=(1013, 641), help='Resized ID shape')

    parser.add_argument('--shadow_color', type=tuple, default=(0, 0, 0, 128), help='Shadow darkness')
    parser.add_argument('--top_left', type=tuple, default=(60, 60), help='Top-left corner offset')
    parser.add_argument('--top_right', type=tuple, default=(60, 60), help='Top-right corner offset')
    parser.add_argument('--bottom_left', type=tuple, default=(60, 60), help='Bottom-left corner offset')
    parser.add_argument('--bottom_right', type=tuple, default=(60, 60), help='Bottom-right corner offset')


    parser.add_argument('--input_file_path', type=str, default='sample.png', help='Input ID path')
    parser.add_argument('--output_image_folder', type=str, default='tmp', help='Output path')
    parser.add_argument('--output_info_folder', type=str, default='tmp', help='Output path')
    parser.add_argument('--paper_texture_path', type=str, default='papers/1.png', help='paper path')
    parser.add_argument('--prefix_name', type=str, default='scanned', help='prefix name')
    parser.add_argument('--output_json', type=str, default='scanned_info.json', help='prefix name')

    args = parser.parse_args()

    args.input_file_path = "sample.png"
    args.output_folder = './tmp'
    args.paper_texture_path = "papers/1.png"
    args.position1 = int(random.uniform(600, 1300))
    args.position2 = int(random.uniform(600, 1300))

    paper_name = os.path.splitext(os.path.basename(args.paper_texture_path))[0]
    output_name = f"{args.prefix_name}_{paper_name}_{(args.input_file_path).split('/')[-1].rsplit('.')[0]}"
    args.file_name = output_name + '.jpg'
    args.json_name = output_name + '.json'
    output_image_path = os.path.join(args.output_image_folder, args.file_name )
    output_json_path = os.path.join(args.output_info_folder, args.json_name )

    args.brightness = 1.0 + random.uniform(-0.5, 0.5)
    args.contrast = 1.0 + random.uniform(-0.5, 0.5)
    args.sharpness_factor = 1.0 + random.uniform(-0.5, 0.5)
    args.noise_std = random.uniform(0, 5)
    args.blur_radius = random.uniform(0, 1)
    args.shadow_offset = (int(random.uniform(-3, 3)), int(random.uniform(-3, 3)))
    args.shadow_blur_radius = random.uniform(0, 3)
    args.rotate = random.uniform(-90, 90)
    bps_dict = vars(args)
    
    scanned_image = simulate_scan(bps_dict)
    scanned_image.convert("RGB").save(output_image_path, quality=args.save_quality)
    #generate_scanned_images(bps_dict)

    with open(output_json_path, 'w') as f:
            json.dump(bps_dict, f, indent=4)


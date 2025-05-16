#!/usr/bin/env python
# coding=utf-8

import os
import math
import json
from utils import *
import sys
import random
import argparse



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Scanned ID card processing parameters")

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


    parser.add_argument('--input_file_path', type=str, default='../data/scanned_data/sample.png', help='Input ID path')
    parser.add_argument('--output_image_folder', type=str, default='../data/tmp', help='Output image path')
    parser.add_argument('--output_info_folder', type=str, default='../data/tmp', help='Output info path')
    parser.add_argument('--paper_texture_path', type=str, default='../data/scanned_data/1.png', help='paper path')
    parser.add_argument('--prefix_name', type=str, default='scanned', help='prefix name')

    args = parser.parse_args()

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


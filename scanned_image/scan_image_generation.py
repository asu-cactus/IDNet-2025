#!/usr/bin/env python
# coding=utf-8

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import numpy as np
import math
import json
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from generate_training_data import evaluate_parameters_with_frauld
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

candidate_models = []
with_model = 0

best_sv_pv = generate_scanned_images(
                bps['brightness'], bps['contrast'], bps['sharpness_factor'], bps['noise_std'], bps['blur_radius'],
                bps['shadow_offset1'], bps['shadow_offset2'], bps['shadow_color'], bps['shadow_blur_radius'],
                bps['id_resized_shape1'], bps['id_resized_shape2'],
                bps['top_left1'], bps['top_left2'],bps['top_right1'], bps['top_right2'], bps['bottom_left1'], bps['bottom_left2'],bps['bottom_right1'], bps['bottom_right2'],
                testing = False,
                candidate_models = candidate_models,
                with_model = with_model

    
)
def randomly_generation():

def user_defined_generation():


if __name == "__main__":


#!/usr/bin/env python
# coding=utf-8
from itertools import combinations
import os

param_r = 500
param_eta = 3

# Running W/O guided model
for count in range(3):
    c = 'ssim'
    output_name = 'surname_' + c + '_20_' + str(count) + '_' + str(param_r) + '_' + str(param_eta) + '.log'
    os.system(f"python Hyperband_search.py {param_r} {param_eta} 20 0 1 1 {c}> logs_hyperband/{output_name}")

# Running W/ guided model
for count in range(3):
    combinations_list = ['resnet50']
    for c in combinations_list:
        output_name = 'model_name_' + c + '_20_' + str(count) + '_' + str(param_r) + '_' + str(param_eta) + '.log'
        os.system(f"python Hyperband_search.py {param_r} {param_eta} 20 1 1 1 {c}> logs_hyperband/{output_name}")


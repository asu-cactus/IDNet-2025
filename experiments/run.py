#!/usr/bin/env python
# coding=utf-8
from itertools import combinations
import os

models = ['vit-large', 'resnet50', 'inception-v3', 'vgg16', 'densenet']

#for count in range(5):
#    for r in [1]:
#        combinations_object = combinations(models, r)
#        combinations_list = list(combinations_object)
#        for c in combinations_list:
#            print(list(c))
#            c = list(c)
#            output_name = "_".join(c) + '_20_' + str(count) + '.log'
#            args = ' '.join(c)
#            os.system(f"python Bayesian_search.py 20 {args}> logs_new/{output_name}")

for count in range(5):
    c = 'resnet50'

    output_name = 'surname_' + c + '_1_' + str(count) + '.log'
    os.system(f"python Bayesian_search.py 1 1 1 1 {c}> logs_new/{output_name}")

    output_name = 'surname_' + c + '_10_' + str(count) + '.log'
    os.system(f"python Bayesian_search.py 10 1 1 1 {c}> logs_new/{output_name}")

    output_name = 'surname_' + c + '_20_' + str(count) + '.log'
    os.system(f"python Bayesian_search.py 20 1 1 1 {c}> logs_new/{output_name}")

for count in range(3,5):
    c = 'ssim'

    output_name = 'surname_' + c + '_1_' + str(count) + '.log'
    os.system(f"python Bayesian_search.py 1 0 1 1 {c}> logs_new/{output_name}")

    output_name = 'surname_' + c + '_10_' + str(count) + '.log'
    os.system(f"python Bayesian_search.py 10 0 1 1 {c}> logs_new/{output_name}")

    #output_name = 'surname_' + c + '_20_' + str(count) + '.log'
    #os.system(f"python Bayesian_search.py 20 0 {c}> logs_new/{output_name}")
assert 0
for count in range(5):
    combinations_list = ['vit-large', 'resnet50']
    for c in combinations_list:
        #print(list(c))
        #c = list(c)
        output_name = 'ssim_name_' + c + '_20_' + str(count) + '.log'
        os.system(f"python Bayesian_search.py 20 0  1 1 {c}> logs_new/{output_name}")

        output_name = 'model_name_' + c + '_20_' + str(count) + '.log'
        os.system(f"python Bayesian_search.py 20 1 1 1 {c}> logs_new/{output_name}")

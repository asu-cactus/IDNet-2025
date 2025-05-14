#!/bin/bash
python Bayesian_search.py 20 1 0 1 resnet50 >logs/l0_0.log
python Bayesian_search.py 20 1 0 1 resnet50 >logs/l0_1.log
python Bayesian_search.py 20 1 0 1 resnet50 >logs/l0_2.log

python Bayesian_search.py 20 1 0.2 1 resnet50 >logs/l0.2_0.log
python Bayesian_search.py 20 1 0.2 1 resnet50 >logs/l0.2_1.log
python Bayesian_search.py 20 1 0.2 1 resnet50 >logs/l0.2_2.log

python Bayesian_search.py 20 1 0.5 1 resnet50 >logs/l0.5_0.log
python Bayesian_search.py 20 1 0.5 1 resnet50 >logs/l0.5_1.log
python Bayesian_search.py 20 1 0.5 1 resnet50 >logs/l0.5_2.log

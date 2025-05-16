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
#from google_fonts.search_fonts import search_fonts

def get_configs(area):
    with open(f"data/configures/{area}_parameters.json") as f:
        conf = json.load(f)
    return conf

def optimization(pbounds, segment_key, confs, testing, candidate_models, with_model, lambda0, lambda1):

    # Initialize Bayesian Optimization
    optimizer = BayesianOptimization(
        f=lambda xx, yy, font_size, stroke_width, xc, yc, zc, font_style_idx, save_quality: evaluate_parameters(
            xx, yy, font_size, stroke_width, xc, yc, zc, font_style_idx, save_quality,
            segment = segment_key,
            confs = confs,
            testing = testing,
            candidate_models = candidate_models,
            with_model = with_model,
            lambda0 = lambda0,
            lambda1 = lambda1
        ),
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )
    
    # Define the evaluation function for the optimizer
    optimizer.set_gp_params(normalize_y=True)
    
    # Define the acquisition function
    utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)
    
    # Add the evaluation function to the optimizer
    # Perform the optimization
    optimizer.maximize(
        init_points=50,
        n_iter=200,
        acquisition_function=utility
    )

    # Get the best parameters
    best_params = optimizer.max['params']
    print("Best Parameters:", best_params)
    
    return optimizer
    # Evaluate the best parameters one more time to get the final SSIM and PSNR
    best_sv_pv = evaluate_parameters(
        best_params['xx'], best_params['yy'], best_params['font_size'],
        best_params['stroke_width'], best_params['xc'], best_params['yc'],
        best_params['zc'], best_params['font_style_idx'], best_params["save_quality"],
        segment = segment_key,
        confs = confs,
        testing = False,
        candidate_models = candidate_models,
        with_model = with_model,
        lambda0 = lambda0,
        lambda1 = lambda1
    )
    print("Best Evaluation on validation data:", best_sv_pv)
    best_sv_pv = evaluate_parameters(
        best_params['xx'], best_params['yy'], best_params['font_size'],
        best_params['stroke_width'], best_params['xc'], best_params['yc'],
        best_params['zc'], best_params['font_style_idx'], best_params["save_quality"],
        segment = segment_key,
        confs = confs,
        testing = True,
        candidate_models = candidate_models,
        with_model = with_model,
        lambda0 = lambda0,
        lambda1 = lambda1
    )
    print("Best Evaluation on testing data:", best_sv_pv)
    return optimizer

if __name__ == '__main__':
    import argparse

    since = time.time()                                                                                                                                                                                                                                                                                                   
    parser = argparse.ArgumentParser(description="Template configuration.")

    parser.add_argument('--area', type=str, default="ALB", help='Place of template')
    parser.add_argument('--segment', type=str, default="surname", help='Segment of the template')
    parser.add_argument('--target_samples', type=int, default=10, help='Number of target samples to be used')
    parser.add_argument('--with_model', type=int, choices=[0, 1], default=1, help='Whether to use model guided method (1=yes, 0=no)')
    parser.add_argument('--lambda0', type=float, default=1, help='The fractions of similarity score')
    parser.add_argument('--lambda1', type=float, default=1, help='The fractions of consistency score')
    parser.add_argument('--candidate_models', nargs='*', default="resnet50 vit-large", help='List of candidate model names if using model guided method')
    parser.add_argument('--config_info', type=str, default="data/configures/ALB_parameters.json", help='Information about the segment and guided model')
    parser.add_argument('--fonts_path', type=str, default="data/Fonts", help='Information about the segment and guided model')
    parser.add_argument('--output_file', type=str, default="ALB_parameters.json", help='Information about the segment and guided model')

    # Parse the arguments
    args = parser.parse_args()

    # Access variables
    area = args.area
    segment = args.segment
    target_samples = args.target_samples
    with_model = args.with_model
    candidate_models = args.candidate_models if args.candidate_models else []
    if args.with_model and len(args.candidate_models) == 0:
        print("Error: You should provide model name if you use model guided method")

    # Example print (can be removed)
    print(f"Area: {area}")
    print(f"Segment: {segment}")
    print(f"Target Samples: {target_samples}")
    print(f"With Model: {with_model}")
    print(f"Candidate Models: {candidate_models}")
    
    print("candidate_models:", candidate_models)
    segment_key = area + "_" + segment
    with open(args.config_info, 'r') as f:
        confs = json.load(f)

    #font_list = search_fonts(area, segment, 2)
    font_files = []
    #with open("./google_fonts/label_index_new.json") as f:
    #    label_indexs = json.load(f)
    #for fl in font_list:
    #    font_files.append(os.path.join('./google_fonts/newfonts', label_indexs[str(fl)]))
    for fl in os.listdir(args.fonts_path):
        if fl.endswith('.ttf'):
            font_files.append(os.path.join(args.fonts_path, fl))
    confs['font_files'] = font_files

    val_datas = []
    with open(confs['training_samples']) as f:
        ts = json.load(f)
    for v in ts['val']:
        if v[1] == 0:
            name = v[0].split('/')[-1]
            if name.split('_')[0].upper() == area:
                val_datas.append(name)
    confs['val_data'] = val_datas[:target_samples]

    val_datas = []
    for v in ts['test']:
        if v[1] == 0:
            name = v[0].split('/')[-1]
            if name.split('_')[0].upper() == area:
                val_datas.append(name)
    confs['test_data'] = val_datas

    # Define the bounds for the parameters
    search_parameters = confs[segment]
    pbounds = {
        'xx': (search_parameters['xx'] - 5, search_parameters['xx'] + 5),
        'yy': (search_parameters['yy'] - 5, search_parameters['yy'] + 5),
        'font_size': (search_parameters['font_size'] - 5, search_parameters['font_size'] + 5),
        'stroke_width': (0, 1),
        'xc': (max(0, search_parameters['xc'] - 10), min(255, search_parameters['xc'] + 10)),
        'yc': (max(0, search_parameters['yc'] - 10), min(255, search_parameters['yc'] + 10)),
        'zc': (max(0, search_parameters['zc'] - 10), min(255, search_parameters['zc'] + 10)),
        'font_style_idx': (0, len(font_files) - 1),
        'save_quality': (60, 100),
    }
    optimizer = optimization(pbounds = pbounds, segment_key = segment_key, confs = confs, testing = False, candidate_models = candidate_models, with_model = with_model, lambda0 = lambda0, lambda1 = lambda1)
    bps = optimizer.max['params']
    tmp = {}
    tmp['x_position'] = int(bps['xx'])
    tmp['y_position'] = int(bps['yy'])
    tmp['font_size'] = int(bps['font_size'])
    tmp['stroke_width'] = int(bps['stroke_width'])
    tmp['x_color'] = int(bps['xc'])
    tmp['y_color'] = int(bps['yc'])
    tmp['z_color'] = int(bps['zc'])
    tmp['font_style'] = font_files[int(bps['font_style_idx'])].split('/')[-1] 
    tmp['save_quality'] = int(bps['save_quality'])

    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r') as f:
                configures = json.load(f)
        except:
            configures = {}
    else:
        configures = {}
    if 'segments' not in configures:
        configures['segments'] = {}
    configures['segments'][args.segment] = tmp
    with open(args.output_file, 'w') as f:
        json.dump(configures, f, indent = 4)
    time_elapsed = time.time() - since                                                                                                                                                                                                                                                                                    
    print('Bayesian complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))                                                                                                                                                                                                                         


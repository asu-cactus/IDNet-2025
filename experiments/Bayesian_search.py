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
from google_fonts.search_fonts import search_fonts

def get_configs(area):
    with open(f"datas/configures/{area}_parameters.json") as f:
        conf = json.load(f)
    return conf

def optimization(pbounds, segment_key, confs, testing, candidate_models, with_model, l0, l1):

    # Initialize Bayesian Optimization
    optimizer = BayesianOptimization(
        f=lambda xx, yy, font_size, stroke_width, xc, yc, zc, font_style_idx, save_quality1, save_quality2: evaluate_parameters(
            xx, yy, font_size, stroke_width, xc, yc, zc, font_style_idx, save_quality1, save_quality2,
            segment = segment_key,
            confs = confs,
            testing = testing,
            candidate_models = candidate_models,
            with_model = with_model,
            l0 = l0,
            l1 = l1
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
    '''
    optimizer.maximize(
        init_points=10,
        n_iter=30,
        acq='ei',  # Expected Improvement
        f=lambda xx, yy, font_size, stroke_width, xc, yc, zc, blur_ratio: evaluate_parameters(
            xx, yy, font_size, stroke_width, xc, yc, zc, blur_ratio,
            template_path='AZ_Template.png',
            sample_image_path='AZ_Sample.png',
            font_file='fonts/Arial.ttf',
            list_4d=['D02141248']
        )
    )
    '''
    
    # Get the best parameters
    best_params = optimizer.max['params']
    print("Best Parameters:", best_params)
    
    # Evaluate the best parameters one more time to get the final SSIM and PSNR
    best_sv_pv = evaluate_parameters(
        best_params['xx'], best_params['yy'], best_params['font_size'],
        best_params['stroke_width'], best_params['xc'], best_params['yc'],
        best_params['zc'], best_params['font_style_idx'], best_params["save_quality1"], best_params["save_quality2"],
        segment = segment_key,
        confs = confs,
        testing = False,
        candidate_models = candidate_models,
        with_model = with_model,
        l0 = l0,
        l1 = l1
    )
    print("Best Evaluation on validation data:", best_sv_pv)
    best_sv_pv = evaluate_parameters(
        best_params['xx'], best_params['yy'], best_params['font_size'],
        best_params['stroke_width'], best_params['xc'], best_params['yc'],
        best_params['zc'], best_params['font_style_idx'], best_params["save_quality1"], best_params["save_quality2"],
        segment = segment_key,
        confs = confs,
        testing = True,
        candidate_models = candidate_models,
        with_model = with_model,
        l0 = l0,
        l1 = l1
    )
    print("Best Evaluation on testing data:", best_sv_pv)
    return optimizer

if __name__ == '__main__':
    
    since = time.time()                                                                                                                                                                                                                                                                                                   
    area = "ALB"
    segment = 'surname' 
    target_samples = int(sys.argv[1])
    with_model = int(sys.argv[2])
    lambda0 = float(sys.argv[3])
    lambda1 = float(sys.argv[4])
    candidate_models = sys.argv[5:]
    print("candidate_models:", candidate_models)
    segment_key = area + "_" + segment
    confs = get_configs(area)
    surname_font_list = [(106, 513), (122, 127), (64, 70), (103, 53), (69, 50), (115, 36), (59, 32), (12, 26), (104, 23), (105, 22), (81, 20), (65, 11), (50, 5), (47, 4), (129, 4), (142, 2), (97, 1), (75, 1)]
    #name_font_list = [(10, 833), (115, 103), (150, 40), (101, 8), (103, 8), (61, 2), (98, 2), (100, 2), (90, 1), (12, 1)]
    font_list = locals()[f"{segment}_font_list"]
    #font_list = search_fonts(area, segment, 2)
    font_files = []
    with open("./google_fonts/label_index_new.json") as f:
        label_indexs = json.load(f)
    for fl in font_list:
        font_files.append(os.path.join('./google_fonts/newfonts', label_indexs[str(fl[0])]))
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
    pbounds = {
        'xx': (690, 720),
        'yy': (220, 250),
        'font_size': (65,75),
        'stroke_width': (0,1),
        'xc': (0, 20),
        'yc': (0, 20),
        'zc': (0, 20),
        'font_style_idx': (0, len(font_files) - 1),
        'save_quality1': (60, 100),
        'save_quality2': (60, 100)
    }
    optimization(pbounds = pbounds, segment_key = segment_key, confs = confs, testing = False, candidate_models = candidate_models, with_model = with_model, l0 = lambda0, l1 = lambda1)
    time_elapsed = time.time() - since                                                                                                                                                                                                                                                                                    
    print('Bayesian complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))                                                                                                                                                                                                                         


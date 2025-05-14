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

def get_configs(area):
    with open(f"datas/configures/{area}_parameters.json") as f:
        conf = json.load(f)
    return conf

def optimization(pbounds, segment_key, confs, testing):

    # Initialize Bayesian Optimization
    optimizer = BayesianOptimization(
        f=lambda xx, yy, font_size, stroke_width, xc, yc, zc, font_style_idx, save_quality1, save_quality2: evaluate_parameters(
            xx, yy, font_size, stroke_width, xc, yc, zc, font_style_idx, save_quality1, save_quality2,
            segment = segment_key,
            confs = confs,
            testing = testing,
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
    )
    print("Best Evaluation on validation data:", best_sv_pv)
    best_sv_pv = evaluate_parameters(
        best_params['xx'], best_params['yy'], best_params['font_size'],
        best_params['stroke_width'], best_params['xc'], best_params['yc'],
        best_params['zc'], best_params['font_style_idx'], best_params["save_quality1"], best_params["save_quality2"],
        segment = segment_key,
        confs = confs,
        testing = True,
    )
    print("Best Evaluation on testing data:", best_sv_pv)
    return optimizer

if __name__ == '__main__':
    
    dataset = sys.argv[1]
    area = "ALB"
    target_samples = 10
    segment_key = area + "_" + "surname"
    confs = get_configs(area)
    font_list = [(106, 513), (122, 127), (64, 70), (103, 53), (69, 50), (115, 36), (59, 32), (12, 26), (104, 23), (105, 22), (81, 20), (65, 11), (50, 5), (47, 4), (129, 4), (142, 2), (97, 1), (75, 1)]
    font_files = []
    with open("./google_fonts/label_index_new.json") as f:
        label_indexs = json.load(f)
    for fl in font_list:
        font_files.append(os.path.join('./google_fonts/newfonts', label_indexs[str(fl[0])]))
    confs['font_files'] = font_files

    val_datas = []
    with open(confs['training_samples']) as f:
        ts = json.load(f)

    for v in ts['train']:
        if v[1] == 0:
            name = v[0].split('/')[-1]
            if name.split('_')[0].upper() == area:
                val_datas.append(name)
    for v in ts['val']:
        if v[1] == 0:
            name = v[0].split('/')[-1]
            if name.split('_')[0].upper() == area:
                val_datas.append(name)
    #confs['val_data'] = val_datas[:target_samples]
    confs['val_data'] = val_datas

    val_datas = []
    for v in ts['test']:
        if v[1] == 0:
            name = v[0].split('/')[-1]
            if name.split('_')[0].upper() == area:
                val_datas.append(name)
    confs['test_data'] = val_datas
    confs['save_path'] = "idnet_new_results.csv"


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
    #optimization(pbounds = pbounds, segment_key = segment_key, confs = confs, testing = False)
    # best parameter for val_20
    #best_params = {'font_size': 68.62697592465854, 'font_style_idx': 2.783675434759178, 'save_quality1': 94.85609783351282, 'save_quality2': 60.84373867308548, 'stroke_width': 0.0, 'xc': 13.9517517090142, 'xx': 701.3437951269875, 'yc': 10.508349994381451, 'yy': 229.1105583679292, 'zc': 14.564671944905843}
    #best parameter for val 1
    #best_params = {'font_size': 74.27058360425606, 'font_style_idx': 12.387104304610082, 'save_quality1': 92.75986590460452, 'save_quality2': 63.93488491197379, 'stroke_width': 0.06351251677938197, 'xc': 5.758904076597606, 'xx': 711.0867296873433, 'yc': 3.3929076990073748, 'yy': 241.38907197713823, 'zc': 0.8835240240241737}
    #best parameter for val 1 new
    #best_params = {'font_size': 69.69599301181329, 'font_style_idx': 14.89886591197567, 'save_quality1': 72.50376035127091, 'save_quality2': 61.12388402194125, 'stroke_width': 0.48883629724817146, 'xc': 11.68268601682228, 'xx': 713.0018210236373, 'yc': 18.098894901648915, 'yy': 228.60495764764107, 'zc': 2.3996409029610666}

    #best parameter for old bo
    #best_params = {'font_size': 68, 'font_style_idx': 1, 'save_quality1': 72, 'save_quality2': 61.12388402194125, 'stroke_width': 1, 'xc': 25, 'xx': 713.0018210236373, 'yc': 18.098894901648915, 'yy': 228.60495764764107, 'zc': 2.3996409029610666}
    #best_params = {'font_size': 74.09762130184448, 'font_style_idx': 0.5711064266542756, 'save_quality1': 89.70065468269782, 'save_quality2': 85.67798635487873, 'stroke_width': 0.6299610064045151, 'xc': 6.1773675213692725, 'xx': 706.8361935592A}
    best_params = {'font_size': 74.09762130184448, 'font_style_idx': 0.5711064266542756, 'save_quality1': 89.70065468269782, 'save_quality2': 85.67798635487873, 'stroke_width': 0.6299610064045151, 'xc': 6.1773675213692725, 'xx': 706.8361935592151, 'yc': 15.994226745444418, 'yy': 235.1988988792965, 'zc': 3.870191409951249}

    candidate_models = []
    #best_sv_pv = evaluate_parameters_custom(
    best_sv_pv = evaluate_cyclegan1(
        best_params['xx'], best_params['yy'], best_params['font_size'],
        best_params['stroke_width'], best_params['xc'], best_params['yc'],
        best_params['zc'], best_params['font_style_idx'], best_params["save_quality1"], best_params["save_quality2"],
        segment = segment_key,
        confs = confs,
        testing = True,
        candidate_models = candidate_models,
        with_model = False,
        dataset = dataset
    )
    print("Best Evaluation on testing data:", best_sv_pv)


from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import numpy as np
import math
import json
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from evaluate_parameters import evaluate_parameters_with_fraud
import sys
import time

def get_configs(area):
    with open(f"datas/configures/{area}_parameters.json") as f:
        conf = json.load(f)
    return conf

def optimization(pbounds, target_samples, testing, candidate_models, with_model):

    # Initialize Bayesian Optimization
    optimizer = BayesianOptimization(
        f = lambda brightness, contrast, sharpness_factor, noise_std, blur_radius, shadow_offset1, shadow_offset2, 
        shadow_color, shadow_blur_radius, id_resized_shape1, id_resized_shape2, 
        top_left1, top_left2, top_right1, top_right2, bottom_left1, bottom_left2, bottom_right1, bottom_right2,
        save_quality1, save_quality2: evaluate_parameters_with_fraud(
            brightness, contrast, sharpness_factor, noise_std, blur_radius, shadow_offset1, shadow_offset2, 
                  shadow_color, shadow_blur_radius, id_resized_shape1, id_resized_shape2,
            top_left1, top_left2, top_right1, top_right2, bottom_left1, bottom_left2, bottom_right1, bottom_right2,
            save_quality1, save_quality2,
            target_samples = target_samples, testing = testing,
            candidate_models = candidate_models,
            with_model = with_model
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
        init_points=20,
        n_iter=50,
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
    bps = optimizer.max['params']
    print("Best Parameters:", bps)
    
    # Evaluate the best parameters one more time to get the final SSIM and PSNR
    best_sv_pv = evaluate_parameters_with_fraud(
        bps['brightness'], bps['contrast'], bps['sharpness_factor'], bps['noise_std'], bps['blur_radius'], 
        bps['shadow_offset1'], bps['shadow_offset2'], bps['shadow_color'], bps['shadow_blur_radius'], 
        bps['id_resized_shape1'], bps['id_resized_shape2'], 
        bps['top_left1'], bps['top_left2'],bps['top_right1'], bps['top_right2'], 
        bps['bottom_left1'], bps['bottom_left2'],bps['bottom_right1'], bps['bottom_right2'],
        bps['save_quality1'], bps['save_quality2'],
        target_samples = target_samples, testing = False,
        candidate_models = candidate_models,
        with_model = with_model
    )
    print("Best Evaluation on validation data:", best_sv_pv)
    best_sv_pv = evaluate_parameters_with_fraud(
        bps['brightness'], bps['contrast'], bps['sharpness_factor'], bps['noise_std'], bps['blur_radius'], 
        bps['shadow_offset1'], bps['shadow_offset2'], bps['shadow_color'], bps['shadow_blur_radius'], 
        bps['id_resized_shape1'], bps['id_resized_shape2'], 
        bps['top_left1'], bps['top_left2'],bps['top_right1'], bps['top_right2'], 
        bps['bottom_left1'], bps['bottom_left2'],bps['bottom_right1'], bps['bottom_right2'],
        bps['save_quality1'], bps['save_quality2'],
        target_samples = target_samples, testing = True,
        candidate_models = candidate_models,
        with_model = with_model
    )

    print("Best Evaluation on testing data:", best_sv_pv)
    return optimizer

if __name__ == '__main__':
    
    since = time.time()                                                                                                                                                                                                                                                                                                   
    target_samples = int(sys.argv[1])
    with_model = int(sys.argv[2])
    candidate_models = sys.argv[3:]
    print("candidate_models:", candidate_models)

    # Define the bounds for the parameters
    pbounds = {
        'brightness': (0.5, 1.5),
        'contrast': (0.5, 1.5),
        'sharpness_factor': (0.5, 1.5),
        'noise_std': (0, 5),
        'blur_radius': (0, 1),
        'shadow_offset1':(-10, 10),
        'shadow_offset2':(-10, 10),
        'shadow_color':(0, 128),
        'shadow_blur_radius': (0, 5),
        'id_resized_shape1':(1008, 1018),
        'id_resized_shape2':(636, 646),
        'top_left1': (1, 100),
        'top_left2': (1, 100),
        'top_right1': (1, 100),
        'top_right2': (1, 100),
        'bottom_left1': (1, 100),
        'bottom_left2': (1, 100),
        'bottom_right1': (1, 100),
        'bottom_right2': (1, 100),
        'save_quality1': (60, 100),
        'save_quality2': (60, 100)
    }
    optimization(pbounds = pbounds, target_samples = target_samples, testing = False, candidate_models = candidate_models, with_model = with_model)
    time_elapsed = time.time() - since                                                                                                                                                                                                                                                                                    
    print('Bayesian complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))                                                                                                                                                                                                                         


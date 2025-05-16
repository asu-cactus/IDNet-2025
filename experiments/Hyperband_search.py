from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import numpy as np
import math
import json
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from utils import *
import sys
import time
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Dict, Any, Callable

class HyperbandOptimizer:
    def __init__(self, pbounds: Dict[str, Tuple[float, float]], 
                 evaluate_func: Callable, 
                 max_iter: int = 100, 
                 eta: int = 3,
                 timeout: int = 60):
        """
        Robust Hyperband optimizer with guaranteed score-config synchronization.
        
        Args:
            pbounds: Parameter bounds dictionary {name: (min, max)}
            evaluate_func: Function to evaluate configurations
            max_iter: Maximum resources per configuration
            eta: Aggressiveness factor (default=3)
            timeout: Maximum seconds allowed per evaluation (default=60)
        """
        self.pbounds = pbounds
        self.evaluate_func = evaluate_func
        self.max_iter = max_iter
        self.eta = eta
        self.timeout = timeout
        self.best_score = -float('inf')
        self.best_params = None
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate Hyperband parameters."""
        assert self.eta >= 2, "eta must be ≥2 to ensure proper halving"
        assert self.max_iter >= 1, "max_iter must be ≥1"
        assert self.timeout > 0, "timeout must be positive"
        
    def random_config(self) -> Dict[str, Any]:
        """Generate a random configuration within bounds."""
        config = {}
        for param, bounds in self.pbounds.items():
            '''if isinstance(bounds[0], int):
                config[param] = random.randint(bounds[0], bounds[1])
            else:
                config[param] = random.uniform(bounds[0], bounds[1])'''
            config[param] = random.uniform(bounds[0], bounds[1])
        return config
    
    def run(self) -> Tuple[Dict[str, Any], float]:
        """
        Run Hyperband optimization with robust error handling.
        
        Returns:
            tuple: (best_params, best_score)
        """
        s_max = int(math.log(self.max_iter) / math.log(self.eta))
        B = (s_max + 1) * self.max_iter
        
        for s in reversed(range(s_max + 1)):
            n = int(math.ceil((B / self.max_iter) * (self.eta**s) / (s + 1)))
            r = self.max_iter * self.eta**(-s)
            T = [self.random_config() for _ in range(n)]
            
            print(f"Bracket s={s}: Starting with {n} configurations")
            
            for i in range(s + 1):
                n_i = len(T)
                if n_i == 0:
                    print(f"Bracket s={s}: No configs remaining, skipping")
                    break
                    
                r_i = int(r * self.eta**i)
                print(f"Round i={i}: Evaluating {n_i} configs with resource {r_i}")
                
                # Synchronous evaluation with error handling
                valid_scores = []
                valid_configs = []
                
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for config in T:
                        args = (
                            config['xx'],
                            config['yy'],
                            config['font_size'],
                            config['stroke_width'],
                            config['xc'],
                            config['yc'],
                            config['zc'],
                            config['font_style_idx'],
                            config['save_quality1'],
                            config['save_quality2']
                        )
                        futures.append(executor.submit(self.evaluate_func, *args))
                    
                    for config, future in zip(T, futures):
                        try:
                            result = future.result(timeout=self.timeout)
                            score = result[0] if isinstance(result, tuple) else result
                            if score is not None:
                                valid_scores.append(score)
                                valid_configs.append(config)
                            else:
                                print(f"Warning: Config {config} returned None score")
                        except TimeoutError:
                            print(f"Timeout evaluating config {config}")
                        except Exception as e:
                            print(f"Error evaluating config {config}: {str(e)}")
                
                if not valid_scores:
                    print(f"Bracket s={s} round i={i}: All evaluations failed!")
                    T = []
                    break
                    
                # Ensure we keep at least 1 configuration
                k = max(int(math.ceil(len(valid_configs) / self.eta)), 1)
                top_indices = np.argsort(valid_scores)[-k:]
                T = [valid_configs[i] for i in top_indices]
                
                # Update best parameters
                current_best_idx = np.argmax(valid_scores)
                current_score = valid_scores[current_best_idx]
                
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_params = valid_configs[current_best_idx]
                    print(f"New best score: {self.best_score:.4f}")
                
                print(f"Kept {len(T)}/{n_i} configs in this round")
        
        return self.best_params, self.best_score



def get_configs(area):
    with open(f"data/configures/{area}_parameters.json") as f:
        conf = json.load(f)
    return conf



def optimization(pbounds, segment_key, confs, testing, candidate_models, with_model, param_r, param_eta, lambda0, lambda1):
    # Create evaluation closure that captures all needed parameters
    def evaluator(xx, yy, font_size, stroke_width, xc, yc, zc, 
                 font_style_idx, save_quality1, save_quality2):
        return evaluate_parameters(
            xx, yy, font_size, stroke_width, xc, yc, zc,
            font_style_idx, save_quality1, save_quality2,
            segment=segment_key,
            confs=confs,
            testing=testing,
            candidate_models=candidate_models,
            with_model=with_model,
            l0 = lambda0,
            l1 = lambda1
        )
    
    # Initialize Hyperband with the wrapped evaluator
    hyperband = HyperbandOptimizer(
        pbounds=pbounds,
        evaluate_func=evaluator,
        max_iter=param_r,
        eta=param_eta,
        timeout=120
    )
    
    # Run optimization - no arguments needed now
    best_params, best_score = hyperband.run()
    
    print("Best Parameters:", best_params)
    print("Best Score:", best_score)
    
    # Evaluate the best parameters on validation and test data
    best_sv_pv = evaluate_parameters(
        best_params['xx'], best_params['yy'], best_params['font_size'],
        best_params['stroke_width'], best_params['xc'], best_params['yc'],
        best_params['zc'], best_params['font_style_idx'], best_params["save_quality1"], best_params["save_quality2"],
        segment=segment_key,
        confs=confs,
        testing=False,
        candidate_models=candidate_models,
        with_model=with_model,
        l0 = lambda0,
        l1 = lambda1
    )
    print("Best Evaluation on validation data:", best_sv_pv)
    
    best_sv_pv = evaluate_parameters(
        best_params['xx'], best_params['yy'], best_params['font_size'],
        best_params['stroke_width'], best_params['xc'], best_params['yc'],
        best_params['zc'], best_params['font_style_idx'], best_params["save_quality1"], best_params["save_quality2"],
        segment=segment_key,
        confs=confs,
        testing=True,
        candidate_models=candidate_models,
        with_model=with_model,
        l0 = lambda0,
        l1 = lambda1
    )
    print("Best Evaluation on testing data:", best_sv_pv)
    
    return hyperband
    

if __name__ == '__main__':
    since = time.time()
    area = "ALB"
    segment = 'surname'
    param_r = int(sys.argv[1])
    param_eta = int(sys.argv[2])
    target_samples = int(sys.argv[3])
    with_model = int(sys.argv[4])
    lambda0 = float(sys.argv[5])
    lambda1 = float(sys.argv[6])
    candidate_models = sys.argv[7:]
    print("candidate_models:", candidate_models)
    segment_key = area + "_" + segment
    confs = get_configs(area)
    
    surname_font_list = [(106, 513), (122, 127), (64, 70), (103, 53), (69, 50), (115, 36), (59, 32), (12, 26), (104, 23), (105, 22), (81, 20), (65, 11), (50, 5), (47, 4), (129, 4), (142, 2), (97, 1), (75, 1)]
    name_font_list = [(10, 833), (115, 103), (150, 40), (101, 8), (103, 8), (61, 2), (98, 2), (100, 2), (90, 1), (12, 1)]
    font_list = locals()[f"{segment}_font_list"]
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
    
    optimization(pbounds=pbounds, segment_key=segment_key, confs=confs, testing=False, candidate_models=candidate_models, with_model=with_model, param_r=param_r, param_eta=param_eta, lambda0=lambda0, lambda1=lambda1)
    time_elapsed = time.time() - since
    print('Hyperband complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

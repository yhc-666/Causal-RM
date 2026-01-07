import optuna
import argparse
import yaml
import warnings
import tensorflow as tf
from trainer import Trainer
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,5"  # 指定使用的 GPU 编号
config_file_path = "../conf/config.yaml"

# 读取配置文件
with open(config_file_path, 'r') as file:
    config = yaml.safe_load(file)

def objective(trial):
     # 优化 eta 参数
    model_name = config['model_name']
    params = {}

    # Ensure every model has an eta
    # params['eta'] = eta  # Always set eta
    params['max_iters'] = config['max_iters']  # 从配置文件读取
    params['batch_size'] = config['batch_size']  # 从配置文件读取
    params['sample_times'] = config['sample_times']  # 从配置文件读取
    # Suggest model-specific hyperparameters
    if model_name == 'bpr':
        params['eta'] = 0.001
        params['dim'] = trial.suggest_categorical('dim', [32, 64, 128, 256])
        params['lam'] = trial.suggest_categorical('lam', [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
    elif model_name == 'ours1_bpr':
        params['eta'] = 0.001
        params['dim'] = trial.suggest_categorical('dim', [32, 64, 128, 256])
        params['lam1'] = trial.suggest_categorical('lam1', [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])     
        params['lam2'] = trial.suggest_categorical('lam2', [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100])
        params['lam3'] = trial.suggest_categorical('lam3', [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100])  
        params['ind'] = trial.suggest_categorical('ind', [0, 1])
        params['weight'] = trial.suggest_float('weight', 0, 1)      
    elif model_name == 'ours2_wmf':
        params['eta'] = 0.001
        params['dim'] = trial.suggest_categorical('dim', [32, 64, 128, 256])
        params['lam1'] = trial.suggest_categorical('lam1', [1e-8, 5e-8, 1e-7, 5e-7, 1e-6])     
        params['lam2'] = trial.suggest_categorical('lam2', [0.01])
        params['lam3'] = trial.suggest_categorical('lam3', [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100])        
        params['weight'] = trial.suggest_categorical('weight', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        params['clip'] = trial.suggest_categorical('clip', [0.005, 0.01, 0.015, 0.02])
    elif model_name == 'relmf':
        params['eta'] = 0.001
        params['dim'] = trial.suggest_categorical('dim', [32, 64, 128, 256])
        params['lam'] = trial.suggest_categorical('lam', [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
        params['clip'] = trial.suggest_categorical('clip', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        params['dual_unbias'] = False
    elif model_name == 'relmf_du':
        params['eta'] = 0.001
        params['dim'] = trial.suggest_categorical('dim', [32, 64, 128, 256])
        params['lam'] = trial.suggest_categorical('lam', [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
        params['clip'] = trial.suggest_categorical('clip', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        params['dual_unbias'] = True
    elif model_name == 'ubpr':
        params['eta'] = 0.001
        params['dim'] = trial.suggest_categorical('dim', [32, 64, 128, 256])
        params['lam'] = trial.suggest_categorical('lam', [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
        params['beta'] = trial.suggest_categorical('beta', [0.1, 0.0])
    elif model_name == 'wmf':
        params['eta'] = 0.001
        params['dim'] = trial.suggest_categorical('dim', [32, 64, 128, 256])
        params['lam'] = trial.suggest_categorical('lam', [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
        params['weight'] = trial.suggest_categorical('weight', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        params['dual_unbias'] = False
    elif model_name == 'upl_bpr':
        params['eta'] = 0.001
        params['dim'] = trial.suggest_categorical('dim', [32, 64, 128, 256])
        params['lam'] = trial.suggest_categorical('lam', [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
        params['beta'] = trial.suggest_categorical('beta', [0.1, 0.0])
        params['pair_weight'] = 0
    elif model_name == 'Ours':
        # params['lambda1'] = trial.suggest_categorical('lambda1', [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
        # params['lambda2'] = trial.suggest_categorical('lambda2', [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
        # params['eta'] = 0.001
        # params['weight_decay1'] = trial.suggest_categorical('weight_decay1', [1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2])
        # params['weight_decay2'] = trial.suggest_categorical('weight_decay2', [1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2])
        
        
        # # Embedding dimensions and other parameters
        # params['embedding_dim'] = trial.suggest_categorical('embedding_dim', [32, 64, 128, 256])
        

        params['lambda1'] = trial.suggest_categorical('lambda1', [1e-7])
        params['lambda2'] = trial.suggest_categorical('lambda2', [1e-7])
        params['eta'] = 0.001
        params['weight_decay1'] = trial.suggest_categorical('weight_decay1', [1e-6])
        params['weight_decay2'] = trial.suggest_categorical('weight_decay2', [1e-6])
        
        
        # Embedding dimensions and other parameters
        params['embedding_dim'] = trial.suggest_categorical('embedding_dim', [32])        
        
        
        params['nu'] = 0.2  # 定义 nu 参数的范围
        params['percentile'] = trial.suggest_categorical('percentile', [30])  # 定义 percentile 参数的范围 
        params['HN_percentile'] = trial.suggest_categorical('HN_percentile', [30])
        params['subsample'] = 10000  # BPR等损失的子采样大小
        
        # Random number of hidden layers and their sizes for representation layers
        # num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 3)  # 随机选择 1 到 5 层
        # params['hidden_dim'] = [trial.suggest_categorical(f'hidden_layer_{i}', [32, 64, 128, 256]) for i in range(num_hidden_layers)]

        # # Random number of hidden layers and their sizes for prediction layers
        # num_prediction_layers = trial.suggest_int('num_prediction_layers', 1, 3)  # 随机选择 1 到 3 层
        # params['prediction_hidden_dim'] = [trial.suggest_categorical(f'prediction_layer_{i}', [32, 64, 128, 256]) for i in range(num_prediction_layers)]

        num_hidden_layers = trial.suggest_int('num_hidden_layers', 1,1)  # 随机选择 1 到 5 层
        params['hidden_dim'] = [trial.suggest_categorical(f'hidden_layer_{i}', [32]) for i in range(num_hidden_layers)]

        # Random number of hidden layers and their sizes for prediction layers
        num_prediction_layers = trial.suggest_int('num_prediction_layers', 1,1)  # 随机选择 1 到 3 层
        params['prediction_hidden_dim'] = [trial.suggest_categorical(f'prediction_layer_{i}', [32]) for i in range(num_prediction_layers)]

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Initialize the Trainer class with parameters
    trainer = Trainer(
        data=config['data'],  # 从配置文件读取
        batch_size=config['batch_size'],  # 从配置文件中读取
        max_iters=config['max_iters'],  # 从配置文件中读取
        eta=params['eta'],  # 优化的 eta
        model_name=model_name,
        params=params
    )

    # Run the simulation and return the performance metric
    trainer.run(num_sims=config['run_sims'])
    return trainer.get_performance_metric()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200)

    print(f"Best trial: {study.best_trial.params}")

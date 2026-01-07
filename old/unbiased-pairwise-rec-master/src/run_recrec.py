# """
# Codes for running the semi-synthetic experiments
# in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
# """
#
# import argparse
# import sys
# import yaml
# import warnings
#
# import numpy as np
# import pandas as pd
# import tensorflow as tf
#
# from trainer import Trainer
#
# parser = argparse.ArgumentParser()
#
# parser.add_argument('model_name', type=str, choices=['ReCRec-I','ReCRec-F','ReCRec-D'],default='RecRec-I')
# parser.add_argument('--preprocess_data', action='store_true')
# parser.add_argument('--dataset', type=str, choices=['yahoo','coat','kuai'],default='coat')
# parser.add_argument('--max_iters',type=int,default=8000)
# parser.add_argument('--batch_size',type=int,default=15)
# parser.add_argument('--lr',type=float,default=0.005)
# parser.add_argument('--lam',type=float,default=0.00001)
# parser.add_argument('--lamp',type=float,default=0.8)
# parser.add_argument('--threshold',type=int,default=4)
# parser.add_argument('--dim',type=int,default=200)
#
#
#
#
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# if __name__ == "__main__":
#     warnings.filterwarnings("ignore")
#     tf.get_logger().setLevel("ERROR")
#     args = parser.parse_args()
#     print(tf.test.is_gpu_available())
#     # hyper-parameters
#
#     batch_size = args.batch_size
#     max_iters = args.max_iters
#     lam = args.lam
#     lamp = args.lamp
#     eta = args.lr
#     model_name = args.model_name
#     dataset_name = args.dataset
#     threshold = args.threshold
#     dim = args.dim
#     trainer = Trainer(batch_size=batch_size, max_iters=max_iters,
#                       lam=lam,lamp=lamp, eta=eta, model_name=model_name,dataset_name = dataset_name, dim = dim)
#     trainer.run()
#
#     print(f'Finished Running {model_name}!')
#
#
import optuna
import argparse
import yaml
import warnings
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from trainer_recrec import Trainer

# Argument Parser Setup
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', '-m', type=str, choices=['ReCRec-I', 'ReCRec-F', 'ReCRec-D'], default='ReCRec-I')
parser.add_argument('--dataset','-d', type=str, choices=['yahoo', 'coat', 'kuai'], default='coat')
args = parser.parse_args()


# Function to create objective for Optuna

def objective(trial):
    # Suggest hyperparameters within a reasonable range based on defaults
    batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048, 4096, 8192])
    max_iters = trial.suggest_int('max_iters', 4000, 10000, step=1000)
    lam = trial.suggest_loguniform('lam', 1e-6, 1e-2)
    lamp = trial.suggest_uniform('lamp', 0.5, 1.0)
    eta = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    threshold = trial.suggest_int('threshold', 3, 5)
    dim = trial.suggest_int('dim', 100, 300, step=50)

    # Initialize Trainer with suggested hyperparameters
    trainer = Trainer(batch_size=batch_size, max_iters=max_iters,
                      lam=lam, lamp=lamp, eta=eta, model_name=args.model_name,
                      dataset_name=args.dataset, dim=dim)
    # Run the training process (for now using num_sims=1 for quicker evaluation)
    trainer.run()

    # Return performance metric for optimization
    return trainer.get_performance_metric()


# Function to run the model with the best parameters from YAML or Optuna
def run_with_best_params(best_params):
    trainer = Trainer(
        batch_size=best_params['batch_size'],
        max_iters=best_params['max_iters'],
        lam=best_params['lam'],
        lamp=best_params['lamp'],
        eta=best_params['lr'],
        model_name=args.model_name,
        dataset_name=args.dataset,
        dim=best_params['dim']
    )
    trainer.run()
    print(f"Model {args.model_name} trained with best hyperparameters.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")

    # Set environment variables to suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Run Optuna optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Output best trial
best_trial = study.best_trial
print(f"Best trial: {best_trial.params}")

# Save best parameters to a YAML file
with open(f'best_hyperparameters_{args.model_name}.yaml', 'w') as yaml_file:
    yaml.dump(best_trial.params, yaml_file, default_flow_style=False)

print(f"Best hyperparameters saved to 'best_hyperparameters_{args.model_name}.yaml'")

# Run the model with the best parameters found
run_with_best_params(best_trial.params)
study = optuna.create_study(direction="maxmize")
study.optimize(objective, n_trials=50)

# Output best trial
best_trial = study.best_trial
print(f"Best trial: {best_trial.params}")

# Save best parameters to a YAML file
with open(f'best_hyperparameters_{args.model_name}.yaml', 'w') as yaml_file:
    yaml.dump(best_trial.params, yaml_file, default_flow_style=False)

print(f"Best hyperparameters saved to 'best_hyperparameters_{args.model_name}.yaml'")

# Run the model with the best parameters found
run_with_best_params(best_trial.params)
# Load the best parameters from the saved YAML file
with open(f'best_hyperparameters_{args.model_name}.yaml', 'r') as yaml_file:
    best_params = yaml.safe_load(yaml_file)

print(f"Running model {args.model_name} with best hyperparameters from YAML file.")
run_with_best_params(best_params)

print(f'Finished Running {args.model_name}!')
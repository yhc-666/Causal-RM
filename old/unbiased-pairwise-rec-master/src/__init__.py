import optuna
import argparse
import yaml  # For saving and loading hyperparameters
import warnings
import tensorflow as tf
import os  # For checking if the YAML file exists
from trainer import Trainer

# Argument Parser Setup
parser = argparse.ArgumentParser()
possible_model_names = ['cjmf', 'BISER']
parser.add_argument('--model_name', '-m', type=str, required=True, choices=possible_model_names,
                    help="Specify the model name.")
parser.add_argument('--run_sims', '-r', type=int, default=10, required=True, help="Number of simulations to run.")
parser.add_argument('--data', '-d', type=str, required=True, choices=['coat', 'yahoo', 'kuai'],
                    help="Specify the dataset.")
args = parser.parse_args()


# Function to set fixed `wu` and `wi` values based on the dataset
def get_fixed_wu_wi(dataset):
    if dataset == 'coat':
        wu = 0.1
        wi = 0.5
    elif dataset == 'yahoo':
        wu = 0.9
        wi = 0.1
    elif dataset == 'kuai':
        wu = 0.05  # Example value for kuai, adjust as needed
        wi = 0.05  # Example value for kuai, adjust as needed
    else:
        raise ValueError("Invalid dataset. Please choose from 'coat', 'yahoo', or 'kuai'.")
    return wu, wi


# Get batch_size adjusted for CJMF's `C` value
def get_cjmf_batch_size(batch_size, C):
    return int(batch_size * (C - 1) / C)


# Function to get batch size based on the dataset
def get_batch_size(dataset):
    if dataset == 'coat':
        return 256
    elif dataset == 'yahoo':
        return 1024
    elif dataset == 'kuai':
        return 2048
    else:
        raise ValueError("Invalid dataset. Please choose from 'coat', 'yahoo', or 'kuai'.")


# CJMF optimization using Optuna
def objective_cjmf(trial):
    batch_size = get_batch_size(args.data)  # Base batch size per dataset
    wu, wi = get_fixed_wu_wi(args.data)  # Fixed `wu` and `wi` per dataset

    # Optimized parameters for CJMF
    params = {
        'hidden': trial.suggest_int('hidden', 100, 500),  # Number of hidden units
        'lam': trial.suggest_loguniform('lam', 1e-5, 1e-2),  # Regularization parameter
        'clip': trial.suggest_float('clip', 0.01, 0.5),  # Clipping value
        'batch_size': get_cjmf_batch_size(batch_size, trial.suggest_int('C', 2, 10)),  # Adjusted batch size for CJMF
        'max_iters': trial.suggest_int('max_iters', 500, 2000),  # Number of maximum iterations
        'eta': trial.suggest_loguniform('eta', 1e-4, 1e-1),  # Learning rate
        'unbiased_eval': True,  # Always true for unbiased evaluation
        'C': trial.suggest_int('C', 2, 10),  # C value for CJMF
        'alpha_cjmf': trial.suggest_float('alpha_cjmf', 100000, 300000),  # Alpha for CJMF
        'beta_cjmf': trial.suggest_float('beta_cjmf', 0.1, 0.9),  # Beta for CJMF
        'wu': wu,  # Fixed wu
        'wi': wi,  # Fixed wi
        'best_model_save': True,  # Always save the best model
        'random_state': 1  # Fixed random state for reproducibility
    }

    # Initialize the trainer
    trainer = Trainer(
        data=args.data,
        batch_size=batch_size,
        max_iters=params['max_iters'],
        eta=params['eta'],
        model_name=args.model_name,
        params=params
    )

    # Run simulations and return performance metric
    trainer.run(num_sims=1)
    return trainer.get_performance_metric()


# BISER optimization using Optuna (same approach, adapted to BISER parameters)
def objective_biser(trial):
    batch_size = get_batch_size(args.data)  # Base batch size per dataset

    wu, wi = get_fixed_wu_wi(args.data)  # Fixed `wu` and `wi` per dataset

    # Optimized parameters for BISER
    params = {
        'hidden': trial.suggest_int('hidden', 100, 500),  # Number of hidden units
        'lam': trial.suggest_loguniform('lam', 1e-5, 1e-2),  # Regularization parameter
        'clip': trial.suggest_float('clip', 0.01, 0.5),  # Clipping value
        'batch_size': batch_size,  # Fixed batch size per dataset
        'max_iters': trial.suggest_int('max_iters', 500, 2000),  # Number of maximum iterations
        'eta': trial.suggest_loguniform('eta', 1e-4, 1e-1),  # Learning rate
        'unbiased_eval': True,  # Always true for unbiased evaluation
        'wu': wu,  # Fixed wu
        'wi': wi,  # Fixed wi
        'alpha': trial.suggest_float('alpha', 0.2, 0.6),  # Alpha for BISER
        'neg_sample': trial.suggest_int('neg_sample', 5, 20),  # Negative sampling
        'best_model_save': True,  # Always save the best model
        'random_state': 1  # Fixed random state for reproducibility
    }

    # Initialize the trainer
    trainer = Trainer(
        data=args.data,
        batch_size=batch_size,
        max_iters=params['max_iters'],
        eta=params['eta'],
        model_name=args.model_name,
        params=params
    )

    # Run simulations and return performance metric
    trainer.run(num_sims=50)
    return trainer.get_performance_metric()


# Function to run the model with the best parameters from YAML or Optuna
def run_with_best_params(best_params):
    batch_size = get_batch_size(args.data)
    best_params['batch_size'] = batch_size
    wu, wi = get_fixed_wu_wi(args.data)
    best_params['wu'] = wu
    best_params['wi'] = wi

    trainer = Trainer(
        data=args.data,
        batch_size=batch_size,  # Use the best batch size or fallback to a default size
        max_iters=best_params['max_iters'],
        eta=best_params['eta'],  # Use the best learning rate
        model_name=args.model_name,
        params=best_params  # Use the best parameters found
    )

    trainer.run(num_sims=50)
    print(f"Model {args.model_name} trained with best hyperparameters.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")

    # Check if a YAML file with best hyperparameters already exists
    yaml_file_name = f'best_hyperparameters_{args.model_name}_{args.data}.yaml'

    if os.path.exists(yaml_file_name):
        # Load the best parameters from the existing YAML file
        print(f"Loading hyperparameters from existing YAML file: {yaml_file_name}")
        with open(yaml_file_name, 'r') as yaml_file:
            best_params = yaml.safe_load(yaml_file)

        print(f"Running model {args.model_name} with best hyperparameters from YAML file.")
        run_with_best_params(best_params)

    else:
        # Run Optuna optimization if no YAML file is found
        print(f"No existing YAML file found for {args.model_name} on {args.data}. Running Optuna optimization.")
        study = optuna.create_study(direction="maximize")

        if args.model_name == 'cjmf':
            study.optimize(objective_cjmf, n_trials=100)
        elif args.model_name == 'BISER':
            study.optimize(objective_biser, n_trials=100)

        # Output best trial
        best_trial = study.best_trial
        wu, wi = get_fixed_wu_wi(args.data)

        # Set the best_trial parameters
        best_trial.params['batch_size'] = get_batch_size(args.data)
        best_trial.params['wu'] = wu
        best_trial.params['wi'] = wi

        # Save best parameters to a YAML file with dataset and model name
        with open(yaml_file_name, 'w') as yaml_file:
            yaml.dump(best_trial.params, yaml_file, default_flow_style=False)

        print(f"Best hyperparameters saved to '{yaml_file_name}'")

        # Run the model with the best parameters found
        run_with_best_params(best_trial.params)
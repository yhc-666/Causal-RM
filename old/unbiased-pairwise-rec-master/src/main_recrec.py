"""
Codes for running the semi-synthetic experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""

import argparse
import sys
import yaml
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

# from data.preprocessor import preprocess_dataset
from trainer_recrec import Trainer

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', '-m',type=str, choices=['ReCRec-I','ReCRec-F','ReCRec-D'], default='RecRec-I')
parser.add_argument('--preprocess_data','-pd', action='store_true')
parser.add_argument('--dataset','-d', type=str, choices=['yahooR3','coat','kuaiRand'], default='coat')
parser.add_argument('--max_iters',type=int,default=8000)
parser.add_argument('--batch_size',type=int,default=25)
parser.add_argument('--lr',type=float,default=0.004764525409344301)
parser.add_argument('--lam',type=float,default=9.658748228257309e-05)
parser.add_argument('--lamp',type=float,default=0.8360265295741566)
parser.add_argument('--threshold',type=int,default=4)
parser.add_argument('--dim',type=int,default=150)




import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    tf.get_logger().setLevel("ERROR")
    args = parser.parse_args()
    print(tf.test.is_gpu_available())
    # hyper-parameters
    
    batch_size = args.batch_size
    max_iters = args.max_iters
    lam = args.lam
    lamp = args.lamp
    eta = args.lr
    model_name = args.model_name
    dataset_name = args.dataset
    threshold = args.threshold
    dim = args.dim


    trainer = Trainer(batch_size=batch_size, max_iters=max_iters,
                      lam=lam,lamp=lamp, eta=eta, model_name=model_name,dataset_name = dataset_name, dim = dim)
    trainer.run()

    print(f'Finished Running {model_name}!')
    
    
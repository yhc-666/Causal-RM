"""
Codes for preprocessing the real-world datasets
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback".
"""
import argparse
import warnings

from preprocess.preprocessor import preprocess_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', '-d', nargs='*', type=str, default=['kuai'], required=False, choices=['coat', 'yahoo', 'kuai'])
parser.add_argument('--sample_times', '-st', type=int, default=1, required=False)

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    args = parser.parse_args()
    sample_times = args.sample_times
    for data in args.datasets:
        preprocess_dataset(data=data, sample_times=sample_times)

        print('\n', '=' * 25, '\n')
        print(f'Finished Preprocessing {data}!')
        print('\n', '=' * 25, '\n')
"""
Codes for preprocessing the real-world datasets
in the paper "Unbiased Pairwise Learning from Biased Implicit Feedback".
"""
# import warnings
#
# from preprocess.preprocessor import preprocess_dataset
#
# # 直接在这里指定你要处理的数据集
# datasets = ['coat','yahoo']  # 可以根据需要修改成 ['coat'] 或 ['yahoo']
#
# if __name__ == "__main__":
#
#     warnings.filterwarnings("ignore")
#
#     for data in datasets:
#         preprocess_dataset(data=data)
#
#         print('\n', '=' * 25, '\n')
#         print(f'Finished Preprocessing {data}!')
#         print('\n', '=' * 25, '\n')

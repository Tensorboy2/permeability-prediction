'''
convert_to_npz.py

Module for converting folders to npy format.
'''
import numpy as np
import os
from tqdm import tqdm
path = os.path.dirname(__file__)

def to_npy(folder_path):
    file_list = sorted(f for f in os.listdir(folder_path) if f.startswith('k_') and  f.endswith('.npy'))
    all_objects = []
    for filename in tqdm(file_list):
        k = np.load(os.path.join(folder_path, filename))
        all_objects.append(k)
    all_k = np.stack(all_objects)
    np.save(os.path.join(path,f'{folder_path}_k.npy'),all_k)

if __name__ == '__main__':
    folder_path='train_and_val'
    to_npy(os.path.join(path,folder_path))
    folder_path='test'
    to_npy(os.path.join(path,folder_path))
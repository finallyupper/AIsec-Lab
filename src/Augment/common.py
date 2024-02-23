from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import numpy as np
import tqdm
from tqdm import tqdm 
import argparse

def url_index_mapping(urls_lst):
    """
    input:
    urls_lst : str list e.g. ['l1st.co.kr', '24h.com.vn', ...]

    returns:
    mapped : numpy array e.g. [['l1st.co.kr', '0'], ['24h.com.vn', '1'], '2', ...]
    """
    sorted_urls_lst = sorted(urls_lst)
    size = len(sorted_urls_lst)
    indices = [str(i) for i in range(size)]
    mapped= [list(item) for item in zip(sorted_urls_lst, indices)]

    return np.array(mapped)

def sample_traces(x, y, num_classes, N):
    """ 
    This function randomly samples N traces per website 
    """
    train_index = []
    
    for c in range(num_classes):
        idx = np.where(y == c)[0] 
        if len(idx) <= N:
            selected_idx = idx
        else:
            selected_idx = np.random.choice(idx, N, replace=False)
        train_index.extend(selected_idx)
        
    train_index = np.array(train_index)
    np.random.shuffle(train_index)
    
    x_train = x[train_index]
    y_train = y[train_index]
    
    not_selected_index = np.setdiff1d(np.arange(len(y)), train_index)
    not_selected_x = x[not_selected_index]
    not_selected_y = y[not_selected_index]
    
    return x_train, y_train, not_selected_x, not_selected_y

def fix_length(filepath, fixed = 5000, fix = 'yes'):
    """
    input:
    filepath: str
    fixed: int, default 5000 
    returns:
    times, sizes : float list, int list (length = 5000)
    """
    times, sizes =[], []

    with open(filepath, "r") as f:
        try:
            for x in f:
                x = x.split("\t")
                times.append(float(x[0]))
                sizes.append(int(x[1]))
            #print('Loaded all times and sizes information')

            if len(times) != len(sizes):
                raise Exception('times and sizes have different length')

            else:
                if fix == 'yes':
                    if len(times) < fixed: #padding
                        n_pad = fixed - len(times)
                        times += [0.0] * n_pad 
                        sizes += [0] * n_pad 
                    elif len(times) > fixed: #truncation
                        times = times[:fixed]
                        sizes = sizes[:fixed] 
                    # Done
                    if len(times) == fixed:
                        #print('Successfully fixed size as', fixed)
                        return times, sizes
                    else:
                        raise Exception('Got some error while padding(or truncation)')
                else:
                    return times, sizes
        except KeyboardInterrupt:
            sys.exit(-1)
        except:
            return

def only_direction(seq:list)->list:
    """
    input : 
    seq : int list e.g. [+105, -105, +660, -660, ...]

    returns:
    directions : int list e.g. [1, -1, 1, -1, ...]
    """
    directions = []
    for s in seq:
        if s > 0:
            directions.append(1)
        else:
            directions.append(-1)
    return directions 


def get_npz(base_path, urls_lst, fix = 'yes', onlydir = False):
    """ 
    e.g.
    base_path = '/scratch/DA/dataset/tcp4/attack/sup'
    urls_lst = ['11st.co.kr', 'abc.com', ... ]
    fix = fix length if 'yes', else 'no'
    onlydir = only save direction if True, else False
    """
    X = []
    y = []
    for url, idx in tqdm(urls_lst, desc = 'Processing urls'):
        url_dir = os.path.join(base_path, url)
        traces = os.listdir(url_dir) 

        for trace in traces:
            filepath = os.path.join(url_dir, trace) # '../11st.co.kr/25'
            times, sizes = fix_length(filepath, fix = fix)

            if onlydir:
                sizes = only_direction(sizes)

            element =[[list(item) for item in zip(times, sizes)]]
            X.extend(element)
            y.append(int(idx))

    return np.array(X), np.array(y)


def get_split(datapath, n_samples, type = '', onlydir = False):
    """ 
    datapath = path of attack-sup or attack-inf
    type = 'sup' or 'inf'
    n_samples = 30, 50
    """
    urls = os.listdir(datapath) 
    urls_lst = url_index_mapping(urls)

    x, y = get_npz(datapath, urls_lst, fix = 'yes', onlydir = onlydir)
    num_classes = len(np.unique(y))

    print(f'{type}_x shape: {x.shape}, {type}_y shape: {y.shape}')
    
    selected_x, selected_y, not_selected_x, not_selected_y = sample_traces(x, y, num_classes, n_samples)

    return selected_x, selected_y, not_selected_x, not_selected_y
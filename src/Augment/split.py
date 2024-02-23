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

from winter_internship.data_preprocessing.common.common import get_split

def parse_args():
    """
    Parse command line arguments

    Accepted arguments:
      (r)oot -- Directory that stores datasets
      (o)utput -- Directory that store the processed datasets
      cfg -- type of configurations (1 or 2)
      onlydir -- if True, save only direction. Else, also saves size information
    Returns
    -------
    Namespace
        Argument namespace object

    """
    parser = argparse.ArgumentParser("Pretrain netCLR with customed features")

    # Required Arguments
    parser.add_argument("-r", "--root",
                        required=True,
                        type=str,
                        help="Directory that stores datasets") #help = Help message for an argument

    parser.add_argument("-o", "--output",
                        required=True,
                        type=str,
                        help="Directory that store the processed datasets")
    parser.add_argument("--cfg",
                        required = True,
                        type=int,
                        default=-1,
                        help="type of configurations (1 or 2)")
    # Optional Arguments
    parser.add_argument("--onlydir",
                    type=bool,
                    default=False,
                    help="if True, save only direction. Else, also saves size information")
    return parser.parse_args()




def main(root, output, cfg, onlydir = False):
    sup_dir = os.path.join(root, 'sup')
    inf_dir = os.path.join(root, 'inf')
    urls = os.listdir(sup_dir)  # or inf_dir

    print(f'data path : {root}')
    print(f'output : {output}')
    print(f'cfg {cfg} selected')

    if cfg == 1:
        n_samples = 50

        sup_selected_x, sup_selected_y, sup_not_selected_x, sup_not_selected_y = get_split(sup_dir, n_samples, type = 'sup', onlydir = onlydir)
        inf_selected_x, inf_selected_y, inf_not_selected_x, inf_not_selected_y = get_split(inf_dir, n_samples, type = 'inf', onlydir = onlydir)
        
        print(f'x train shape: {sup_not_selected_x.shape} y train shape: {sup_not_selected_y.shape}')
        print(f'x_test_fast shape: {sup_selected_x.shape} y_test_fast shape: {sup_selected_y.shape}')
        print(f'x_test_slow shape: {inf_selected_x.shape} y_test_slow: {inf_selected_y.shape}')

        np.savez(output,\
                x_train = sup_not_selected_x, y_train = sup_not_selected_y,\
                x_test_fast = sup_selected_x, y_test_fast = sup_selected_y, \
                x_test_slow = inf_selected_x, y_test_slow = inf_selected_y, \
                url_index_mapping = urls)  
        
        print(f'Successfully saved {output.split('/')[-1]}') # ex. cfg1_fine_tuning_data.npz

    elif cfg == 2:
        n_samples= 30

        sup_selected_x, sup_selected_y, sup_not_selected_x, sup_not_selected_y = get_split(sup_dir, n_samples, type = 'sup', onlydir = onlydir)
        inf_selected_x, inf_selected_y, inf_not_selected_x, inf_not_selected_y = get_split(inf_dir, n_samples, type = 'inf', onlydir = onlydir)

        print(f'x train shape: {inf_not_selected_x.shape} y train shape: {inf_not_selected_y.shape}')
        print(f'x_test_fast shape: {sup_selected_x.shape} y_test_fast shape: {sup_selected_y.shape}')
        print(f'x_test_slow shape: {inf_selected_x.shape} y_test_slow: {inf_selected_y.shape}')

        np.savez(output,\
                x_train = inf_not_selected_x, y_train = inf_not_selected_y,\
                x_test_fast = sup_selected_x, y_test_fast = sup_selected_y, \
                x_test_slow = inf_selected_x, y_test_slow = inf_selected_y, \
                url_index_mapping = urls)  

    else:
        raise Exception('Unavailable configurations! Only 1 or 2 exists..')
    


if __name__ == "__main__":
    try:
        args = parse_args()
        main(
            root = args.root, # root = '/scratch/DA/dataset/tcp5'
            output = args.output, # output = '/scratch/DA/dataset/tcp4/cfg1_fine_tuning_data.npz'
            cfg = args.cfg,
            onlydir = args.onltdir 
        )
    except KeyboardInterrupt:
        sys.exit(-1)        
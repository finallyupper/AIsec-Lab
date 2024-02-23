import os
import sys
import numpy as np
import tqdm
from tqdm import tqdm 

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


def no_fix_length(filepath):
    """
    input:
    filepath: str
    returns:ndarray, ndarray
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
            return np.array(times), np.array(sizes)
        except KeyboardInterrupt:
            sys.exit(-1)
        except:
            return


# 파일을 로딩해 5000길이에 맞춰 시간, 패킷 정보 split
def fix_length(filepath, fixed = 5000):
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
        except KeyboardInterrupt:
            sys.exit(-1)
        except:
            return

def only_direction(seq):
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

def find_maxlen(root:str)->int:
    """ 
    e.g.
    root =  '/scratch/DA/dataset/tcp5_filtered/pretrain/sup'
    """
    urls = os.listdir(root) # urls..
    MAX = 0
    for url in urls:
        urlpath = os.path.join(root, url)
        files = os.listdir(urlpath)
        for file in files:
            filepath = os.path.join(urlpath, file)
            with open (filepath, "r") as f:
                l = len(f.readlines())
                if l > MAX:
                    MAX = l 
    return MAX 

def main():
    root = '/scratch/DA/dataset/tcp5/pretrain' ###
    sup_dir = os.path.join(root, 'sup')
    inf_dir = os.path.join(root, 'inf')

    sup_max_len = find_maxlen(sup_dir)
    inf_max_len = find_maxlen(inf_dir)
    

    # 1. pretrain-sup
    urls = os.listdir(sup_dir) # = inf_dir
    urls_lst_1 = url_index_mapping(urls) # url list 만들기
 
    attack_sup_x = []
    attack_sup_y = []
    for url, idx in tqdm(urls_lst_1, desc = 'Processing urls'):
        # ex. url='aa.com', idx = '3'
        url_dir = os.path.join(sup_dir, url)
        traces = os.listdir(url_dir) 
        for trace in traces:
            filepath = os.path.join(url_dir, trace) # '../11st.co.kr/25'
            times, sizes = no_fix_length(filepath)

            times = np.pad(times, (0, sup_max_len - len(times)), mode='constant')
            sizes = np.pad(sizes, (0, sup_max_len - len(sizes)), mode='constant')
            
            x = np.column_stack((times, sizes)) # (a, 2)
            #x =[[list(item) for item in zip(times, sizes)]]
            attack_sup_x.append(x) #extend(x)
            attack_sup_y.append(int(idx))

    attack_sup_x = np.array(attack_sup_x)
    attack_sup_y = np.array(attack_sup_y)
    print(f'pt_sup_x shape:{attack_sup_x.shape}, pt_sup_y shape: {attack_sup_y.shape}')
    

    # 2. pretrain-inf
    urls = os.listdir(inf_dir) # = inf_dir
    urls_lst_2 = url_index_mapping(urls) # url list 만들기

    attack_inf_x = []
    attack_inf_y = []

    for url, idx in tqdm(urls_lst_2, desc = 'Processing urls'):
        # ex. url='aa.com', idx = '3'
        url_dir = os.path.join(inf_dir, url)
        traces = os.listdir(url_dir) 
        for trace in traces:
            filepath = os.path.join(url_dir, trace) # '../11st.co.kr/25'
            times, sizes = no_fix_length(filepath)

            times = np.pad(times, (0, inf_max_len - len(times)), mode='constant')
            sizes = np.pad(sizes, (0, inf_max_len - len(sizes)), mode='constant')
            
            x = np.column_stack((times, sizes)) # (a, 2)
            attack_inf_x.append(x) #extend(x)
            attack_inf_y.append(int(idx))


    attack_inf_x = np.array(attack_inf_x)
    attack_inf_y = np.array(attack_inf_y)
    num_classes = len(np.unique(attack_inf_y))
    print(f'pt_inf_x shape:{attack_inf_x.shape}, pt_inf_y shape: {attack_inf_y.shape}')

    # direction, size 둘다 살린 버전
    basedir = '/scratch/DA/dataset/tcp5'
    savedir1 = os.path.join(basedir, 'cfg1_pretrain_long.npz')
    savedir2 = os.path.join(basedir, 'cfg2_pretrain_long.npz')
    np.savez(savedir1, \
             X_pretrain_superior = attack_sup_x, y_pretrain_superior = attack_sup_y, \
                url_index_mapping_superior = urls_lst_1)
    np.savez(savedir2, \
             X_pretrain_inferior = attack_inf_x, y_pretrain_inferior = attack_inf_y, \
                url_index_mapping_inferior = urls_lst_2)
    print('Successfully saved cfg1 & cfg2 pretrain npzs..!')

main()
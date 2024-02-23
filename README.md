# AIsec-Lab
## Introduction
Purpose of this code is to preprocess the given dataset to fit the given paper. The input data is already splitted into superior/inferior dataset that includes instances (no ack, empty files)(Total 200 sites with 50000 instances).  It supports to preprocess dataset into two configurations. Details can be found in the [paper](https://people.cs.umass.edu/~amir/papers/CCS23-SSL-Web-Fingerprint.pdf) [1]
## Usage Examples
```
DATA_PATH = "/scratch/DA/dataset/tcp5"  
OUTPUT_PATH = "/scratch/DA/dataset/tcp5cfg1_fine_tuning_data.npz"
```
Data preprocessing 
```
python split.py -r "${DATA_PATH}" -o "${OUTPUT_PATH}" --cfg 1 --onlydir False

python split.py -r "${DATA_PATH}" -o "${OUTPUT_PATH}" --cfg 2 --onlydir False

```
## Directory Structure
```
src
    ├─Augment
    │      common.py
    │      split.py
    │
    └─others
        ├─tcp5
        │      pre_training_data.py
        │
        └─tcp5_filtered
                pt2.py
```
## References
[1] https://people.cs.umass.edu/~amir/papers/CCS23-SSL-Web-Fingerprint.pdf  
[2] https://github.com/notem/reWeFDE/tree/master
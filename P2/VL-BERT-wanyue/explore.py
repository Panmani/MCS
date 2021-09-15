import sys
sys.path.append('../ERNIE/ernie-vil')

import re
import json
import nltk
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

random_name_path = '../ERNIE/ernie-vil/data/vcr/unisex_names_table.csv'
val_gen_path = 'data/vc-temporal-captions/val_sample_1_num_5_top_k_0_top_p_0.9.json'
trn_gen_path = 'data/vc-temporal-captions/train_sample_1_num_5_top_k_0_top_p_0.9.json'
vc_val_path = '../ERNIE/ernie-vil/data/visualcomet/val_annots.json'


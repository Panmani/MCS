import pickle
from extract_features import *

if __name__ == '__main__':

    feat_dir = "{}2017_feat".format(split)
    LOAD_DATASET_DIR = os.path.join(OUT_DIR, feat_dir)


    pkl_files = [pkl_file for pkl_file in os.listdir(LOAD_DATASET_DIR)
                        if os.path.isfile(os.path.join(LOAD_DATASET_DIR, pkl_file))]

    for pkl_file in pkl_files:
        features_dict = pickle.load( open( os.path.join(LOAD_DATASET_DIR, pkl_file), "rb" ) )
        print(features_dict)

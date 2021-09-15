import pickle
import lmdb
import base64

from tqdm import tqdm
from image_feature_reader import ImageFeaturesH5Reader, load_obj_tsv

coco_feat_folder = '/home/haoxuan/code/lxmert/data/mscoco_imgfeat'

coco_feat_trn = load_obj_tsv(coco_feat_folder + '/train2014_obj36.tsv')
coco_feat_val = load_obj_tsv(coco_feat_folder + '/val2014_obj36.tsv')

coco_feat = coco_feat_trn + coco_feat_val





env = lmdb.open("data/coco/COCO.lmdb", map_size=1099511627776)
txn = env.begin(write=True)

keys = []

for entry in tqdm(coco_feat):
    item = {
        'image_id': entry['img_id'].split('_')[-1],
        'image_h': entry['img_h'],
        'image_w': entry['img_w'],
        'objects_id': base64.b64encode(entry['objects_id']),
        'objects_conf': base64.b64encode(entry['objects_conf']),
        'num_boxes': entry['num_boxes'],
        'boxes': base64.b64encode(entry['boxes'].reshape(-1)),
        'features': base64.b64encode(entry['features'].reshape(-1))}
    item_pk = pickle.dumps(item)
    txn.put(key = item['image_id'].encode(), value=item_pk)
    keys.append(item['image_id'])

keys_pk = pickle.dumps(keys)
txn.put(key = 'keys'.encode(), value=keys_pk)

txn.commit()
env.close()
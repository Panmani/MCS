import numpy as np
import copy
import pickle
import lmdb # install lmdb by "pip install lmdb"
import base64
import sys
import csv
import time




csv.field_size_limit(sys.maxsize)
class ImageFeaturesH5Reader(object):
    """
    Reader class
    """
    def __init__(self, features_path, dataset='visualcomet'):
        self.dataset = dataset
        self.features_path = features_path
        self.env = lmdb.open(self.features_path, max_readers=1, readonly=True,
                            lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self._image_ids = pickle.loads(txn.get('keys'.encode()))

        self.features = [None] * len(self._image_ids)
        self.num_boxes = [None] * len(self._image_ids)
        self.boxes = [None] * len(self._image_ids)
        self.boxes_ori = [None] * len(self._image_ids)

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, image_id):
        image_id = str(image_id).encode()

        # Read chunk from file everytime if not loaded in memory.
        with self.env.begin(write=False) as txn:
            item = pickle.loads(txn.get(image_id))
            image_id = item['image_id']
            image_h = int(item['image_h'])
            image_w = int(item['image_w'])
            num_boxes = int(item['num_boxes'])

            features = np.frombuffer(base64.b64decode(item["features"]), dtype=np.float32).reshape(num_boxes, 2048)
            boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(num_boxes, 4)

            if self.dataset == 'visualcomet':
                class_prob = np.frombuffer(base64.b64decode(item["cls_prob"]), dtype=np.float32).reshape(num_boxes, -1)#[:,1:]
            elif self.dataset == 'coco':
                class_id = np.frombuffer(base64.b64decode(item["objects_id"]), dtype=np.int64)
                class_conf = np.frombuffer(base64.b64decode(item["objects_conf"]), dtype=np.float32)

            g_feat = np.sum(features, axis=0) / num_boxes
            num_boxes = num_boxes + 1
            features = np.concatenate([np.expand_dims(g_feat, axis=0), features], axis=0)

            # image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
            # image_location[:, :4] = boxes
            # image_location[:, 4] = (image_location[:, 3] - image_location[:, 1]) *   \
            #         (image_location[:, 2] - image_location[:, 0]) / (float(image_w) * float(image_h))
            #
            # image_location_ori = copy.deepcopy(image_location)
            # image_location[:, 0] = image_location[:, 0] / float(image_w)
            # image_location[:, 1] = image_location[:, 1] / float(image_h)
            # image_location[:, 2] = image_location[:, 2] / float(image_w)
            # image_location[:, 3] = image_location[:, 3] / float(image_h)
            #
            # g_location = np.array([0, 0, 1, 1, 1])
            # image_location = np.concatenate([np.expand_dims(g_location, axis=0), image_location], axis=0)
            #
            # g_location_ori = np.array([0, 0, image_w, image_h, image_w * image_h])
            # image_location_ori = np.concatenate([np.expand_dims(g_location_ori, axis=0), image_location_ori], axis=0)

        if self.dataset == 'visualcomet':
            data_json = {"image_h": image_h,
                         "image_w": image_w,
                         "class_prob": class_prob,
                         "features": features,
                         "boxes": boxes,
                         "num_boxes": num_boxes}
        elif self.dataset == 'coco':
            data_json = {"image_h": image_h,
                         "image_w": image_w,
                         "class_id": class_id,
                         "class_conf": class_conf,
                         "features": features,
                         "boxes": boxes,
                         "num_boxes": num_boxes}

        return data_json




FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.
    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])

            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes,), np.int64),
                ('objects_conf', (boxes,), np.float32),
                ('attrs_id', (boxes,), np.int64),
                ('attrs_conf', (boxes,), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data
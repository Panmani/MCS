import os
import io

import detectron2

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# doit
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image

# doit_for_image
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, fast_rcnn_inference_single_image
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances

# import some common libraries
import numpy as np
import cv2
import torch
from torch import nn

# Show the image in ipynb
from IPython.display import clear_output, Image, display
import PIL.Image

from tqdm import tqdm
import pickle
import json

# from coco2vcr import COCO_PATH, OUT_DIR, split

# DEVICE = 'cpu'
DEVICE = 'cuda'
NUM_OBJECTS = 36
img_type = '.jpg'

# IMAGE_DIR = "/home/jamesp/data/vcr/vcr1images"
IMAGE_DIR = "../../VLBERT/data/vcr/vcr1images/"
OUTPUT_DIR = "../../tmp_comet"


# def showarray(a, fmt='jpeg'):
#     a = np.uint8(np.clip(a, 0, 255))
#     f = io.BytesIO()
#     PIL.Image.fromarray(a).save(f, fmt)
#     display(Image(data=f.getvalue()))


# # Load VG Classes
# data_path = 'data/genome/1600-400-20'
#
# vg_classes = []
# with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
#     for object in f.readlines():
#         vg_classes.append(object.split(',')[0].lower().strip())
#
# vg_attrs = []
# with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
#     for object in f.readlines():
#         vg_attrs.append(object.split(',')[0].lower().strip())
#
#
# MetadataCatalog.get("vg").thing_classes = vg_classes
# MetadataCatalog.get("vg").attr_classes = vg_attrs


def doit_for_objects(predictor, raw_image):
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        # print("Original image size: ", (raw_height, raw_width))

        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        # print("Transformed image size: ", image.shape[:2])
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)

        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]
        # print('Proposal Boxes size:', proposal.proposal_boxes.tensor.shape)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        # print('Pooled features size:', feature_pooled.shape)

        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]

        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)

        # Note: BUTD uses raw RoI predictions,
        #       we use the predicted boxes instead.
        # boxes = proposal_boxes[0].tensor

        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:],
                score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
            )
            if len(ids) == NUM_OBJECTS:
                break

        instances = detector_postprocess(instances, raw_height, raw_width)
        roi_features = feature_pooled[ids].detach()
        max_attr_prob = max_attr_prob[ids].detach()
        max_attr_label = max_attr_label[ids].detach()
        instances.attr_scores = max_attr_prob
        instances.attr_classes = max_attr_label

        # print(instances)

        return instances, roi_features


def doit_for_image(predictor, raw_image):
        # Process Boxes
    raw_boxes = np.array([[0, 0, raw_image.shape[1], raw_image.shape[0]]]) # [left, top, right, bottom]
    if cfg.MODEL.DEVICE == "cpu":
        raw_boxes = Boxes(torch.from_numpy(raw_boxes))
    elif cfg.MODEL.DEVICE == "cuda":
        raw_boxes = Boxes(torch.from_numpy(raw_boxes).cuda())
    else:
        raise ValueError

    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        # print("Original image size: ", (raw_height, raw_width))

        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        # print("Transformed image size: ", image.shape[:2])

        # Scale the box
        new_height, new_width = image.shape[:2]
        scale_x = 1. * new_width / raw_width
        scale_y = 1. * new_height / raw_height
        #print(scale_x, scale_y)
        boxes = raw_boxes.clone()
        boxes.scale(scale_x=scale_x, scale_y=scale_y)

        # ----
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [boxes]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        # print('Pooled features size:', feature_pooled.shape)

        # Predict classes        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled) and boxes for each proposal.
        pred_class_logits, pred_attr_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        pred_class_prob = nn.functional.softmax(pred_class_logits, -1)
        pred_scores, pred_classes = pred_class_prob[..., :-1].max(-1)

        attr_prob = pred_attr_logits[..., :-1].softmax(-1)
        max_attr_prob, max_attr_label = attr_prob.max(-1)

        # Detectron2 Formatting (for visualization only)
        roi_features = feature_pooled
        instances = Instances(
            image_size=(raw_height, raw_width),
            pred_boxes=raw_boxes,
            scores=pred_scores,
            pred_classes=pred_classes,
            attr_scores = max_attr_prob,
            attr_classes = max_attr_label
        )

        return instances, roi_features

def get_features_dict(predictor, img_path, object_vocab):
    img = cv2.imread(img_path)

    image_instances, image_features = doit_for_image(predictor, img)
    object_instances, object_features = doit_for_objects(predictor, img)

    features_dict = {
        'img_path'         : img_path,
        'image_features'   : np.array(image_features.tolist()),
        'object_features'  : np.array(object_features.tolist()),
        # 'image_instances'  : image_instances,
        # 'object_instances' : object_instances,
    }

    object_height = object_instances._image_size[0]
    object_width = object_instances._image_size[1]
    object_boxes = object_instances.get('pred_boxes').tensor.tolist()
    object_labels = object_instances.get('pred_classes').tolist()

    object_names = []
    for label in object_labels:
        object_names.append(object_vocab[label])

    metadata = {
        'height' : object_height,
        'width'  : object_width,
        'boxes'  : object_boxes,
        'labels' : object_labels,
        'names'  : object_names,
    }

    # image_height = image_instances.image_height
    # image_width = image_instances.image_width
    # image_boxes = image_instances.get('pred_boxes').tensor.tolist()
    # image_labels = image_instances.get('pred_classes').tensor.tolist()

    return features_dict, metadata


if __name__ == '__main__':

    # =================== predictor ===================
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file("../configs/VG-Detection/faster_rcnn_R_101_C4_attr_caffemaxpool.yaml")
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.DEVICE = DEVICE
    # VG Weight
    cfg.MODEL.WEIGHTS = "../faster_rcnn_from_caffe_attr.pkl"
    predictor = DefaultPredictor(cfg)
    # =================================================

    # img_dir = "{}2017".format(split)
    # ann_filename = '{}2017_annots.json'.format(split)
    # metadata_dir = "{}2017_meta".format(split)
    # feat_dir = "{}2017_feat".format(split)
    #
    # dataset_path = os.path.join(COCO_PATH, img_dir)
    # if not os.path.isdir(dataset_path):
    #     print("Cannot find dataset images at {}".format(dataset_path))
    #     exit()
    #
    # if not os.path.isdir(OUT_DIR):
    #     os.mkdir(OUT_DIR)
    # if not os.path.isdir(os.path.join(OUT_DIR, feat_dir)):
    #     os.mkdir(os.path.join(OUT_DIR, feat_dir))
    # if not os.path.isdir(os.path.join(OUT_DIR, metadata_dir)):
    #     os.mkdir(os.path.join(OUT_DIR, metadata_dir))

    # dataset_imgs = [f for f in os.listdir(dataset_path)
    #                     if os.path.isfile(os.path.join(dataset_path, f))]

    # with open(os.path.join(COCO_PATH, 'annotations', 'captions_{}.json'.format(img_dir))) as json_file:
    #     data = json.load(json_file)

    object_vocab = []
    with open("../demo/data/genome/1600-400-20/objects_vocab.txt", 'r') as obj_vocab_file:
        for line in obj_vocab_file:
            object_vocab.append(line.strip())

    movie_dirs = sorted(os.listdir(IMAGE_DIR))
    print(len(movie_dirs))

    # dataset_imgs = []
    # for img_dict in data['images']:
    #     dataset_imgs.append(img_dict['file_name'])

    for movie in tqdm(movie_dirs):
        img_ids = list(set([id[:id.rfind('.')] for id in os.listdir(os.path.join(IMAGE_DIR,movie))]))
        for id in sorted(img_ids):
            img_path = os.path.join(IMAGE_DIR,movie,id+'.jpg')
            metadata = json.load(open(os.path.join(IMAGE_DIR,movie,id+'.json')))
            boxes = np.array(metadata['boxes'])[:,:4]
            h = metadata['height']
            w = metadata['width']
            boxes = np.row_stack((np.array([0,0,w,h]),boxes))
            features_dict, metadata = get_features_dict(predictor, img_path, object_vocab)
            # print(feature_dict, metadata)
            # obj_rep = doit(im, boxes).to("cpu").numpy()

            # features = {'image_features' : obj_rep[0],
            #             'object_features' : obj_rep[1:]}
            pickle.dump(features_dict, open(os.path.join(OUTPUT_DIR, id+'.pkl'),'wb'))

            metadata_fn = os.path.join(OUTPUT_DIR, id + ".json")
            with open(metadata_fn, "w") as metadata_file:
                json.dump(metadata, metadata_file)

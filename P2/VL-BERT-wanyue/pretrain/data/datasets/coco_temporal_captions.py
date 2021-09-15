import os
import sys
import re
import json
import copy
import nltk
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import random
import time
import jsonlines
from PIL import Image
import base64
import logging

import torch
from torch.utils.data import Dataset
from image_feature_reader import ImageFeaturesH5Reader, load_obj_tsv

from external.pytorch_pretrained_bert import BertTokenizer

from common.utils.zipreader import ZipReader
from common.utils.create_logger import makedirsExist

def split_temporal_event(caption):
    # token: 0-intent, 1-before, 2-after
    patts = ['Because (.*) wanted to (.*), (.*)',
             'Before (.*), (.*) needed to (.*)',
             'After (.*), (.*) will most likely (.*)']
    temporal, event = '', ''
    for i, patt in enumerate(patts):
        regex = re.compile(patt)
        result = regex.match(caption)
        if result:
            if i == 0:
                event = result.group(3).rstrip()
                temporal = ' '.join([result.group(1).rstrip(), result.group(2).rstrip()])
            else:
                event = result.group(1).rstrip()
                temporal = ' '.join([result.group(2).rstrip(), result.group(3).rstrip()])
            break
    if event[-1] != '.':
        event += '.'
    return i, event, temporal

class COCOTemporalCaptionsDataset(Dataset):
    def __init__(self, ann_file, image_set, root_path, data_path, seq_len=64,
                 with_precomputed_visual_feat=True, mask_raw_pixels=True,
                 with_rel_task=False, with_mlm_task=True, with_mvrc_task=True,
                 transform=None, test_mode=False,
                 zip_mode=False, cache_mode=False, cache_db=False, ignore_db_cache=True,
                 tokenizer=None, pretrained_model_name=None,
                 add_image_as_a_box=False,
                 aspect_grouping=False, **kwargs):

        """
        COCO Temporal Captions Dataset

        :param ann_file: annotation jsonl file
        :param root_path: root path to data folder
        :param data_path: path to vc-temporal caption
        :param image_path: path to vcr/vcr1images
        :param transform: transform
        :param test_mode: test mode means no labels available
        :param zip_mode: reading images and metadata in zip archive
        :param cache_mode: cache whole dataset to RAM first, then __getitem__ read them from RAM
        :param ignore_db_cache: ignore previous cached database, reload it from annotation file
        :param tokenizer: default is BertTokenizer from pytorch_pretrained_bert
        :param add_image_as_a_box: add whole image as a box
        :param aspect_grouping: whether to group images via their aspect
        :param kwargs:
        """
        super(COCOTemporalCaptionsDataset, self).__init__()

        assert not cache_mode, 'currently not support cache mode!'
        assert not test_mode
        assert not aspect_grouping
        assert with_precomputed_visual_feat

        self.seq_len = seq_len
        self.with_rel_task = with_rel_task
        self.with_mlm_task = with_mlm_task
        self.with_mvrc_task = with_mvrc_task
        self.data_path = data_path
        self.image_root_path = os.path.join(root_path, image_set)

        self.ann_file = os.path.join(root_path, data_path, ann_file)
        self.with_precomputed_visual_feat = with_precomputed_visual_feat
        self.mask_raw_pixels = mask_raw_pixels
        self.add_image_as_a_box = add_image_as_a_box

        self.transform = transform
        self.test_mode = test_mode
        self.zip_mode = zip_mode
        self.cache_mode = cache_mode
        self.cache_db = cache_db
        self.ignore_db_cache = ignore_db_cache
        self.aspect_grouping = aspect_grouping
        self.cache_dir = os.path.join(root_path, 'cache')

        if not os.path.exists(self.cache_dir):
            makedirsExist(self.cache_dir)

        self.tokenizer = tokenizer if tokenizer is not None \
            else BertTokenizer.from_pretrained(
            'bert-base-uncased' if pretrained_model_name is None else pretrained_model_name,
            cache_dir=self.cache_dir)

        self.annot_data = self._load_json(self.ann_file)
        if 'train' in self.ann_file:
            self.mode='train'
        elif 'val' in self.ann_file:
            self.mode='val'
        else:
            assert False

        self.image_feature_reader = ImageFeaturesH5Reader('data/coco/COCO.lmdb', dataset='coco')

        print('Caption sample:', np.random.choice(self.annot_data[0]['captions']))
        print('Pretrain task:', with_rel_task, with_mlm_task, with_mvrc_task)

    @property
    def data_names(self):
        return ['image', 'boxes', 'im_info', 'text',
                'relationship_label', 'mlm_labels', 'mvrc_ops', 'mvrc_labels']

    def __getitem__(self, index):
        annot_i = self.annot_data[index]
        image_id = annot_i['img_fn'].split('/')[1][:-4]

        image_features = self.image_feature_reader[image_id]
        boxes = image_features['boxes']

        boxes_max_conf = image_features['class_conf']
        boxes_object = image_features['class_id']
        boxes_cls_scores = np.zeros((len(boxes_max_conf), 1601))
        for i in range(boxes_cls_scores.shape[0]):
            boxes_cls_scores[i] = (1 - boxes_max_conf[i]) / 1600
            boxes_cls_scores[i][boxes_object[i]] = boxes_max_conf[i]

        inds = np.argsort(boxes_max_conf)[::-1]

        boxes = boxes[inds]
        boxes_cls_scores = boxes_cls_scores[inds]
        boxes = torch.as_tensor(boxes)

        if self.with_precomputed_visual_feat:
            image = None
            w0, h0 = image_features['image_w'], image_features['image_h']
            boxes_features = image_features['features']
            boxes_features = boxes_features[inds]
            boxes_features = torch.as_tensor(boxes_features)

        if self.add_image_as_a_box:
            image_box = torch.as_tensor([[0.0, 0.0, w0 - 1.0, h0 - 1.0]])
            boxes = torch.cat((image_box, boxes), dim=0)
            if self.with_precomputed_visual_feat:
                image_box_feat = boxes_features.mean(dim=0, keepdim=True)
                boxes_features = torch.cat((image_box_feat, boxes_features), dim=0)

        im_info = torch.tensor([w0, h0, 1.0, 1.0, index])
        if self.transform is not None:
            image, boxes, _, im_info = self.transform(image, boxes, None, im_info)

        if image is None and (not self.with_precomputed_visual_feat):
            w = int(im_info[0].item())
            h = int(im_info[1].item())
            image = im_info.new_zeros((3, h, w), dtype=torch.float)

        # clamp boxes
        w = im_info[0].item()
        h = im_info[1].item()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=0, max=w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=0, max=h - 1)

        # Task #1: Caption-Image Relationship Prediction
        _p = random.random()
        if _p < 0.5 or (not self.with_rel_task):
            relationship_label = 1
            caption = annot_i['captions'][0]
            #caption = np.random.choice(annot_i['captions'])
        else:
            relationship_label = 0
            rand_index = random.randrange(0, len(self.annot_data))
            while rand_index == index:
                rand_index = random.randrange(0, len(self.annot_data))
            caption = np.random.choice(self.annot_data[rand_index]['captions'])

        if self.with_rel_task:
            relationship_label, event, temporal = split_temporal_event(caption)
            caption = event + ' [SEP] ' + temporal

        # original_caption = copy.deepcopy(caption)
        # caption = re.sub('^After|^Before|^Because', '[MASK]', caption)
        # caption = re.sub('wanted to|needed to', '[MASK] [MASK]', caption)
        # caption = re.sub('will most likely', '[MASK] [MASK] [MASK]', caption)
        # print('1>>', caption)

        # Task #2: Masked Language Modeling
        if self.with_mlm_task:
            caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)
            caption_tokens, mlm_labels = self.random_word_wwm(caption_tokens)
        else:
            caption_tokens = self.tokenizer.tokenize(caption)
            mlm_labels = [-1] * len(caption_tokens)
        text_tokens = ['[CLS]'] + caption_tokens + ['[SEP]']
        mlm_labels = [-1] + mlm_labels + [-1]

        # print(caption)
        # print(text_tokens)
        # print(mlm_labels)
        # return text_tokens, mlm_labels

        # Task #3: Masked Visual Region Classification
        if self.with_mvrc_task:
            if self.add_image_as_a_box:
                mvrc_ops, mvrc_labels = self.random_mask_region(boxes_cls_scores)
                mvrc_ops = [0] + mvrc_ops
                mvrc_labels = [np.zeros_like(boxes_cls_scores[0])] + mvrc_labels
                num_real_boxes = boxes.shape[0] - 1
                num_masked_boxes = 0
                if self.with_precomputed_visual_feat:
                    boxes_features[0] *= num_real_boxes
                    for mvrc_op, box_feat in zip(mvrc_ops, boxes_features):
                        if mvrc_op == 1:
                            num_masked_boxes += 1
                            boxes_features[0] -= box_feat
                    boxes_features[0] /= (num_real_boxes - num_masked_boxes + 1e-5)
            else:
                mvrc_ops, mvrc_labels = self.random_mask_region(boxes_cls_scores)
            assert len(mvrc_ops) == boxes.shape[0], \
                "Error: mvrc_ops have length {}, expected {}!".format(len(mvrc_ops), boxes.shape[0])
            assert len(mvrc_labels) == boxes.shape[0], \
                "Error: mvrc_labels have length {}, expected {}!".format(len(mvrc_labels), boxes.shape[0])
        else:
            mvrc_ops = [0] * boxes.shape[0]
            mvrc_labels = [np.zeros_like(boxes_cls_scores[0])] * boxes.shape[0]

        # zero out pixels of masked RoI
        if (not self.with_precomputed_visual_feat) and self.mask_raw_pixels:
            for mvrc_op, box in zip(mvrc_ops, boxes):
                if mvrc_op == 1:
                    x1, y1, x2, y2 = box
                    image[:, int(y1):(int(y2) + 1), int(x1):(int(x2) + 1)] = 0

        mvrc_labels = np.stack(mvrc_labels, axis=0)

        text = self.tokenizer.convert_tokens_to_ids(text_tokens)

        if self.with_precomputed_visual_feat:
            boxes = torch.cat((boxes, boxes_features), dim=1)

        # truncate seq to max len
        if len(text) + len(boxes) > self.seq_len:
            text_len_keep = len(text)
            box_len_keep = len(boxes)
            while (text_len_keep + box_len_keep) > self.seq_len and (text_len_keep > 0) and (box_len_keep > 0):
                if box_len_keep > text_len_keep:
                    box_len_keep -= 1
                else:
                    text_len_keep -= 1
            if text_len_keep < 2:
                text_len_keep = 2
            if box_len_keep < 1:
                box_len_keep = 1
            boxes = boxes[:box_len_keep]
            text = text[:(text_len_keep - 1)] + [text[-1]]
            mlm_labels = mlm_labels[:(text_len_keep - 1)] + [mlm_labels[-1]]
            mvrc_ops = mvrc_ops[:box_len_keep]
            mvrc_labels = mvrc_labels[:box_len_keep]

        #return text, text_tokens, original_caption
        return image, boxes, im_info, text, relationship_label, mlm_labels, mvrc_ops, mvrc_labels
        #return image, boxes, im_info, text, temporal_label, mlm_labels, mvrc_ops, mvrc_labels

    def random_word_wwm(self, tokens):
        output_tokens = []
        output_label = []

        for i, token in enumerate(tokens):
            sub_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(token)
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    for sub_token in sub_tokens:
                        output_tokens.append("[MASK]")
                # 10% randomly change token to random token
                elif prob < 0.9:
                    for sub_token in sub_tokens:
                        output_tokens.append(random.choice(list(self.tokenizer.vocab.keys())))
                        # -> rest 10% randomly keep current token
                else:
                    for sub_token in sub_tokens:
                        output_tokens.append(sub_token)

                        # append current token to output (we will predict these later)
                for sub_token in sub_tokens:
                    try:
                        output_label.append(self.tokenizer.vocab[sub_token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        output_label.append(self.tokenizer.vocab["[UNK]"])
                        logging.warning("Cannot find sub_token '{}' in vocab. Using [UNK] insetad".format(sub_token))
            else:
                for sub_token in sub_tokens:
                    # no masking token (will be ignored by loss function later)
                    output_tokens.append(sub_token)
                    output_label.append(-1)

        ## if no word masked, random choose a word to mask
        # if all([l_ == -1 for l_ in output_label]):
        #    choosed = random.randrange(0, len(output_label))
        #    output_label[choosed] = self.tokenizer.vocab[tokens[choosed]]

        return output_tokens, output_label

    def random_mask_region(self, regions_cls_scores):
        num_regions, num_classes = regions_cls_scores.shape
        output_op = []
        output_label = []
        for k, cls_scores in enumerate(regions_cls_scores):
            prob = random.random()
            # mask region with 15% probability
            if prob < 0.15:
                prob /= 0.15

                if prob < 0.9:
                    # 90% randomly replace appearance feature by "MASK"
                    output_op.append(1)
                else:
                    # -> rest 10% randomly keep current appearance feature
                    output_op.append(0)

                # append class of region to output (we will predict these later)
                output_label.append(cls_scores)
            else:
                # no masking region (will be ignored by loss function later)
                output_op.append(0)
                output_label.append(np.zeros_like(cls_scores))

        # # if no region masked, random choose a region to mask
        # if all([op == 0 for op in output_op]):
        #     choosed = random.randrange(0, len(output_op))
        #     output_op[choosed] = 1
        #     output_label[choosed] = regions_cls_scores[choosed]

        return output_op, output_label

    @staticmethod
    def _converId(img_id):
        """
        conversion for image ID
        """
        img_id = img_id.split('-')
        if 'train' in img_id[0]:
            new_id = int(img_id[1])
        elif 'val' in img_id[0]:
            new_id = int(img_id[1]) + 1000000
        elif 'test' in img_id[0]:
            new_id = int(img_id[1]) + 2000000
        else:
            print("no split known")
        return new_id

    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())

    @staticmethod
    def group_aspect(database):
        print('grouping aspect...')
        t = time.time()

        # get shape of all images
        widths = torch.as_tensor([idb['width'] for idb in database])
        heights = torch.as_tensor([idb['height'] for idb in database])

        # group
        group_ids = torch.zeros(len(database))
        horz = widths >= heights
        vert = 1 - horz
        group_ids[horz] = 0
        group_ids[vert] = 1

        print('Done (t={:.2f}s)'.format(time.time() - t))

        return group_ids

    def __len__(self):
        return len(self.annot_data)

    def _load_image(self, path):
        return Image.open(path).convert('RGB')

    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)
import jsonlines
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
import pprint
pp = pprint.PrettyPrinter(width=41, compact=True)
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# wo_pretrain: no pretrain
# base: low
# ours: low_high

def get_pred_classes(file_name):
    logit = np.load(file_name)
    prob = softmax(logit, axis = 1)
    return np.argmax(prob, axis = 1)

def recover_sentence(input_list):
    return ' '.join(str(x) for x in input_list)

no_pt_pred_class = get_pred_classes("wo_pretrain.npy")
low_pt_pred_class = get_pred_classes("base.npy")
low_high_pt_pred_class = get_pred_classes("ours.npy")

vcr_data_root = Path('../../DATASETs/VCR+VisualCOMET/vcr1annots')
# comet_data_root = Path('visualcomet')

# vcr_ann = {}
# vcr_ann_count = {}
ann_list = []
with jsonlines.open(vcr_data_root / 'val.jsonl') as reader:
    for line in reader:
        img_id = line['img_fn']
        ann_list.append(line)
        # pp.pprint(line)
        # if img_id in vcr_ann:
        #     vcr_ann_count[img_id] += 1
        #     print(img_id, vcr_ann_count[img_id])
        # else:
        #     vcr_ann[img_id] = line
        #     vcr_ann_count[img_id] = 1

# print(pred_class.shape)
# print(len(ann_list))

for i in range(len(ann_list)):
    if ann_list[i]['answer_label'] == low_high_pt_pred_class[i] and \
        ann_list[i]['answer_label'] != no_pt_pred_class[i]:

        print("==========================")
        # pp.pprint(ann_list[i])
        print("Image: ", ann_list[i]['img_fn'])
        # download_cmd = "scp yueen@pineapple.cs.columbia.edu:VLBERT/data/vcr/vcr1images/{} ./imgs".format(ann_list[i]['img_fn'])
        # print(download_cmd)
        # os.system(download_cmd)

        print("Question:", recover_sentence(ann_list[i]['question']))
        print("Answers:")
        for ans in ann_list[i]["answer_choices"]:
            print("\t", recover_sentence(ans))

        print("True Answer:", ann_list[i]['answer_label'], " ------- ", recover_sentence(ann_list[i]["answer_choices"][ann_list[i]['answer_label']]) )
        print("No pretrain:", no_pt_pred_class[i], " ------- ", recover_sentence(ann_list[i]["answer_choices"][no_pt_pred_class[i]]))
        print("PT_low_high:", low_high_pt_pred_class[i], " ------- ", recover_sentence(ann_list[i]["answer_choices"][low_high_pt_pred_class[i]]))
        # print(ann_list[i]['answer_label'])
        # print(no_pt_pred_class[i])
        # print(low_pt_pred_class[i])
        # print(low_high_pt_pred_class[i])

        img = mpimg.imread(os.path.join("imgs", os.path.basename(ann_list[i]['img_fn'])))
        imgplot = plt.imshow(img)
        plt.show()


# match_count = 0
# comet_ann = {}
# comet_ann_count = {}
# with open(comet_data_root / 'val_annots.json') as reader:
#     data = json.load(reader)
#     for entry in data:
#         img_id = entry['img_fn']
#         if img_id in comet_ann:
#             comet_ann_count[img_id] += 1
#             print(img_id, comet_ann_count[img_id])
#         else:
#             comet_ann[img_id] = entry
#             comet_ann_count[img_id] = 1
#
#             match_count += vcr_ann_count[img_id]
#
# vcr_count = np.array(list(vcr_ann_count.values()))
# comet_count = np.array(list(comet_ann_count.values()))
# vcr_count.sort()
# comet_count.sort()
# print(vcr_count)
# print(comet_count)
#
# print(np.sum(vcr_count))
# print(np.sum(comet_count))
#
# print(len(vcr_ann.keys()))
# print(len(comet_ann.keys()))
#
# # print(list(vcr_ann_count.values()))
# # print(list(comet_ann_count.values()))
#
# # print(vcr_ann_count)
# # print(comet_ann_count)
# # print(len(vcr_ann))
# # print(len(comet_ann))
#
#
# print(match_count)
#

import jsonlines
import json
from pathlib import Path
import numpy as np
import pprint
pp = pprint.PrettyPrinter(width=41, compact=True)
# np.set_printoptions(threshold=sys.maxsize)

top_k = 1
SPLIT = 'train'
ANN_FILE = '../image-inference-80000-ckpt/train_sample_1_num_5_top_k_0_top_p_0.9.json'
OUT_FILE = "train_pred_annots.json"

dataset_path = Path('../../DATASET/')
vcr_data_root = Path('vcr1annots')
comet_data_root = Path('visualcomet')

# Read Predicted VisualCOMET annotations
with open(ANN_FILE) as f:
    comet_json = json.load(f)

comet_pred_dict = {}
for entry in comet_json:
    img_fn = entry['img_fn']
    # print(len(entry['generations'])) = 5
    if img_fn not in comet_pred_dict:
        comet_pred_dict[img_fn] = {entry['inference_relation'] : entry['generations'][0:top_k]}
        comet_pred_dict[img_fn]['metadata_fn'] = entry['metadata_fn']
        comet_pred_dict[img_fn]['movie'] = entry['movie']
        comet_pred_dict[img_fn]['place'] = entry['place']
        comet_pred_dict[img_fn]['event'] = entry['event']
    else:
        if entry['inference_relation'] not in comet_pred_dict[img_fn]:
            comet_pred_dict[img_fn][entry['inference_relation']] = entry['generations'][0:top_k]
        else:
            comet_pred_dict[img_fn][entry['inference_relation']] += entry['generations'][0:top_k]

# pp.pprint(comet_pred_dict)


# # Merge VCR with COMET annotations
# answer_labels = []
# ds_size = 0
# with jsonlines.open(dataset_path / vcr_data_root / 'val.jsonl') as reader:
#     for ann in reader:
#         if ann['img_fn'] in comet_pred_dict:
#             ds_size += 1
#             comet_entry = comet_pred_dict[ann['img_fn']]
#             db_i = {
#                 'annot_id': ann['annot_id'],
#                 'objects': ann['objects'],
#                 'img_fn': ann['img_fn'],
#                 'question': ann['question'],
#                 'answer_choices': ann['answer_choices'],
#                 'answer_label': ann['answer_label'],
#                 'rationale_choices': ann['rationale_choices'],
#                 'rationale_label': ann['rationale_label'],
#                 'movie': comet_entry['movie'],
#                 # 'metadata_fn': comet_entry['metadata_fn'],
#                 'place': comet_entry['place'],
#                 'event': comet_entry['event'],
#                 'intent': comet_entry['intent'],
#                 'before': comet_entry['before'],
#                 'after': comet_entry['after'],
#             }
#
#             # print(db_i['answer_choices'])
#             # print(db_i['answer_label'])
#             answer_labels.append(db_i['answer_label'])
#
# answer_labels = np.array(answer_labels)

# print(len(answer_labels))
# pred_padded = np.load("comet_a_pred.npy")
# print(pred_padded)
# print(pred_padded.shape)
# # print(np.sum(pred_padded, axis = 1))
#
# # for idx, elm in enumerate(np.sum(pred, axis = 1)):
# #     if elm == 0:
# #         print(idx) # 17958
# #         break
#
# valid_size = 17958
# pred = pred_padded[:valid_size, :]
# # print(np.sum(pred_padded) == np.sum(pred))
# print(np.sum(pred, axis = 1))
# pred_labels = np.argmax(pred, axis = 1)
# print((np.sum(pred_labels == answer_labels) + 0) / valid_size)


# Convert dict to a list which matches the orignal VisualCOMET json format
flatten = lambda t: [item for sublist in t for item in sublist]
comet_pred_list = []
stn_len_list = []
for img_fn in comet_pred_dict:
    cur_dict = {'img_fn'    : img_fn,
                'metadata_fn': comet_pred_dict[img_fn]['metadata_fn'],
                'movie'     : comet_pred_dict[img_fn]['movie'],
                'place'     : comet_pred_dict[img_fn]['place'],
                'event'     : comet_pred_dict[img_fn]['event'],
                'intent'    : comet_pred_dict[img_fn]['intent'],
                'before'    : comet_pred_dict[img_fn]['before'],
                'after'     : comet_pred_dict[img_fn]['after'],
                'split'     : SPLIT,
                }
    print("===============================")
    stn_len_list.append(len(flatten(cur_dict['intent'])) + len(flatten(cur_dict['before'])) + len(flatten(cur_dict['after'])))
    # stn_len_list.append(len(flatten(cur_dict['before'])))
    # stn_len_list.append(len(flatten(cur_dict['after'])))
    print(len(flatten(cur_dict['intent'])))
    print(len(flatten(cur_dict['before'])))
    print(len(flatten(cur_dict['after'])))
    comet_pred_list.append(cur_dict)

stn_len_list = np.array(stn_len_list)
print(np.std(stn_len_list))
print(np.mean(stn_len_list))

with open(OUT_FILE, 'w') as pred_file:
    json.dump(comet_pred_list, pred_file)

# # Read True VisualCOMET annotations
# with open(dataset_path / comet_data_root / 'val_annots.json') as f:
#     comet_json = json.load(f)
# comet_dict = {}
# for entry in comet_json:
#     comet_dict[entry['img_fn']] = entry
#     print("==============================================")
#     pp.pprint(entry)
#     pp.pprint(comet_pred_dict[entry['img_fn']])
#
# print(len(comet_pred_dict))
# print(len(comet_dict))

import jsonlines
import json
from pathlib import Path
import numpy as np
import pprint
import sys
import string
pp = pprint.PrettyPrinter(width=41, compact=True)
np.set_printoptions(threshold=sys.maxsize)

def convert_to_vcr_sentence(sentence):
    """
    Integers in sentences of the VisualCOMET dataset refers to objects in the
    image. When the sentences are read from json files, they are only treated
    as strings. This function will correct this issue by convert all integer
    tokens to a list containing the number.
    E.g.
    Original: "Where is 5 ?" --> ["Where", "is", "5", "?"]
    Correct:  "Where is 5 ?" --> ["Where", "is", [4], "?"]
    NOTE: index is 1 smaller than the original value
    """

    def is_int(token):
        """
        Check if token can be converted to an integer
        """
        try:
            int(token)
            return True
        except ValueError:
            return False

    def split_on_punctuation(token):
        """
        Mainly for dealing with 2's, which should be converted to [[2], "'", 's']
        """
        token_spaced = ''
        for char in token:
            if char in string.punctuation:
                token_spaced += ' ' + char + ' '
            else:
                token_spaced += char
        return token_spaced.split()

    stn_split = sentence.split()
    stn_w_index = []
    for token in stn_split:
        token_split = split_on_punctuation(token)
        for sub_token in token_split:
            if is_int(sub_token):
                stn_w_index.append([int(sub_token) - 1])
            else:
                stn_w_index.append(sub_token)
    return stn_w_index


dataset_path = Path('../../DATASET/')
vcr_data_root = Path('vcr1annots')
comet_data_root = Path('visualcomet')

vcr_ann = {}
vcr_ann_count = {}

with open(dataset_path / comet_data_root / 'val_annots.json') as f:
    comet_json = json.load(f)

comet_dict = {}
for entry in comet_json:
    if entry['img_fn'] in comet_dict:
        for field in ['intent', 'before', 'after']:
            # print("========================================================", field)
            # print(comet_dict[entry['img_fn']][field])
            # print(entry[field])
            comet_dict[entry['img_fn']][field] += entry[field]
            # print(comet_dict[entry['img_fn']][field])
    else:
        comet_dict[entry['img_fn']] = entry

# pp.pprint(comet_dict)

answer_labels = []
ds_size = 0
with jsonlines.open(dataset_path / vcr_data_root / 'val.jsonl') as reader:
    for ann in reader:
        if ann['img_fn'] in comet_dict:
            ds_size += 1
            comet_entry = comet_dict[ann['img_fn']]
            db_i = {
                'annot_id': ann['annot_id'],
                'objects': ann['objects'],
                'img_fn': ann['img_fn'],
                'question': ann['question'],
                'answer_choices': ann['answer_choices'],
                'answer_label': ann['answer_label'],
                'rationale_choices': ann['rationale_choices'],
                'rationale_label': ann['rationale_label'],
                'movie': comet_entry['movie'],
                # 'metadata_fn': comet_entry['metadata_fn'],
                'place': comet_entry['place'],
                'event': comet_entry['event'],
                'intent': comet_entry['intent'],
                'before': comet_entry['before'],
                'after': comet_entry['after'],
            }

            # print(db_i['answer_choices'])
            # print(db_i['answer_label'])
            # print(db_i['question'])
            for stn in comet_entry['intent']:
                print(convert_to_vcr_sentence(stn))
            answer_labels.append(db_i['answer_label'])

answer_labels = np.array(answer_labels)

# print(answer_labels)

pred_padded = np.load("pred_comet_a_pred.npy")
# print(pred_padded)
print(pred_padded.shape)
# print(np.sum(pred_padded, axis = 1))


# for idx, elm in enumerate(np.sum(pred, axis = 1)):
#     if elm == 0:
#         print(idx) # 17958
#         break

valid_size = 17958

pred = pred_padded[:valid_size, :]
# print(np.sum(pred_padded) == np.sum(pred))
# print(np.sum(pred, axis = 1))
pred_labels = np.argmax(pred, axis = 1)
# print(pred_labels)
print((np.sum(pred_labels == answer_labels) + 0) / valid_size)

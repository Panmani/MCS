import json
import os
import string
from tqdm import tqdm
import pickle
import spacy
# from ms_coco_classnames import coco_classes

COCO_PATH = "../../../DATASET/COCO-Caption"
OUT_DIR = "../../../DATASET/COCO-Caption"
split = "train"
NUM_PART = 2

def split_on_punctuation(token):
    """
    From VLBERT/vcr/data/datasets/vcr.py
    Mainly for dealing with 2's, which should be converted to [[2], "'", 's']
    """
    token_spaced = ''
    for char in token:
        if char in string.punctuation:
            token_spaced += ' ' + char + ' '
        else:
            token_spaced += char
    return token_spaced.split()

# def singularize(word):
#     token_list = nlp(word)
#     lemma = token_list[0].lemma_
#     # if len(token_list) == 1:
#     #     lemma = token_list[0].lemma_
#     #     print("====", lemma)
#     # elif len(token_list) == 0:
#     #     lemma = word
#     # else:
#     #     raise ValueError
#     return lemma

if __name__ == '__main__':

    assert split in ['train', 'val', 'test']


    img_dir = "{}2017".format(split)
    metadata_dir = "{}2017_meta".format(split)
    feat_dir = "{}2017_feat".format(split)

    GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery',
                            'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                            'Frankie', 'Pat', 'Quinn']

    nlp = spacy.load("en_core_web_sm")

    # with open(os.path.join(COCO_PATH, 'annotations', 'instances_{}2017.json'.format(split))) as json_file:
    #     ann = json.load(json_file)
    # Find images with person(s)
    img_id_with_person = []
    with open(os.path.join(COCO_PATH, 'annotations', 'person_keypoints_{}2017.json'.format(split))) as json_file:
        ann = json.load(json_file)
        for ann in tqdm(ann['annotations'], "Filter images with person"):
            if ann["image_id"] not in img_id_with_person:
                img_id_with_person.append(ann["image_id"])
        print("{} images have person".format(len(img_id_with_person)))

    person_nouns = set()
    with open("person_nouns.txt", 'r') as per_noun_file:
        for line in per_noun_file:
            per_noun = line.strip()
            token = nlp(per_noun)[0]
            person_nouns.add(token.lemma_)
    print("{} person nouns".format(len(person_nouns)))

    # object_vocab = []
    # with open("../demo/data/genome/1600-400-20/objects_vocab.txt", 'r') as obj_vocab_file:
    #     for line in obj_vocab_file:
    #         object_vocab.append(line.strip())
    #
    # print(coco_classes.values())
    # for coco_class in coco_classes.values():
    #     if coco_class not in object_vocab:
    #         print("=====", coco_class)

    # for entry in ann:
    #     print(entry)
    # info
    # licenses
    # images
    # annotations
    # categories

    with open(os.path.join(COCO_PATH, 'annotations', 'captions_{}2017.json'.format(split))) as json_file:
        coco_ann = json.load(json_file)
    print("{} images in total".format(len(coco_ann["images"])))

    img_id2filename = {}
    for img_info in coco_ann["images"]:
        img_id2filename[img_info['id']] = img_info["file_name"]

    non_person_ann = []
    non_person_tokens = set()
    records = []
    name_idx_rotator = 0
    for ann in tqdm(coco_ann['annotations'], "Generate records for GPT2"):
        if ann["image_id"] in img_id_with_person:
            file_name = img_id2filename[ann["image_id"]]
            metadata_fn = os.path.join(metadata_dir, file_name[:-len('.jpg')] + ".json")
            caption = ann["caption"]

            # caption_split = caption.split()
            # caption_punc_split = []
            # for token in caption_split:
            #     caption_punc_split += split_on_punctuation(token)

            # doc = nlp(" ".join(caption_punc_split))
            doc = nlp(caption)
            # if len(doc) != len(caption_punc_split):
            #     print("=======", " ".join(caption_punc_split))
            # has_person_noun = bool(set(caption_punc_split) & person_nouns)
            # if not has_person_noun:
            #     # print(file_name, caption_punc_split)
            #     continue

            has_person_noun = False
            caption_name_added_split = []
            for token in doc:
                # token_spacy = nlp(token.lower())[0]
                # token_single = singularize(token)
                if token.lemma_.lower() in person_nouns:
                    caption_name_added_split += [token.text, GENDER_NEUTRAL_NAMES[name_idx_rotator % len(GENDER_NEUTRAL_NAMES)]]
                    name_idx_rotator += 1
                    has_person_noun = True
                else:
                    caption_name_added_split.append(token.text)

                if token.pos_ == "NOUN" and token.lemma_.lower() not in person_nouns:
                    non_person_tokens.add(token.text.lower())

            caption_with_name = " ".join(caption_name_added_split)

            # doc = nlp(caption)
            # for chunk in doc.noun_chunks:
            #     if chunk.root.dep_ == "nsubj":
            #         print(chunk.text, "|||", caption)
            #         # print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)

            # sub_tokens = []
            # for token in doc:
            #     if token.dep_ == "nsubj":
            #         sub_tokens.append(token.text)
            #         # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
            # print(sub_tokens, caption)

            if has_person_noun:
                record = {
                    "img_fn"      : os.path.join(img_dir, file_name),
                    "metadata_fn" : metadata_fn,
                    "split"       : split,
                    "event"       : caption_with_name,
                }
                records.append(record)
                print(caption_with_name)
            else:
                non_person_ann.append(caption_with_name)

        #     with open(os.path.join(OUT_DIR, feat_dir, file_name[:-len('.jpg')] + ".pkl"), 'rb') as pkl_file:
        #         features_dict = pickle.load(pkl_file)
        #
        #     with open(metadata_fn, "w") as metadata_file:
        #         metadata = {}
        #         metadata
        #         metadata_file
        #         info['event_name'] = event_name
        #
        # else:
        #     print("Image {} has no person".format(img_id2filename[ann["image_id"]]))
    print(non_person_tokens)
    print("{} annotations are discarded".format(len(non_person_ann)))
    print("{} annotations for images with person".format(len(records)))

    part_size = int(len(records) / NUM_PART)
    if NUM_PART == 1:
        ann_filename = "{}2017_annots.json".format(split)
        with open(os.path.join(OUT_DIR, ann_filename), 'w') as out_file:
            json.dump(records, out_file)
        print("{} is the output file".format(ann_filename))
    elif NUM_PART > 1:
        for part_id in range(NUM_PART):
            ann_filename = "{}_p{}_2017_annots.json".format(split, part_id)
            start_idx = part_size * part_id
            end_idx = part_size * (part_id + 1)
            if part_id < NUM_PART - 1:
                part_records = records[start_idx: end_idx]
            else:
                part_records = records[start_idx:]
            with open(os.path.join(OUT_DIR, ann_filename), 'w') as out_file:
                json.dump(part_records, out_file)
                print("{} records were written into {}".format(len(part_records), ann_filename))
            # print("{} is the output file".format(ann_filename))
    else:
        raise ValueError

    # event = record['event_name']
    # inference = record['inference_relation']
    # inference_text = record['inference_text_name']
    # record['metadata_fn']
    # record['img_fn']
    #
    # metadata['boxes']
    # metadata['segms']
    # metadata['names']
    # metadata['width']
    # metadata['height']

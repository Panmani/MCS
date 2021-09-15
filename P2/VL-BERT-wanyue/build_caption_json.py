import re
import os
import json
import nltk
import copy

import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--stage', default=1, type=int, choices=[1,2])

parser.add_argument('--use_gt', default=False, type=boolean_string)
parser.add_argument('--question_format', default=False, type=boolean_string)
parser.add_argument('--include_event', default=True, type=boolean_string)
parser.add_argument('--include_temporal', default=True, type=boolean_string)
parser.add_argument('--unify_level', default=0, type=int, choices=[0, 1, 2], help='0=not unified; 1=unify at temporal level; 2=unify at event level')
parser.add_argument('--additional_event', default=False, type=boolean_string)

parser.add_argument('--annot_folder', type=str, default='data/cc-temporal-captions')
parser.add_argument('--val_gen_path', type=str, default='val_sample_1_num_5_top_k_0_top_p_0.9.json')
parser.add_argument('--trn_gen_path', type=str, default='train_sample_1_num_5_top_k_0_top_p_0.9.json')
parser.add_argument('--val_vcr_annots_path', type=str, default='data/vcr/val.jsonl')
parser.add_argument('--trn_vcr_annots_path', type=str, default='data/vcr/train.jsonl')
parser.add_argument('--val_vc_annots_path', type=str, default='data/visualcomet/val_annots.json')
parser.add_argument('--trn_vc_annots_path', type=str, default='data/visualcomet/train_annots.json')
parser.add_argument('--noun_vocab_path', default='person_nouns.txt', type=str, help='external vocab path for subject extractor')


parser.add_argument('--test', default=False, type=boolean_string)
args = parser.parse_args()

#random_name_path = '../ERNIE/ernie-vil/data/vcr/unisex_names_table.csv'
GENDER_NEUTRAL_NAMES = ['Casey', 'Riley', 'Jessie', 'Jackie', 'Avery', 'Jaime', 'Peyton', 'Kerry', 'Jody', 'Kendall',
                        'Frankie', 'Pat', 'Quinn']


def extract_subject(event, noun_vocab=None):
    pattern = '([0-9]+ and [0-9]+)|([0-9]+)'
    regex = re.compile(pattern)
    subjects = regex.findall(event)
    for subs in subjects:
        for sub in subs:
            if len(sub):
                return sub
    tokens = nltk.word_tokenize(event)
    if noun_vocab is not None:
        cand_subjects = []
        subject = ''
        for t in tokens:
            if t in GENDER_NEUTRAL_NAMES:
                subject = t
                break
            if t in noun_vocab:
                cand_subjects.append(t)

        if not subject and len(cand_subjects):
            subject = cand_subjects[0]
        # 对于COCO有问题！
        # if subject:
        #     subject = subject + ' ' + np.random.choice(GENDER_NEUTRAL_NAMES)
        return subject

    token_tags = nltk.pos_tag(tokens)
    subject = ''
    for tt in token_tags:
        if tt[1] in ['VBZ', 'IN', 'VBP']:
            break
        subject += tt[0]
        subject += ' '
    subject = subject.rstrip()

    if args.test:
        with open('special_event.txt', 'a+') as f:
            f.write(event)
            f.write('\n')

    return subject


def build_raw_caption(event, temporal_token, generation, subject,
                      include_event=False, question_format=False):
    if event[-1] == '.':
        event = event[:-1]
    if question_format:
        assert include_event

    if not include_event:
        if temporal_token == 'intent':
            return 'Because, {} wanted to {}.'.format(subject, generation)
        elif temporal_token == 'before':
            return 'Before, {} needed to {}.'.format(subject, generation)
        elif temporal_token == 'after':
            return 'After, {} will most likely {}.'.format(subject, generation)
    # Final Choice:
    elif not question_format:
        if temporal_token == 'intent':
            return 'Because {} wanted to {}, {}.'.format(subject, generation, event)
        elif temporal_token == 'before':
            return 'Before {}, {} needed to {}.'.format(event, subject, generation)
        elif temporal_token == 'after':
            return 'After {}, {} will most likely {}.'.format(event, subject, generation)
    else:
        if temporal_token == 'intent':
            return 'Why is {}? Because {} wanted to {}.'.format(event, subject, generation)
        elif temporal_token == 'before':
            return 'What was {} doing before {}? {} needed to {}.'.format(subject, event, subject, generation)
        elif temporal_token == 'after':
            return 'What will {} do after {}? {} will most likely to {}.'.format(subject, event, subject, generation)
    return None


def replace_random_name(caption, subject2name={}):
    pattern = '([0-9]+)'
    regex = re.compile(pattern)
    subjects_need_replace = list(set(regex.findall(caption)))

    subject_count = len(subjects_need_replace)
    if subject_count > len(GENDER_NEUTRAL_NAMES):
        random_names = np.random.choice(GENDER_NEUTRAL_NAMES, subject_count, replace=True)
    else:
        random_names = np.random.choice(GENDER_NEUTRAL_NAMES, subject_count, replace=False)

    for i in range(len(subjects_need_replace)):
        if subjects_need_replace[i] not in subject2name:
            subject2name[subjects_need_replace[i]] = random_names[0]
            random_names = random_names[1:]

    new_caption = ''
    for char in caption:
        if char in subject2name:
            new_caption += subject2name[char]
        else:
            new_caption += char

    return new_caption.rstrip(), subject2name


def build_caption(data_entry, args, additional_event=None):
    noun_vocab=None
    if args.noun_vocab_path:
        with open(args.noun_vocab_path, 'r') as f:
            noun_vocab = [line.rstrip() for line in f.readlines()]

    subject = extract_subject(data_entry['event'], noun_vocab)
    captions = []
    subject2name = {}

    # Event sanity check: remove '\n' and repeated punctuation
    # event = data_entry['event'].replace('\n', '')
    # event_token = event.split(' ')
    # while not event_token[-1] or event_token[-1] == '.':
    #     event_token.pop()
    # event = ' '.join(event_token)
    # data_entry['event'] = event

    if not args.include_temporal:
        caption = data_entry['event'] + '.'
        if 'vc-temporal-captions' in args.annot_folder:
            caption, subject2name = replace_random_name(caption, subject2name)
        if additional_event is not None:
            img_id = data_entry['img_fn'].split('/')[1][:-4]
            additional_captions = additional_event[img_id]
            add_caption = additional_captions[0]
            for cap in additional_captions:
                if len(cap.split(' ')) < len(add_caption.split(' ')):
                    add_caption = cap

            caption += ' ' + add_caption


        return [caption]

    for i in range(len(data_entry['generations'])):
        caption = build_raw_caption(data_entry['event'],
                                    data_entry['inference_relation'],
                                    data_entry['generations'][i],
                                    subject,
                                    include_event=args.include_event,
                                    question_format=args.question_format)
        if 'vc-temporal-captions' in args.annot_folder:
            caption, subject2name = replace_random_name(caption, subject2name)
        captions.append(caption)
    return captions

# def build_unified_caption(event, temporal_gen, level=1):
#     assert level in [1, 2]
#     subject = extract_subject(event)
#     temporal_captions_list = []
#     output_caption = []
#     subject2name = {}
#     if level==1:
#         for temporal_token, generations in temporal_gen:
#             temporal_captions = []
#             for generation in generations:
#                 caption = build_raw_caption(event, temporal_token, generation, subject,
#                                             include_event=False, question_format=False)
#                 temporal_captions.append(caption)
#             temporal_captions_list.append(temporal_captions)
#         temporal_captions_list = list(product(*temporal_captions_list))
#
#         for i in range(len(temporal_captions_list)):
#             caption = ' '.join([event] + list(temporal_captions_list[i]))
#             caption, subject2name = replace_random_name(caption, subject2name)
#             output_caption.append(caption)
#
#     elif level==2:
#         pass
#
#     return output_caption


# def unify(data_entry, imgfn2imgid, imgfn2events, imgfnevent2gens, level=1):
#     imgfn = data_entry['img_fn']
#     event = data_entry['event']
#
#     imgfnevent = imgfn + event
#     tempo_gen = imgfnevent2gens[imgfnevent]
#     captions = build_unified_caption(event, tempo_gen, level=1)
#
#     return {'img_id': imgfn2imgid[imgfn],
#             'event': event,
#             'temporal_generation': tempo_gen,
#             'captions': captions}
def main_stage_2(data_split, args):
    vcr_annots_path = args.trn_vcr_annots_path if data_split == 'train' else args.val_vcr_annots_path
    with open(vcr_annots_path, 'r') as json_file:
        image_annot = [json.loads(json_str) for json_str in list(json_file)]

    for task in ['qa', 'qar']:
        pretrain_annot = []
        for i in tqdm(range(len(image_annot))):
            data_entry = image_annot[i]
            question = data_entry['question_orig']
            answer = data_entry['answer_orig']
            rationale = data_entry['rationale_orig']
            caption = question + ' <SEP> ' + answer
            if task == 'qar':
                caption += ' <SEP> ' + rationale

            caption, _ = replace_random_name(caption, {})

            pretrain_annot.append({
                'img_id': data_entry['img_id'],
                'captions': [caption]
            })

        file_name = '{}_{}_caption.json'.format(data_split, task)
        args.annot_folder = 'data/vcr'
        with open(os.path.join(args.annot_folder, file_name), 'w') as outfile:
            json.dump(pretrain_annot, outfile)

def main(data_split, args):
    assert data_split in ['train', 'val']

    gen_path = args.trn_gen_path if data_split == 'train' else args.val_gen_path
    vc_annots_path = args.trn_vc_annots_path if data_split == 'train' else args.val_vc_annots_path
    vcr_annots_path = args.trn_vcr_annots_path if data_split == 'train' else args.val_vcr_annots_path

    with open(os.path.join(args.annot_folder, gen_path)) as json_file:
        caption_annot = json.load(json_file)
    with open(vc_annots_path) as json_file:
        gt_caption_annot = json.load(json_file)
    with open(vcr_annots_path, 'r') as json_file:
        image_annot = [json.loads(json_str) for json_str in list(json_file)]
    additional_event=None
    if args.additional_event:
        with open(os.path.join(args.annot_folder, 'inferred_caption.json')) as json_file:
            additional_event = json.load(json_file)

    imgfn2imgid = {image_annot[i]['img_fn']: image_annot[i]['img_id'] for i in range(len(image_annot))}
    imgfn2gt = {}
    for i in range(len(gt_caption_annot)):
        info = {'event': gt_caption_annot[i]['event'],
                'intent': gt_caption_annot[i]['intent'],
                'before': gt_caption_annot[i]['before'],
                'after': gt_caption_annot[i]['after']}
        if gt_caption_annot[i]['img_fn'] not in imgfn2gt:
            imgfn2gt[gt_caption_annot[i]['img_fn']] = [info]
        else:
            imgfn2gt[gt_caption_annot[i]['img_fn']].append(info)

    imgfn2events = {} # for unifying caption
    imgfnevent2gens = {} # for unifying caption
    temporal_caption_annot = []
    for i in tqdm(range(len(caption_annot))):
        # Use Ground Truth Temporal Info:
        if args.use_gt:
            gt_event = caption_annot[i]['event']
            gt_generations = []

            found = False
            for info in imgfn2gt[caption_annot[i]['img_fn']]:
                if info['event'] == gt_event:
                    found = True
                    gt_generations = info[caption_annot[i]['inference_relation']]
            assert found

            data_entry = {
                'inference_relation': caption_annot[i]['inference_relation'],
                'event': gt_event,
                'generations': gt_generations
            }
        # Use Generated Temporal Info:
        else:
            data_entry = caption_annot[i]


        captions = build_caption(data_entry, args, additional_event=additional_event)
        if not captions: # gt里面有些temporal caption是空的
            continue

        caption_annot[i]['captions'] = captions
        if 'cc' not in args.annot_folder: # if use coco, don't need img_id
            caption_annot[i]['img_id'] = imgfn2imgid[caption_annot[i]['img_fn']]

        temporal_caption_annot.append(caption_annot[i])
        if args.unify_level > 0:
            imgfn2events[caption_annot[i]['img_fn']] = \
                list(set(imgfn2events.get(caption_annot[i]['img_fn'], []) + [caption_annot[i]['event']]))

            imgfnevent2gens[caption_annot[i]['img_fn']+caption_annot[i]['event']] = \
                imgfnevent2gens.get(caption_annot[i]['img_fn']+caption_annot[i]['event'], []) + \
                [(caption_annot[i]['inference_relation'], caption_annot[i]['generations'])]


    if args.include_temporal:
        file_name = '{}_with_events.json'.format(data_split) if args.include_event else '{}.json'.format(data_split)
    else:
        file_name = '{}_only_events.json'.format(data_split) if args.include_event else '{}.json'.format(data_split)

    if args.additional_event:
        file_name = 'adev_'+file_name

    if args.question_format:
        file_name = 'qf_'+file_name
    if args.use_gt:
        file_name = 'gt_'+file_name


    if args.unify_level == 1:
        print('Unify caption level 1.')
        temporal_unicaption_annot = []

        with Pool(64) as p:
            unify_partial = partial(unify, imgfn2imgid=imgfn2imgid, imgfn2events=imgfn2events, imgfnevent2gens=imgfnevent2gens)
            temporal_unicaption_annot = p.map(unify_partial, caption_annot)

        # for i in tqdm(range(len(caption_annot))):
        #     imgfn = caption_annot[i]['img_fn']
        #
        # #for imgfn in tqdm(imgfn2events):
        #     for event in imgfn2events[imgfn]:
        #         imgfnevent = imgfn+event
        #         tempo_gen = imgfnevent2gens[imgfnevent]
        #         captions = build_unified_caption(event, tempo_gen, level=args.unify_level)
        #         temporal_unicaption_annot.append({
        #             'img_id': imgfn2imgid[imgfn],
        #             'event': event,
        #             'temporal_generation': tempo_gen,
        #             'captions': captions
        #         })
        temporal_caption_annot = temporal_unicaption_annot
        file_name = 'l1_'+file_name
    elif args.unify_level == 2:
        pass

    print('Save to {}'.format(file_name))
    if not args.test:
        with open(os.path.join(args.annot_folder, file_name), 'w') as outfile:
            json.dump(temporal_caption_annot, outfile)



if __name__ == '__main__':
    assert args.include_temporal or args.include_event
    if args.question_format:
        args.include_event = True
    if not args.include_temporal:
        args.question_format = False


    print(args)
    for data_split in ['train', 'val']:
        print('data_split: {}'.format(data_split))

        if args.stage == 1:
            main(data_split, args)
        else:
            main_stage_2(data_split, args)

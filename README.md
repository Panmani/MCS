# Machine Commonsense (MCS)
The project aims to experiment with adding temporal information from the VisualCOMET dataset to the multiple choice questions in the VCR dataset and see whether the additional sentences can improve the accuracy on the VCR task.

> Work in progress

## Example

![GT VL-BERT TRAIN](images/1569.png)


```
'img_fn': 'lsmdc_3005_ABRAHAM_LINCOLN_VAMPIRE_HUNTER/3005_ABRAHAM_LINCOLN_VAMPIRE_HUNTER_00.27.43.141-00.27.45.534@0.jpg',
'img_id': 'val-1569'

============================ VCR ============================
{
  "answer_choices":
  a) Something else has captured her attention.
  b) [2] is turned around because [0] is speaking with her.
  c) She is too embarrassed to look at him.
  d) [2] does not want [0] to see something that is upsetting her.

  "answer_label": a)

  ...
}

======================== VisualCOMET ========================
{
  "place": "at a fancy party",

  "event": "1 is trying to talk to the pretty woman in front of him",

  "intent": ["ask the woman on a date",
             "get over his shyness"],

  "before": ["approach 3 at an event",
             "introduce himself to 3",
             "be invited to a dinner party",
             "dress in formal attire"],

  "after": ["ask 3 to dance",
            "try to make a date with 3",
            "greet her by kissing her hand",
            "order a drink from the server"]

  ...
}
```
> The indices in the annotations start from 0 while the indices in images start from 1.


## GPT-2
We use the pre-trained GPT-2 to infer annotations for the training set of the VisualCOMET dataset.

### Usage
Initialize the model
```
$ pip install .
```

Run inference for --split:
```
$ python scripts/run_generation.py --data_dir ../visualcomet/ --model_name_or_path my_experiment/image-inference-80000-ckpt/ --split train --overwrite_cache
```

## VL-BERT
With the predicted VisualCOMET annotations, we re-train the VL-BERT model so that it accommodates temporal information. I.e., the 'intent', 'before', 'after' information from the VisualCOMET annotations.

### Usage
Train:
```
$ ./scripts/dist_run_single.sh 4 vcr/train_end2end.py cfgs/vcr/base_q2a_4x16G_fp32.yaml ./ckpt
```
Eval:
```
$ python vcr/val.py \
  --a-cfg cfgs/vcr/base_q2a_4x16G_fp32.yaml \
  --a-ckpt ckpt/output/vl-bert-original/vcr/base_q2a_4x16G_fp32/vcr1images_train/vl-bert_base_a_res101-best.model \
  --gpus 1 \
  --result-path eval --result-name original
```

## Current Results
When VL-BERT is trained with ground truth VisualCOMET annotations, we get about 3% increase in VCR accuracy:

### Training set
![GT VL-BERT TRAIN](images/train_acc.png)

### Validation set
![GT VL-BERT VAL](images/val_acc.png)

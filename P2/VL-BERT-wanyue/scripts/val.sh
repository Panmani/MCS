#!/usr/bin/env bash

#python vcr/val.py \
#  --a-cfg ./cfgs/vcr/VC_base_q2a_4x16G_fp32.yaml \
#  --r-cfg ./cfgs/vcr/VC_base_q2a_4x16G_fp32.yaml \
#  --a-ckpt checkpoint_pretrain/output/vl-bert/vcr/VC_base_q2a_4x16G_fp32/vcr1images_train/vl-bert_base_a_res101-best.model \
#  --r-ckpt checkpoint_pretrain/output/vl-bert/vcr/VC_base_q2a_4x16G_fp32/vcr1images_train/vl-bert_base_a_res101-best.model \
#  --gpus 0 1 2 3 \
#  --result-path var_result --result-name pretrained

python vcr/val.py \
  --a-cfg ./cfgs/vcr/base_q2a_4x16G_fp32.yaml \
  --r-cfg ./cfgs/vcr/base_q2a_4x16G_fp32.yaml \
  --a-ckpt checkpoint_no_pretrain/output/vl-bert/vcr/base_q2a_4x16G_fp32/vcr1images_train/vl-bert_base_a_res101-best.model \
  --r-ckpt checkpoint_no_pretrain/output/vl-bert/vcr/base_q2a_4x16G_fp32/vcr1images_train/vl-bert_base_a_res101-best.model \
  --gpus 0 1 2 3 \
  --result-path var_result --result-name not_pretrained
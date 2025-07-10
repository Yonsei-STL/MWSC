#!/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=1 python train_finetune.py \
    --image_path '/data/MWSC/data/' \
    --train_label_path '/data/MWSC/data/label/train_data.csv' \
    --val_label_path '/data/MWSC/data/label/val_data.csv' \
    --output_path '/data/MWSC/result/' \
    --timm_model_name 'deit3_base_patch16_224'\
    --pretrain 'False'\
    --num_epochs 50\
    --batch_size 64\
    --learning_rate 0.0001\
    --num_workers 8;


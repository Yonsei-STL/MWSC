#!/bin/bash
cd ..
CUDA_VISIBLE_DEVICES=0 python train.py \
    --image_path '/data/MWSC/data/' \
    --train_label_path '/data/MWSC/data/label/train_data.csv' \
    --val_label_path '/data/MWSC/data/label/val_data.csv' \
    --output_path '/data/MWSC/result/' \
    --clip_base_model 'ViT-B/32'\
    --ablation_mode 1\
    --num_epochs 50\
    --batch_size 64\
    --learning_rate 0.0001\
    --num_workers 8;
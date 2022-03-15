#!/bin/bash
# python3 main_clip.py \
#     --save-frequency 1 \
#     --report-to tensorboard \
#     --val-data="../data/validation.txt"  \
#     --csv-img-key images \
#     --csv-caption-key captions \
#     --csv-separator " " \
#     --warmup 100 \
#     --batch-size=2 \
#     --lr=1e-3 \
#     --wd=0.1 \
#     --epochs=2 \
#     --workers=1 \
#     --model RN50 \
#     --dataset-type csv \
#     --train-data="../data/train.txt"  \
#     --zeroshot-frequency 0 \

# python3 main_clip.py \
#     --val-data="../data/validation.txt"  \
#     --csv-img-key images \
#     --csv-caption-key captions \
#     --csv-separator " " \
#     --dataset-type csv \
#     --openai-pretrained \
#     --model RN50
# #     --resume ../../8gpu_lr5e-4_yfcc_epoch32.pt

python3 main_vqgan.py --base vqgan/configs/my_custom_vqgan.yaml -t True

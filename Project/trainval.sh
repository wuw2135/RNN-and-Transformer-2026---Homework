#!/bin/bash
CDUA_VISIBLE_DEVICES=1

python trainval.py --name Baseline10_dino_re_ --gpu_ids 0 --batch_size 1 --fpn_channels 96 --backbone lwganet --num_epochs 200 --lr 5e-4 --weight_decay 5e-3 --num_workers 4 --segmodel dino --create_model base
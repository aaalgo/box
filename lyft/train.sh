#!/bin/bash

../train.py --augments=augments.json --db scratch/train.db --anchor_stride=2 --model model --nocache --ckpt_epochs 5

#!/bin/sh

uv run main.py validate \
  -c configs/dinov3/coco/instance/eomt_large_640.yaml \
  --model.network.masked_attn_enabled False \
  --trainer.devices 1 \
  --data.batch_size 4 \
  --data.path $DATA/coco \
  --model.ckpt_path ckpt/dinov3/coco/instance/eomt_large_640.bin

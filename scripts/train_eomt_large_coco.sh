#!/bin/sh

uv run main.py fit \
  -c ./configs/dinov3/coco/instance/eomt_large_640.yaml \
  --trainer.devices 1 \
  --data.batch_size 4 \
  --data.path $DATA/coco


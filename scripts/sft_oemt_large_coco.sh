#1/bin/sh

uv run main.py fit \
  -c configs/dinov3/coco/instance/eomt_large_640.yaml \
  --trainer.devices 1 \
  --data.batch_size 4 \
  --data.path $DATA/coco \


# --model.ckpt_path ckpt/dinov3/coco/instance/pytorch_model.bin \
# --model.load_ckpt_class_head False

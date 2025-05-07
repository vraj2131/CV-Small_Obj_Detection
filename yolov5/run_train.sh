#!/usr/bin/env bash
#
# run_train.sh — launch yolov5 training on VedAI

# 1) cd into the yolov5 directory (assumes this script lives in
#    MultiModalFusion/ at the same level as yolov5/ and data/ )
cd "$(dirname "$0")/yolov5" || { echo "❌ yolov5 folder not found"; exit 1; }

# 2) invoke python explicitly
python train.py \
  --data ../data/vedai9.yaml \
  --cfg yolov5m.yaml \
  --weights yolov5m.pt \
  --batch-size 16 \
  --imgsz 512 \
  --epochs 50 \
  --device 0 \
  --project runs/train \
  --name vedai9_experiment \
  --exist-ok

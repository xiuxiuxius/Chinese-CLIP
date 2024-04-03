#!/usr/bin/env

# 环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip
DATAPATH=/data/limiaomiao/project/data
dataset_name=COCO-CN
split=test

resume=${DATAPATH}/experiments/coco-cn_finetune_vit-b-16_roberta-base_bs1024_1gpu_20240327152035/checkpoints/epoch_latest.pt

# 提取特征
python -u cn_clip/eval/extract_features.py  \
    --extract-image-feats    \
    --extract-text-feats   \
    --image-data="${DATAPATH}/datasets/${dataset_name}/lmdb/${split}/imgs"  \
    --text-data="${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl"  \
    --img-batch-size=512   \
    --text-batch-size=512  \
    --context-length=52  \
    --resume=${resume}   \
    --vision-model=ViT-B-16  \
    --text-model=RoBERTa-wwm-ext-base-chinese

# 文搜图
python -u cn_clip/eval/make_topk_predictions.py \
    --image-feats="${DATAPATH}/datasets/${dataset_name}/${split}_imgs.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${dataset_name}/${split}_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/datasets/${dataset_name}/${split}_predictions.jsonl"

# 图搜文
python -u cn_clip/eval/make_topk_predictions_tr.py \
    --image-feats="${DATAPATH}/datasets/${dataset_name}/${split}_imgs.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${dataset_name}/${split}_texts.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/datasets/${dataset_name}/${split}_tr_predictions.jsonl"

# 评估文搜图
python cn_clip/eval/evaluation.py \
    ${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl \
    ${DATAPATH}/datasets/${dataset_name}/${split}_predictions.jsonl \
    output.json
cat output.json

# 评估图搜文
python cn_clip/eval/transform_ir_annotation_to_tr.py \
    --input ${DATAPATH}/datasets/${dataset_name}/${split}_texts.jsonl
python cn_clip/eval/evaluation_tr.py \
    ${DATAPATH}/datasets/${dataset_name}/${split}_texts.tr.jsonl \
    ${DATAPATH}/datasets/${dataset_name}/${split}_tr_predictions.jsonl \
    output.json
cat output.json

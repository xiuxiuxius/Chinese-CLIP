# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------------

# Project : Chinese-CLIP
# File    : mergeJson.py
# Date    : 2024/3/28 16:16
# Desc    : 
# Author  : limiaomiao
# E-mail  : biubiubius@qq.com

# ------------------------------------------------------------------------------

import json
import os
import shutil

from tqdm import tqdm

json_id_text = "/data/limiaomiao/project/data/datasets/MUGE/test_texts.jsonl"
json_id_img = "/data/limiaomiao/project/data/datasets/MUGE/test_predictions.jsonl"
json_text_output = "/data/limiaomiao/project/data/datasets/MUGE/test_submit.jsonl"

text_ids = []
text_contents = []
text_imgs = []

with open(json_id_text, "r") as fin:
    for line in tqdm(fin):
        obj = json.loads(line.strip())
        text_ids.append(obj["text_id"])
        text_contents.append(obj["text"])
        # print(obj)
        # print(text_ids, text_contents)
        # break

with open(json_id_img, "r") as fin:
    for line in tqdm(fin):
        obj = json.loads(line.strip())
        text_imgs.append(obj["image_ids"])
        # print(obj)
        # print(text_imgs)
        # break

with open(json_text_output, "w", encoding = 'utf8') as fout:
    for text_id, text_content, text_img in zip(text_ids, text_contents, text_imgs):
        fout.write("{}\n".format(json.dumps({"query_id": text_id, "query_text": text_content, "item_ids": text_img}, ensure_ascii=False)))


# Change directory
os.chdir('/data/limiaomiao/project/data/datasets/MUGE')

shutil.copy('test_submit.jsonl', 'MR_test_queries.jsonl')

shutil.make_archive('MR_result_20240402', 'zip', '.', 'MR_test_queries.jsonl')

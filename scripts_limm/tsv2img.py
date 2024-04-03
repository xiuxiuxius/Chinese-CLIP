# -*- coding: utf-8 -*-

# ------------------------------------------------------------------------------

# Project : Chinese-CLIP
# File    : tsv2img.py
# Date    : 2024/4/2 18:37
# Desc    : 
# Author  : limiaomiao
# E-mail  : biubiubius@qq.com

# ------------------------------------------------------------------------------

import csv
import base64
import os
import sys
from io import BytesIO
from PIL import Image
import pandas as pd

csv.field_size_limit(2147483647)

tsv_file = "/data/limiaomiao/project/data/datasets/MUGE/test_imgs.tsv"


with open(tsv_file) as fd:
    total_num = sum(1 for line in fd)

with open(tsv_file) as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    img_base64_list = []
    idx = 1

    for row in rd:
        # print(row)
        # break
        id, img_base64 = row
        img_base64_list.append(img_base64)
        # print(id)
        img = Image.open(BytesIO(base64.urlsafe_b64decode(img_base64)))
        muge_test_dir = "/data/limiaomiao/project/multimodal-retrieval-fastapi/imgs/muge_test_images"
        if not os.path.exists(muge_test_dir):
            os.makedirs(muge_test_dir)
        img.save(f"{muge_test_dir}/{id}.jpg")
        print(f"{idx}/{total_num}  -------    {id}.jpg saved.")
        idx += 1
        # break
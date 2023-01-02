import os
import urllib.request
import zipfile
import tarfile

"""
수동 다운로드 목록
mask.tar.gz : https://www.dropbox.com/s/bd9ty7b4fqd5ebf/mask.tar.gz?dl=0 
coco.json : https://www.dropbox.com/s/0sj2q24hipiiq5t/COCO.json?dl=0
pose_model_scratch.pth : https://www.dropbox.com/s/5v654d2u65fuvyr/pose_model_scratch.pth?dl=0
"""

# 데이터 폴더 생성
def download_data():
    data_dir = "./data/"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    weights_dir = "./weights/"
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)

    # 대략 6GB 정도(15분 소요)
    url = "http://images.cocodataset.org/zips/val2014.zip"
    target_path = os.path.join(data_dir, "val2014.zip")

    if not os.path.exists(target_path):
        urllib.request.urlretrieve(url, target_path)

        zip = zipfile.ZipFile(target_path)
        zip.extractall(data_dir)
        zip.close()

    save_path = os.path.join(data_dir, "mask.tar.gz")

    with tarfile.open(save_path, 'r:*') as tar:
        tar.extractall(data_dir)


import json
import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
import torch.utils.data as data

def make_datapath_list(rootpath):

    json_path = osp.join(rootpath, 'COCO.json')
    with open(json_path) as data_file:
        data_this = json.load(data_file)
        data_json = data_this['root']

    # index
    num_samples = len(data_json)
    train_indexes = []
    val_indexes = []
    for count in range(num_samples):
        if data_json[count]['isValidation'] != 0.:
            val_indexes.append(count)
        else:
            train_indexes.append(count)

    # 화상 경로
    train_img_list = list()
    val_img_list = list()

    for idx in train_indexes:
        img_path = os.path.join(rootpath, data_json[idx]['img_paths'])
        train_img_list.append(img_path)

    for idx in val_indexes:
        img_path = os.path.join(rootpath, data_json[idx]['img_paths'])
        val_img_list.append(img_path)

    # 마스크 데이터 경로
    train_mask_list = []
    val_mask_list = []

    for idx in train_indexes:
        img_idx = data_json[idx]['img_paths'][-16:-4]
        anno_path = "./data/mask/train2014/mask_COCO_tarin2014_" + img_idx+'.jpg'
        train_mask_list.append(anno_path)

    for idx in val_indexes:
        img_idx = data_json[idx]['img_paths'][-16:-4]
        anno_path = "./data/mask/val2014/mask_COCO_val2014_" + img_idx+'.jpg'
        val_mask_list.append(anno_path)

    # 어노테이션
    train_meta_list = list()
    val_meta_list = list()

    for idx in train_indexes:
        train_meta_list.append(data_json[idx])

    for idx in val_indexes:
        val_meta_list.append(data_json[idx])

    return train_img_list, train_mask_list, val_img_list, val_mask_list, train_meta_list, val_meta_list

def check_data():
    train_img_list, train_mask_list, val_img_list, val_mask_list, train_meta_list, val_meta_list = make_datapath_list(
        rootpath="./data/")

    index = 52
    print(val_meta_list[index])

    # 화상
    img = cv2.imread(val_img_list[index])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()

    # 마스크
    mask_miss = cv2.imread(val_mask_list[index])
    mask_miss = cv2.cvtColor(mask_miss, cv2.COLOR_BGR2RGB)
    plt.imshow(mask_miss)
    plt.show()

    # 합성
    blend_img = cv2.addWeighted(img, 0.4, mask_miss, 0.6, 0)
    plt.imshow(blend_img)
    plt.show()


check_data()



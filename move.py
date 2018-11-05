import json
import shutil
import os
from config import config
from glob import glob
from tqdm import tqdm

try:
    for i in range(0,61):
        # 0-60共61类
        os.mkdir("D:\challengeai-database\pmm_newdata\images/" + str(i))
except:
    pass

with open(config.train_jsonpath) as trainjs:
    file_train = json.load(trainjs, encoding="utf-8")

with open(config.val_jsonpath) as valjs:
    file_val = json.load(valjs, encoding="utf-8")

file_list = file_train + file_val

# print(file_list)
for file in tqdm(file_list):
    filename = file["image_id"]
    origin_path = "D:\challengeai-database\_trainingset_20181023\AgriculturalDisease_trainingset\images/" + filename
    ids = file["disease_class"]
    save_path = "D:\challengeai-database\pmm_newdata\images/" + str(ids) + "/"
    shutil.copy(origin_path, save_path)

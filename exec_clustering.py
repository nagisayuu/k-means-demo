import os
import shutil
import numpy as np
from PIL import Image
from skimage import data
from sklearn.cluster import KMeans

# 1. 国旗画像のサイズをそろえて保存する
# ./flag_origin 以下に国旗画像
# ./flag_convert 以下に200*100のサイズに変換したjpgを保存
for path in os.listdir('./flag_origin'):
    img = Image.open(f'./flag_origin/{path}')
    img = img.convert('RGB')
    img_resize = img.resize((200, 100))
    img_resize.save(f'./flag_convert/{path}.jpg')


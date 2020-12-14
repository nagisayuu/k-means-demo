import os
import shutil
import numpy as np
from PIL import Image
from skimage import io
from sklearn.cluster import KMeans

# 1. 国旗画像のサイズをそろえて保存する
# ./flag_origin 以下に国旗画像
# ./flag_convert 以下に200*100のサイズに変換したjpgを保存
for path in os.listdir('./flag_origin'):
    img = Image.open(f'./flag_origin/{path}')
    img = img.convert('RGB')
    img_resize = img.resize((200, 100))
    img_resize.save(f'./flag_convert/{path}.jpg')

# 2. 3次元配列の画像データを2次元配列のデータに変換
feature = np.array([io.imread(f'./flag_convert/{path}',plugin='matplotlib') for path in os.listdir('./flag_convert')])
feature = feature.reshape(len(feature), -1).astype(np.float64)

# 3. 学習(15種類のグループにクラスタリングする)
model = KMeans(n_clusters=15).fit(feature)

# 4. 学習結果のラベル
labels = model.labels_

# 5. 学習結果(クラスタリング結果の表示 + ラベルごとにフォルダ分け)
# ./flag_group 以下に画像を分けて保存する
for label, path in zip(labels, os.listdir('./flag_convert')):
    os.makedirs(f"./flag_group/{label}", exist_ok=True)
    shutil.copyfile(f"./flag_origin/{path.replace('.jpg', '')}", f"./flag_group/{label}/{path.replace('.jpg', '')}")
    print(label, path)
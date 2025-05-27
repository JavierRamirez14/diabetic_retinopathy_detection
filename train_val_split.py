import os
import shutil
import pandas as pd
import numpy as np

imgs_path = 'Datos/train_images_resized_150'
train_path = 'Datos/train_150'
val_path = 'Datos/val_150'
path_labels = 'Datos/trainLabels.csv'
train_labels_path = 'Datos/train_labels_150.csv'
val_labels_path = 'Datos/val_labels_150.csv'

def train_val_split(imgs_path, train_path, val_path, path_labels, train_labels_path, val_labels_path, val_size=0.15):
    print("Dividiendo datos en train y validación...")
    list_imgs = os.listdir(imgs_path)
    list_nums = sorted(list(set([img.split('_')[0] for img in list_imgs])))

    np.random.seed(42)
    n_samples = int(len(list_nums) * val_size)
    list_imgs_val_ids = np.random.choice(list_nums, size=n_samples, replace=False)
    list_imgs_train_ids = [x for x in list_nums if x not in list_imgs_val_ids]

    list_imgs_train = [img for img in list_imgs if img.split('_')[0] in list_imgs_train_ids]
    list_imgs_val = [img for img in list_imgs if img.split('_')[0] in list_imgs_val_ids]

    for path in [val_path, train_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    for img in list_imgs_val:
        shutil.copy(os.path.join(imgs_path, img), os.path.join(val_path, img))

    for img in list_imgs_train:
        shutil.copy(os.path.join(imgs_path, img), os.path.join(train_path, img))

    df = pd.read_csv(path_labels)

    df_train = df[df['image'].isin([os.path.splitext(x)[0] for x in os.listdir(train_path)])]
    df_val = df[df['image'].isin([os.path.splitext(x)[0] for x in os.listdir(val_path)])]

    df_train.to_csv(train_labels_path, index=False)
    df_val.to_csv(val_labels_path, index=False)

    print('Proceso de división terminado.')
    print(f'Imágenes en train: {len(list_imgs_train)} (IDs: {len(df_train)})')
    print(f'Imágenes en val: {len(list_imgs_val)} (IDs: {len(df_val)})')

train_val_split(imgs_path, train_path, val_path, path_labels, train_labels_path, val_labels_path)
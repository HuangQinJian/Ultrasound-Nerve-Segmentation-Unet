import os
import shutil
import numpy as np
from PIL import Image
data_path = 'raw/'


def move_mask_files():
    train_data_path = os.path.join(data_path, 'train/')
    mask_data_path = os.path.join(data_path, 'mask/')
    print(train_data_path)
    print(mask_data_path)
    images = os.listdir(train_data_path)
    if not os.path.exists(mask_data_path):
        os.makedirs(mask_data_path)
    print(images)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        image_mask_path = train_data_path+image_mask_name
        print(image_mask_name)
        print(image_mask_path)
        shutil.move(image_mask_path, mask_data_path)
        # os.unlink(image_mask_path)


def tif2ppng():

    train_data_path = os.path.join(data_path, 'train/')
    mask_data_path = os.path.join(data_path, 'mask/')

    train_data_path_png = os.path.join(data_path, 'train_png/')
    mask_data_path_png = os.path.join(data_path, 'mask_png/')
    if not os.path.exists(train_data_path_png):
        os.makedirs(train_data_path_png)
    if not os.path.exists(mask_data_path_png):
        os.makedirs(mask_data_path_png)
    img = os.listdir(train_data_path)
    for image_name in img:
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = Image.open(train_data_path+image_name)
        img_mask = Image.open(mask_data_path+image_mask_name)
        img.save(train_data_path_png+image_name.split('.')[0]+'.png')
        img_mask.save(mask_data_path_png+image_name.split('.')[0]+'.png')


if __name__ == '__main__':
    # move_mask_files()
    tif2ppng()

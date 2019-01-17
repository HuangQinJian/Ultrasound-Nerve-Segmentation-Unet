from __future__ import print_function

import os
import numpy as np

from skimage.io import imsave, imread
from skimage.transform import resize

data_path = 'raw/'

processed_data_path = 'processed/'

image_rows = 96
image_cols = 96

if not os.path.exists('processed'):
    os.mkdir('processed')


def create_train_data():
    image_data_path = os.path.join(data_path, 'total/image')
    mask_data_path = os.path.join(data_path, 'total/mask')
    images = os.listdir(image_data_path)
    # masks = os.listdir(mask_data_path)
    total = len(images)

    imgs = np.ndarray((int(total), image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray(
        (int(total), image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(image_data_path, image_name), as_grey=True)
        img_mask = imread(os.path.join(
            mask_data_path, image_mask_name), as_grey=True)

        img = resize(img, (image_rows, image_cols), preserve_range=True)
        img_mask = resize(img_mask, (image_rows, image_cols),
                          preserve_range=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        # img = resize(img, (image_rows, image_cols), preserve_range=True)
        # img_mask = resize(img_mask, (image_rows, image_cols),
        #                   preserve_range=True)

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print(imgs_mask.shape)
    print('Loading done.')
    ##############################################################################
    image_augume_data_path = os.path.join(data_path, 'total/augume_image')
    mask_augume_data_path = os.path.join(data_path, 'total/augume_mask')
    images_augume = os.listdir(image_augume_data_path)
    # masks = os.listdir(mask_data_path)
    total_augume = len(images_augume)

    imgs_augume = np.ndarray(
        (int(total_augume), image_rows, image_cols), dtype=np.uint8)
    imgs_mask_augume = np.ndarray(
        (int(total_augume), image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating augume training images...')
    print('-'*30)
    for image_name in images_augume:
        img_augume = imread(os.path.join(
            image_augume_data_path, image_name), as_grey=True)
        img_mask_augume = imread(os.path.join(
            mask_augume_data_path, image_name), as_grey=True)

        img_augume = np.array([img_augume])
        img_mask_augume = np.array([img_mask_augume])

        imgs_augume[i] = img_augume
        imgs_mask_augume[i] = img_mask_augume

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total_augume))
        i += 1
    print(imgs_mask_augume.shape)
    print('Loading augume data done.')
    ##############################################################################
    images_concate = np.concatenate((imgs, imgs_augume), axis=0)
    masks_concate = np.concatenate((imgs_mask, imgs_mask_augume), axis=0)

    np.save(processed_data_path+'imgs_train_concate.npy', images_concate)
    np.save(processed_data_path+'imgs_mask_concate.npy', masks_concate)

    print('合并后的图片大小为:', images_concate.shape)
    print('合并后的Mask大小为:', masks_concate.shape)


def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img = resize(img, (image_rows, image_cols), preserve_range=True)
        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(processed_data_path+'imgs_test.npy', imgs)
    np.save(processed_data_path+'imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


if __name__ == '__main__':
    create_train_data()
    create_test_data()

import os

import numpy as np

from skimage import transform

from skimage.io import imsave, imread

data_path = 'raw/'

processed_data_path = 'processed/'

image_rows = 420
image_cols = 580

if not os.path.exists('processed'):
    os.mkdir('processed')


def data_rotation():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = int(len(images) / 2)

    i = 0

    imgs = np.ndarray((total, image_rows,
                       image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray(
        (total, image_rows, image_cols), dtype=np.uint8)

    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img_mask = imread(os.path.join(
            train_data_path, image_mask_name), as_grey=True)

        img_rotated = transform.rotate(img, 10)
        img_mask_rotated = transform.rotate(img_mask, 10)

        img = np.array([img_rotated])
        img_mask = np.array([img_mask_rotated])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1

    print('Loading done.')

    np.save(processed_data_path+'imgs_train_rotated.npy', imgs)
    np.save(processed_data_path+'imgs_mask_rotated.npy', imgs_mask)
    print('Rotated image saving to .npy files done.')


if __name__ == '__main__':
    data_rotation()

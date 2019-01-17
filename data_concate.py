import numpy as np
from skimage import img_as_float

processed_data_path = 'processed/'

masks = np.load(processed_data_path+'imgs_mask_train.npy')
masks = img_as_float(masks)
masks_rotation = np.load(processed_data_path+'imgs_mask_rotated.npy')

images = np.load(processed_data_path+'imgs_train.npy')
# print(images[2])
# print(images.dtype)
images = img_as_float(images)
# print(images.dtype)
# print(images[2])
images_rotation = np.load(processed_data_path+'imgs_train_rotated.npy')
# print(images_rotation.dtype)

masks_concate = np.concatenate((masks, masks_rotation), axis=0)
images_concate = np.concatenate((images, images_rotation), axis=0)
# print(images_concate[12])
# print(images_concate.dtype)

# masks = np.load(processed_data_path+'imgs_mask_train.npy')
# masks_noisy = np.load(processed_data_path+'imgs_mask_noisy.npy')

# images = np.load(processed_data_path+'imgs_train.npy')
# images_noisy = np.load(processed_data_path+'imgs_train_noisy.npy')

# masks_concate = np.concatenate((masks, masks_noisy), axis=0)
# images_concate = np.concatenate((images, images_noisy), axis=0)

print('合并后的图片大小为:', images_concate.shape)
print('合并后的Mask大小为:', masks_concate.shape)

np.save(processed_data_path+'imgs_train_concate.npy', images_concate)
np.save(processed_data_path+'imgs_mask_concate.npy', masks_concate)

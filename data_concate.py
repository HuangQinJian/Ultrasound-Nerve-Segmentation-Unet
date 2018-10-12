import numpy as np

processed_data_path = 'processed/'

masks = np.load(processed_data_path+'imgs_mask_train.npy')
masks_rotation = np.load(processed_data_path+'imgs_mask_rotated.npy')

images = np.load(processed_data_path+'imgs_train.npy')
images_rotation = np.load(processed_data_path+'imgs_train_rotated.npy')

masks_concate = np.concatenate((masks, masks_rotation), axis=0)
images_concate = np.concatenate((images, images_rotation), axis=0)

print('合并后的图片大小为:', images_concate.shape)
print('合并后的Mask大小为:', masks_concate.shape)

np.save(processed_data_path+'imgs_train_concate.npy', images_concate)
np.save(processed_data_path+'imgs_mask_concate.npy', masks_concate)

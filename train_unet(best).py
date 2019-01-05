from __future__ import print_function

import os
import tensorflow as tf
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras import initializers
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

plt.switch_backend('agg')
K.set_image_data_format('channels_last')  # TF dimension ordering in this code


os.environ["CUDA_VISIBLE_DEVICES"] = "1,6,7"

img_rows = 96
img_cols = 96

smooth = 1.
gamma = 2.
alpha = .25

initializer = initializers.RandomUniform(
    minval=-0.05, maxval=0.05, seed=None)

processed_data_path = 'processed/'


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def focal_loss(y_true, y_pred):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + smooth))-K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + smooth))


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(conv4)

    up5 = concatenate([Conv2DTranspose(256, (2, 2), strides=(
        2, 2), padding='same')(conv4), conv3], axis=3)

    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(up5)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(conv5)

    up6 = concatenate([Conv2DTranspose(128, (2, 2), strides=(
        2, 2), padding='same')(conv5), conv2], axis=3)

    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(up6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(conv6)

    up7 = concatenate([Conv2DTranspose(64, (2, 2), strides=(
        2, 2), padding='same')(conv6), conv1], axis=3)

    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(up7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same',
                   kernel_initializer=initializer)(conv7)

    conv8 = Conv2D(1, (1, 1), activation='sigmoid')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])

    model.compile(optimizer=Adam(lr=1e-5),
                  loss=dice_coef_loss, metrics=[dice_coef])
    plot_model(model, to_file='NetStruct.png', show_shapes=True)
    return model


def preprocess(imgs):
    # imgs.shape[0]代表图片数量
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    print('所用图片的数量为：', imgs.shape[0])
    """
    skimage.transform.resize(image, output_shape)
    image: 需要改变尺寸的图片
    output_shape: 新的图片尺寸
    """
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)
    # np.newaxis在使用和功能上等价于None,其实就是None的一个别名,为numpy.ndarray（多维数组）增加一个轴
    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def load_train_data():
    # imgs_train = np.load(processed_data_path+'imgs_train_concate.npy')
    # imgs_mask_train = np.load(processed_data_path+'imgs_mask_concate.npy')
    imgs_train = np.load(processed_data_path+'imgs_train.npy')
    imgs_mask_train = np.load(processed_data_path+'imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def load_test_data():
    imgs_test = np.load(processed_data_path+'imgs_test.npy')
    imgs_id = np.load(processed_data_path+'imgs_id_test.npy')
    return imgs_test, imgs_id


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model.summary()
    model_checkpoint = ModelCheckpoint(
        'weights.h5', monitor='val_loss', save_best_only=True)

    tensorboard = TensorBoard(log_dir='./logs',  # log 目录
                              histogram_freq=1,  # 按照何等频率（epoch）来计算直方图，0为不计算
                              batch_size=32,     # 用多大量的数据计算直方图
                              write_graph=True,  # 是否存储网络结构图
                              write_grads=False,  # 是否可视化梯度直方图
                              write_images=False,  # 是否可视化参数
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None)

    earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=0)

    reduce_lr = ReduceLROnPlateau(
        factor=0.1, patience=2, min_lr=0.00001, verbose=1)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    history = model.fit(imgs_train, imgs_mask_train, batch_size=32, epochs=90, verbose=1, shuffle=True,
                        validation_split=0.2,
                        callbacks=[model_checkpoint, tensorboard, earlystop, reduce_lr])

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='valid')
    plt.title("model dice_coef_loss")
    plt.ylabel("dice_coef_loss")
    plt.xlabel("epoch")
    plt.legend(["train", "valid"], loc="upper left")
    plt.savefig('dice_coef_loss_performance.png')
    plt.clf()
    plt.plot(history.history['dice_coef'], label='train')
    plt.plot(history.history['val_dice_coef'], label='valid')
    plt.title("model dice_coef")
    plt.ylabel("dice_coef")
    plt.xlabel("epoch")
    plt.legend(["train", "valid"], loc="upper left")
    plt.savefig('dice_coef_performance.png')

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)


if __name__ == '__main__':
    train_and_predict()

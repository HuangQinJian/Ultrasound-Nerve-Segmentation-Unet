from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
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

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"

img_rows = 96
img_cols = 96
smooth = 1.
batch_size = 32

data_path = 'raw/'

if not os.path.exists(os.path.join(data_path, 'augume_image')):
    os.makedirs(os.path.join(data_path, 'augume_image'))

if not os.path.exists(os.path.join(data_path, 'augume_mask')):
    os.makedirs(os.path.join(data_path, 'augume_mask'))

initializer = initializers.RandomUniform(
    minval=-0.05, maxval=0.05, seed=None)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


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

    model.compile(optimizer=Adam(lr=1e-4),
                  loss=dice_coef_loss, metrics=[dice_coef])
    plot_model(model, to_file='NetStruct.png', show_shapes=True)
    return model


def train_data_generate_augument():
    # we create two instances with the same arguments
    data_gen_args = dict(featurewise_center=False,  # set input mean to 0 over the dataset
                         samplewise_center=False,  # set each sample mean to 0
                         featurewise_std_normalization=False,  # divide inputs by std of the dataset
                         samplewise_std_normalization=False,  # divide each input by its std
                         zca_whitening=False,  # apply ZCA whitening
                         # randomly rotate images in the range (degrees, 0 to 180)
                         rotation_range=180,
                         # randomly shift images horizontally (fraction of total width)
                         # width_shift_range=0.05,
                         # randomly shift images vertically (fraction of total height)
                         # height_shift_range=0.05,
                         # fill_mode='constant',
                         # cval=0.,
                         horizontal_flip=True,  # randomly flip images
                         vertical_flip=True  # randomly flip images
                         )
    # preprocessing_function=lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the flow methods
    seed = 1

    image_generator = image_datagen.flow_from_directory(
        os.path.join(data_path, 'train_image'),
        # os.path.join(data_path, 'a'),
        class_mode=None,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        color_mode='grayscale',
        save_to_dir=os.path.join(data_path, 'augume_image'),
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        os.path.join(data_path, 'train_mask'),
        # os.path.join(data_path, 'b'),
        class_mode=None,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        color_mode='grayscale',
        save_to_dir=os.path.join(data_path, 'augume_mask'),
        seed=seed)

    # combine generators into one which yields image and masks
    # train_generator = zip(image_generator, mask_generator)
    # return train_generator
    i = 0
    for batch in image_generator:
        i += 1
        if i > 352:
            break
        if i == 1:
            print(batch.shape)
    i = 0
    for batch in mask_generator:
        i += 1
        if i > 352:
            break
        if i == 1:
            print(batch.shape)

    """新生成的图片数量与i有关,假设文件夹下数目为n,i = (n/batch_size)*想增广的图片数量倍数
    1）i=2,每张图片变形新生成两张；
    2）i=1,每张图片变形新生成一张；
    """

    return image_generator, mask_generator


def train_process():
    print('-'*30)
    print('Loading and preprocessing train data...')
    # train_generator = train_data_generate_augument()
    image_generator, mask_generator = train_data_generate_augument()
    train_generator = zip(image_generator, mask_generator)
    print('-'*30)

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model.summary()
    model_checkpoint = ModelCheckpoint(
        'weights.h5', monitor='val_loss', save_best_only=True)

    tensorboard = TensorBoard(log_dir='./logs',  # log 目录
                              histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
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

    # history = model.fit(
    #     image_generator, mask_generator, epochs=100, batch_size=batch_size, validation_split=0.2, callbacks=[reduce_lr, earlystop, model_checkpoint, tensorboard], verbose=1)
    history = model.fit_generator(
        train_generator, epochs=100, steps_per_epoch=image_generator.n//batch_size, callbacks=[model_checkpoint, tensorboard], verbose=1)

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


if __name__ == '__main__':
    # train_process()
    print(train_data_generate_augument())

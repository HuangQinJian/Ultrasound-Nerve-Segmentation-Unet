from __future__ import print_function

import os
import datetime
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from sklearn.cross_validation import KFold

K.set_image_data_format('channels_last')  # TF dimension ordering in this code


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

img_rows = 96
img_cols = 96

smooth = 1.

processed_data_path = 'processed/'


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # pool1 = Dropout(0.25)(pool1)
    # pool1 = BatchNormalization()(pool1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # pool2 = Dropout(0.5)(pool2)
    # pool2 = BatchNormalization()(pool2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # pool3 = Dropout(0.5)(pool3)
    # pool3 = BatchNormalization()(pool3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # pool4 = Dropout(0.5)(pool4)
    # pool4 = BatchNormalization()(pool4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(
        2, 2), padding='same')(conv5), conv4], axis=3)
    # up6 = Dropout(0.5)(up6)
    # up6 = BatchNormalization()(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(
        2, 2), padding='same')(conv6), conv3], axis=3)
    # up7 = Dropout(0.5)(up7)
    # up7 = BatchNormalization()(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(
        2, 2), padding='same')(conv7), conv2], axis=3)
    # up8 = Dropout(0.5)(up8)
    # up8 = BatchNormalization()(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(
        2, 2), padding='same')(conv8), conv1], axis=3)
    # up9 = Dropout(0.5)(up9)
    # up9 = BatchNormalization()(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    # conv9 = Dropout(0.5)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5),
                  loss=dice_coef_loss, metrics=[dice_coef])

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
        imgs_p[i] = resize(imgs[i], (img_rows, img_cols), preserve_range=True)
    # np.newaxis在使用和功能上等价于None,其实就是None的一个别名,为numpy.ndarray（多维数组）增加一个轴
    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    # return a.tolist()
    return a


def load_train_data():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train = np.load(processed_data_path+'imgs_train_concate.npy')
    imgs_mask_train = np.load(processed_data_path+'imgs_mask_concate.npy')

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    imgs_mask_train[imgs_mask_train > 0.5] = 1
    imgs_mask_train[imgs_mask_train <= 0.5] = 0

    return imgs_train, imgs_mask_train


def load_test_data():
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)

    imgs_test = np.load(processed_data_path+'imgs_test.npy')
    imgs_id = np.load(processed_data_path+'imgs_id_test.npy')

    imgs_test = preprocess(imgs_test)

    mean = np.mean(imgs_test)  # mean for data centering
    std = np.std(imgs_test)  # std for data normalization

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    return imgs_test, imgs_id


def train_and_predict():

    imgs_train, imgs_mask_train = load_train_data()
    imgs_test, imgs_id_test = load_test_data()

    nfolds = 5
    batch_size = 32
    epoch = 100
    random_state = 2018

    yfull_test = []

    scores = []

    kf = KFold(len(imgs_train), n_folds=nfolds,
               shuffle=True, random_state=random_state)

    num_fold = 0
    for train_index, test_index in kf:
        model = get_unet()
        X_train, X_valid = imgs_train[train_index], imgs_train[test_index]
        Y_train, Y_valid = imgs_mask_train[train_index], imgs_mask_train[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        model_checkpoint = ModelCheckpoint(
            'weights.h5', monitor='val_loss', save_best_only=True)
        earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
        reduce_lr = ReduceLROnPlateau(
            factor=0.1, patience=5, min_lr=0.00001, verbose=1)

        callbacks = [
            model_checkpoint, earlystop, reduce_lr
        ]
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epoch,
                  shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
                  callbacks=callbacks)

        predictions_valid = model.predict(
            X_valid, batch_size=batch_size, verbose=1)
        score = dice_coef_loss(Y_valid, predictions_valid)
        scores += score
        print('Score dice_coef_loss: ', score)

        # Store test predictions
        test_prediction = model.predict(
            imgs_test, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    print('Average dice_coef_loss: {}, rows: {} cols: {} nfolds: {} epoch: {}'.format(
        scores/nfolds, img_rows, img_cols, nfolds, epoch))

    info_string = '_r_' + str(img_rows) \
        + '_c_' + str(img_cols) \
        + '_folds_' + str(nfolds) \
        + '_ep_' + str(batch_size)

    test_res = merge_several_folds_mean(yfull_test, nfolds)

    np.save('submission_' + info_string + '_' + str(datetime.datetime.now()
                                                    .strftime("%Y-%m-%d-%H-%M"))+'_' + 'imgs_mask_test.npy', test_res)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)
    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(test_res, imgs_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)


if __name__ == '__main__':
    train_and_predict()

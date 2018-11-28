from keras.models import load_model
from keras import backend as K
from keras.preprocessing import image
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from keras import models

smooth = 1.
img_path = '5508.tif'


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def load_img_tensor(img_path):
    img = imread(img_path, as_grey=True)
    print(img.shape)
    img = resize(img, (96, 96), preserve_range=True)
    print(img.shape)
    img_tensor = img[..., np.newaxis]
    img_tensor = np.expand_dims(img_tensor, axis=0)
    print(img_tensor.shape)
    img_tensor = img_tensor.astype('float32')
    mean = np.mean(img_tensor)  # mean for data centering
    std = np.std(img_tensor)  # std for data normalization

    img_tensor -= mean
    img_tensor /= std

    img_tensor /= 255.
    print(img_tensor.shape)
    return img_tensor


def show_activationlayer(img_tensor):
    # 第一层并不能输出，否则会报错    # https://github.com/keras-team/keras/issues/10372
    # Extracts the outputs of the top 8 layers:（提取前 1-8 层的输出）
    layer_outputs = [layer.output for layer in model.layers[1:8]]
    # 创建一个模型，给定模型输入，可以返回这些输出

    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    # 返回 7 个 Numpy 数组组成的列表， 每个层激活对应一个 Numpy 数组
    activations = activation_model.predict(img_tensor)

    first_layer_activation = activations[1]
    print('第二层输出的大小为：', first_layer_activation.shape)

    plt.matshow(first_layer_activation[0, :, :, 3], cmap='viridis')
    plt.show()


def show_all_activationlayer(img_tensor):
    # 第一层并不能输出，否则会报错    # https://github.com/keras-team/keras/issues/10372
    layer_outputs = [layer.output for layer in model.layers[1:8]]
    # 创建一个模型，给定模型输入，可以返回这些输出

    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    # 返回 7 个 Numpy 数组组成的列表， 每个层激活对应一个 Numpy 数组
    activations = activation_model.predict(img_tensor)
    # 层的名称，这样你可以将这些名称画到图中
    layer_names = []
    for layer in model.layers[1:8]:
        layer_names.append(layer.name)

    images_per_row = 16

    # 显示特征图
    for layer_name, layer_activation in zip(layer_names, activations):
        # 特征图中的特征个数 n_features
        n_features = layer_activation.shape[-1]

        # 特征图的形状为 (1, size, size, n_features)  size
        size = layer_activation.shape[1]

        # 在这个矩阵中将激活通道平铺
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        # 将每个过滤器平铺到一个大的水平网格中
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                                 :, :,
                                                 col * images_per_row + row]

                # 对特征进行后处理，使其看起来更美观
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,
                             row * size: (row + 1) * size] = channel_image
        # 显示网格
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()


def show_last_activationlayer(img_tensor):
    layer_outputs = [layer.output for layer in model.layers[1:]]
    # 创建一个模型，给定模型输入，可以返回这些输出
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

    activations = activation_model.predict(img_tensor)

    last_layer_activation = activations[-1]
    print('最后一层输出的大小为：', last_layer_activation.shape)

    plt.matshow(last_layer_activation[0, :, :, 0], cmap='viridis')
    plt.show()


if __name__ == '__main__':
    # 有自定义的损失函数以及评价指标必须声明
    model = load_model('weights.h5', custom_objects={
        'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

    print(model.summary())
    img_tensor = load_img_tensor(img_path)
    # show_activationlayer(img_tensor)
    # show_all_activationlayer(img_tensor)
    show_last_activationlayer(img_tensor)

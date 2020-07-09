from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf

def get_model(shape, layer):
    i = Input(shape)
    o = layer(i)
    model = Model(i, o)
    model.compile(optimizer="adam", loss="categorical_crossentropy")
    return model

def get_vgg16_first_layer_model(img_shape):
    from tensorflow.keras.applications.vgg16 import VGG16
    model = VGG16()
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    layer_name = 'block1_conv1'
    layer = layer_dict[layer_name]
    return get_model(img_shape, layer)

def get_mobilenet_first_layer_model(img_shape):
    from tensorflow.keras.applications.mobilenet import MobileNet
    model = MobileNet()
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    layer_name = 'conv1'
    layer = layer_dict[layer_name]
    layer.padding = "same"
    layer.strides = (1, 1)
    return get_model(img_shape, layer)

def get_xception_first_layer_model(img_shape):
    from tensorflow.keras.applications.xception import Xception
    model = Xception()
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer_name = 'block1_conv1'
    layer = layer_dict[layer_name]
    layer.padding = "same"
    layer.strides = (1, 1)
    return get_model(img_shape, layer)

def get_mobilenet_v2_first_layer_model(img_shape):
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
    model = MobileNetV2()
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    layer_name = 'conv1'
    layer = layer_dict[layer_name]
    layer.padding = "same"
    layer.strides = (1, 1)
    return get_model(img_shape, layer)

def get_resnet50_first_layer_model(img_shape):
    from tensorflow.keras.applications.resnet50 import ResNet50
    model = ResNet50()

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer_name = 'conv1'
    layer = layer_dict[layer_name]
    layer.padding = "same"
    layer.strides = (1, 1)
    return get_model(img_shape, layer)


def print_conv_layer_filters(layer):
    filters, biases = layer.get_weights()

    f_min, f_max = np.amin(filters), np.amax(filters)
    filters = (filters - f_min) / (f_max - f_min)

    n_filters, index = filters.shape[-1], 1

    plt.figure(figsize=[15, 100])

    plt.subplots_adjust(wspace=0.5, hspace=1)

    for i in range(n_filters):
        f = filters[:, :, :, i]
        num = 3
        cols = 6
        for j in range(num):
            ax = plt.subplot(n_filters, cols, index)
            ax.set_title("convolution %d" % (index))
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(f[:, :, j], cmap='viridis')
            index += 1

def to_constant(x, dtype = tf.float64):
    return tf.constant(x, dtype = dtype)

def to_variable(x, dtype = tf.float64):
    return tf.Variable(x, dtype = dtype)

def create_image_tensor(image_path, padding=0):
    img = np.pad(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float64), padding, "constant").tolist()
    return tf.Variable(img)
    pass

def create_matrix(shape, dtype=tf.float64, init_value=0):
    myVar = [[tf.Variable(init_value, dtype=dtype) for x in range(shape)] for y in range(shape)]
    print(myVar)
    tf.print(myVar)
    return myVar

def set_value(matrix, y,x, value):
    matrix[y][x] = matrix[y][x].assign(value)
    return matrix

def get_value(matrix, y,x):
    return matrix[y][x]

def set_value_range(matrix, y_start,y_end, x_start,x_end, value):
    while(y_start!=y_end):
        while(x_start!=x_end):
            matrix[y_start][x_start] = matrix[y_start][x_start].assign(value)
            x_start+=1
        y_start+=1
    return matrix
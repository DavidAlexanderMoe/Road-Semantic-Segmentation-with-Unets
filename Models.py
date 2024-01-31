# python -m Models

# General
import tensorflow as tf
from tensorflow.keras import backend as K
# import segmentation_models as sm

# Network Architecture
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout, activation
from keras.optimizers import Adam

# Metrics
from keras.metrics import MeanIoU


############################################################################################################################################################################

# Dice coefficient - Metric
# really useful in image segmentation tasks with imbalanced data
# could also be used as a loss
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

# https://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf
def dice_loss(y_true, y_pred):
    loss = 1 - dice_coef(y_true, y_pred)
    return loss
# model.compile(...,loss=dice_loss,...)


# IoU (already imported) - Metric
def IoU(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return intersection / union

############################################################################################################################################################################

# BLOCKS
def contraction_block(input_tensor, num_filters, dropout_rate):
    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input_tensor)      # strides default to 1

    # Dropout rate in between the Conv layers -> we got better results
    if dropout_rate > 0:
        x = Dropout(rate = dropout_rate)(x)
    
    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)

    pooled = MaxPooling2D((2, 2))(x)            # OG paper proposed strides=2 but we leave them default to pool_size = 2
    return x, pooled


# will be useful also for autoencoder
def convolution_block(input_tensor, num_filters, dropout_rate):
    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input_tensor)      # strides default to 1
    if dropout_rate > 0:
        x = Dropout(rate = dropout_rate)(x)
    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    return x


def expansive_block(copy, input_tensor, num_filters, dropout_rate):
    # filters = num_filters / 2
    
    # Conv2dtranspose =! upsampling (both the concept and the keras layer). both increase dim of arrays.
    # upsampling2d is the opposite of pooling repeating rows and columns of input.
    # Conv2dtranspose performs upsampling and then convolution. 
    x = Conv2DTranspose(num_filters, (2, 2), strides = (2, 2), padding='same')(input_tensor)       

    # Concatenation: crop the copy from the specular contraction block and concatenate it to the
    # current respective decoder layer of the expansive path
    concatenation = concatenate([x, copy])

    # add simple 2D convolutions
    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(concatenation)

    if dropout_rate > 0:
        x = Dropout(rate = dropout_rate)(x)

    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)

    return x


###### CHANGE NUM FILTERS TO NUM_FILTERS: NUM_FILTERS,*2,*4,*8,*16,*8,*4,*2,NUM_FILTERS and the last conv2d with NUM_CLASSES #########
# Simple UNET
def Unet(input_size, num_classes:int, dropout_rates:list):
    
    input_layer = Input(input_size)

    copy1, p1 = contraction_block(input_tensor=input_layer, num_filters=32, dropout_rate=dropout_rates[0])
    copy2, p2 = contraction_block(input_tensor=p1, num_filters=64, dropout_rate=dropout_rates[0])
    copy3, p3 = contraction_block(input_tensor=p2, num_filters=128, dropout_rate=dropout_rates[1])
    copy4, p4 = contraction_block(input_tensor=p3, num_filters=256, dropout_rate=dropout_rates[1])

    x5 = convolution_block(input_tensor=p4, num_filters=512, dropout_rate = dropout_rates[2])

    p6 = expansive_block(copy=copy4, input_tensor=x5, num_filters=256, dropout_rate = dropout_rates[1])
    p7 = expansive_block(copy=copy3, input_tensor=p6, num_filters=128, dropout_rate = dropout_rates[1])
    p8 = expansive_block(copy=copy2, input_tensor=p7, num_filters=64, dropout_rate = dropout_rates[0])
    p9 = expansive_block(copy=copy1, input_tensor=p8, num_filters=32, dropout_rate = dropout_rates[0])

    # Due to mirror-like shape of the UNET architecture, f9 == num_classes
    # since multiclass task use Softmax activation function
    output = Conv2D(filters=num_classes, kernel_size=(1, 1), activation='softmax')(p9)
    
    model = Model(inputs=[input_layer], outputs=[output], name='Unet')
    return model


# Autoencoder
def autoencoder():
    pass


# Attention
def gating_signal():
    pass


# Residual block: https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf
def residual_block():
    pass


# ResUnet with Attention --- could also do only the residual unet (basically unet with residual blocks, not more than that)
def ResUnet_att():
    pass






















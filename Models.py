# python -m Models

# General
import tensorflow as tf
from tensorflow.keras import backend as K

# Network Architecture
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout, BatchNormalization, Activation, Add
from keras.optimizers import Adam

# Metrics
from keras.metrics import MeanIoU

############################################################################################################################################################################
# METRICS
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

############################################################################################################################################################################

# BLOCKS
def contraction_block(input_tensor, num_filters, doBatchNorm = True, drop_rate = None):
    x = Conv2D(num_filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(input_tensor)      # strides default to 1
    if doBatchNorm:
        x = BatchNormalization()(x)             # to make computations more efficient
    x = Activation('relu')(x)
    
    x = Conv2D(num_filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    if doBatchNorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    pooled = MaxPooling2D((2, 2))(x)            # OG paper proposed strides=2 but we leave them default to pool_size = 2
    pooled = Dropout(drop_rate)(pooled)
    return x, pooled



def expansive_block(copy, input_tensor, num_filters, doBatchNorm = True, drop_rate = None):  
    # Conv2dtranspose =! upsampling (both the concept and the keras layer). both increase dim of arrays.
    # upsampling2d is the opposite of pooling repeating rows and columns of input.
    # Conv2dtranspose performs upsampling and then convolution. 
    x = Conv2DTranspose(num_filters, kernel_size = (3, 3), strides = (2, 2), padding='same')(input_tensor)       

    # Concatenation: crop the copy from the specular contraction block and concatenate it to the
    # current respective decoder layer of the expansive path
    x = concatenate([x, copy])

    # add simple 2D convolutions
    x = Conv2D(num_filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)      # strides default to 1
    if doBatchNorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(num_filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    if doBatchNorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(drop_rate)(x)
    return x


# will be useful also for autoencoder
def convolution_block(input_tensor, num_filters, drop_rate = None, doBatchNorm = True):
    x = Conv2D(num_filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(input_tensor)
    if doBatchNorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    if doBatchNorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(drop_rate)(x)
    return x


# UNET
def Unet(input_size, filters = 16, n_classes = None, activation='sigmoid'):
    input_layer = Input(input_size)

    copy1, p1 = contraction_block(input_tensor=input_layer, num_filters = filters*1, drop_rate=0.1, doBatchNorm=True)
    copy2, p2 = contraction_block(input_tensor=p1, num_filters = filters*2, drop_rate=0.1, doBatchNorm=True)
    copy3, p3 = contraction_block(input_tensor=p2, num_filters = filters*4, drop_rate=0.2, doBatchNorm=True)
    copy4, p4 = contraction_block(input_tensor=p3, num_filters = filters*8, drop_rate=0.2, doBatchNorm=True)

    x5 = convolution_block(input_tensor=p4, num_filters=filters*16, drop_rate = 0.2, doBatchNorm=True)

    p6 = expansive_block(copy=copy4, input_tensor=x5, num_filters = filters*8, drop_rate = 0.2, doBatchNorm=True)
    p7 = expansive_block(copy=copy3, input_tensor=p6, num_filters = filters*4, drop_rate = 0.2, doBatchNorm=True)
    p8 = expansive_block(copy=copy2, input_tensor=p7, num_filters = filters*2, drop_rate = 0.1, doBatchNorm=True)
    p9 = expansive_block(copy=copy1, input_tensor=p8, num_filters = filters*1, drop_rate = 0.1, doBatchNorm=True)

    # num_classes should be 3 if working with OG images and masks
    output = Conv2D(filters=n_classes, kernel_size=(1, 1), activation=activation)(p9)
    model = Model(inputs=[input_layer], outputs=[output], name='Unet')
    return model


# Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_name.h5', verbose=1, save_best_only=True)      # or .keras

callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
             tf.keras.callbacks.TensorBoard(log_dir='logs')]
################################################################################################################################

# Autoencoder
# upsample_block for autoencoder
def upsample_block(input_tensor, num_filters, doBatchNorm = True):
    x = Conv2DTranspose(num_filters, kernel_size=(3, 3), strides = (2, 2), padding='same')(input_tensor)
    x = Conv2D(num_filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    if doBatchNorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size=(3, 3), kernel_initializer='he_normal', padding='same')(x)
    if doBatchNorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def encoder(inputs, filters = 16):
    # will not use the copies since there are no skip connections
    _, p1 = contraction_block(input_tensor = inputs, num_filters = filters*1, dropout_rate=0, doBatchNorm=True)
    _, p2 = contraction_block(input_tensor = p1, num_filters = filters*2, dropout_rate=0, doBatchNorm=True)
    _, p3 = contraction_block(input_tensor = p2, num_filters = filters*4, dropout_rate=0, doBatchNorm=True)
    _, p4 = contraction_block(input_tensor = p3, num_filters = filters*8, dropout_rate=0, doBatchNorm=True)

    p5 = convolution_block(input_tensor = p4, num_filters = filters*16, dropout_rate = 0, doBatchNorm=True)
    return p5

def decoder(inputs, filters = 16):
    u1 = upsample_block(input_tensor = inputs, num_filters = filters*8, doBatchNorm=True)
    u2 = upsample_block(input_tensor = u1, num_filters = filters*4, doBatchNorm=True)
    u3 = upsample_block(input_tensor = u2, num_filters = filters*2, doBatchNorm=True)
    u4 = upsample_block(input_tensor = u3, num_filters = filters*1, doBatchNorm=True)
    decoded = Conv2D(3, 3, padding="same", activation="sigmoid")(u4)        # sigmoid since reconstruction
    return decoded

def autoencoder(input_size):
    input_layer = Input(input_size)
    autoencoder = Model(input_layer, decoder(encoder(input_layer)))
    return autoencoder




# Residual block: https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf (fig 4)
# there are 2 variants (2. is the one proposed in OG paper of ResNet)
# 1. conv - BN - Activation - conv - BN - Activation - shortcut  - BN - shortcut+BN
# 2. conv - BN - Activation - conv - BN - shortcut  - BN - addition() - Activation
def residual_contraction_block(input_tensor, num_filters, doBatchNorm = True): 
    x = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(input_tensor)       # default kernel initializer to "glorot_uniform"
    if doBatchNorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(x)
    if doBatchNorm:
        x = BatchNormalization()(x)
    # x = Activation('relu')(x)    #Activation before addition with residual

    # 1x1 convolution on input image
    residual = Conv2D(num_filters, kernel_size=(1, 1), padding='same')(input_tensor)
    if doBatchNorm:
        residual = BatchNormalization()(residual)
    
    # sum convolution block with residual and perform 2x2 max pooling
    res_x = Add()([residual, x])
    res_x = Activation('relu')(res_x)       # act f in the end like the OG paper proposes
    # pooled = MaxPooling2D((2, 2))(res_x)
    # pooled = Dropout(drop_rate)(pooled)
    # return pooled
    return res_x


def ResUnet(input_size, filters = 16, n_classes = None, activation='sigmoid'):
    input_layer = Input(input_size)

    # downsample
    x1 = residual_contraction_block(input_tensor=input_layer, num_filters = filters*1, doBatchNorm = True)    
    p1 = MaxPooling2D((2, 2))(x1)
    p1 = Dropout(rate=0.1)(p1)

    x2 = residual_contraction_block(input_tensor=p1, num_filters = filters*2, doBatchNorm = True)    
    p2 = MaxPooling2D((2, 2))(x2)
    p2 = Dropout(rate=0.1)(p2)

    x3 = residual_contraction_block(input_tensor=p2, num_filters = filters*4, doBatchNorm = True)    
    p3 = MaxPooling2D((2, 2))(x3)
    p3 = Dropout(rate=0.2)(p3)

    x4 = residual_contraction_block(input_tensor=p3, num_filters = filters*8, doBatchNorm = True)    
    p4 = MaxPooling2D((2, 2))(x4)
    p4 = Dropout(rate=0.2)(p4)

    # intermediate
    x5 = residual_contraction_block(input_tensor=p4, num_filters = filters*16, doBatchNorm = True)
    x5 = Dropout(rate=0.2)(x5)

    # upsample
    x6 = Conv2DTranspose(filters = filters*8, kernel_size = (3, 3), strides = (2, 2), padding='same')(x5)
    x7 = concatenate([x6, x4])
    x8 = residual_contraction_block(input_tensor=x7, num_filters = filters*8, doBatchNorm = True)
    x8 = Dropout(rate=0.2)(x8)

    x9 = Conv2DTranspose(filters = filters*4, kernel_size = (3, 3), strides = (2, 2), padding='same')(x8)
    x10 = concatenate([x9, x3])
    x11 = residual_contraction_block(input_tensor=x10, num_filters = filters*4, doBatchNorm = True)
    x11 = Dropout(rate=0.2)(x11)

    x12 = Conv2DTranspose(filters = filters*2, kernel_size = (3, 3), strides = (2, 2), padding='same')(x11)
    x13 = concatenate([x12, x2])
    x14 = residual_contraction_block(input_tensor=x13, num_filters = filters*2, doBatchNorm = True)
    x14 = Dropout(rate=0.1)(x14)

    x15 = Conv2DTranspose(filters = filters*1, kernel_size = (3, 3), strides = (2, 2), padding='same')(x14)
    x16 = concatenate([x15, x1])
    x17 = residual_contraction_block(input_tensor=x16, num_filters = filters*1, doBatchNorm = True)
    x17 = Dropout(rate=0.1)(x17)

    output = Conv2D(filters=n_classes, kernel_size=(1, 1), activation=activation)(x17)
    model = Model(inputs=[input_layer], outputs=[output], name='ResUnet')
    return model    


##########################################################################################################################à

# Attention
def gating_signal():
    pass

def attention_block():
    pass

# ResUnet with Attention
def ResUnet_att():
    pass





















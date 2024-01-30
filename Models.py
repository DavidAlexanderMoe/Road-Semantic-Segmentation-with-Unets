# run this before calling on jupyter 
# python -m Models

# General
import tensorflow as tf
from tensorflow.keras import backend as K
# import segmentation_models as sm

# Modeling
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout
from keras.optimizers import Adam

# Metrics
from keras.metrics import MeanIoU, CategoricalCrossentropy

# Losses
# from segmentation_models.losses import DiceLoss, CategoricalFocalLoss

# Dice coefficient
# really useful in image segmentation tasks with imbalanced data
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


# BLOCKS
def contraction_block(input_tensor, num_filters, dropout_rate = 0, batch_normalization = False):
    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input_tensor)      # strides default to 1
    
    # Batch normalization should be performed over channels after a convolution, 
    # In the following code axis is set to 3 as our inputs are of shape 
    # [None, height, width, channel]. Channel is axis=3.
    if batch_normalization is True:
        x = BatchNormalization(axis=3)(x)
    
    if dropout_rate > 0:
        x = Dropout(rate = dropout_rate)(x)
    
    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    pooled = MaxPooling2D((2, 2))(x)            # OG paper proposed strides=2 but we leave them default to pool_size = 2
    filters = 2 * num_filters
    return x, pooled, filters




def intermediate_block(input_tensor, num_filters, dropout_rate = 0):
    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input_tensor)      # strides default to 1
    
    if dropout_rate > 0:
        x = Dropout(rate = dropout_rate)(x)
    
    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    filters = num_filters
    return x, filters




def expansive_block(copy, input_tensor, num_filters, dropout_rate = 0, batch_normalization = False):
    filters = num_filters / 2
    
    # Conv2dtranspose =! upsampling (both the concept and the keras layer). both increase dim of arrays.
    # upsampling2d is the opposite of pooling repeating rows and columns of input.
    # Conv2dtranspose performs upsampling and of course convolution. 
    x = Conv2DTranspose(filters, (2, 2), strides = 2, padding='same')(input_tensor)       

    # Concatenation: crop the copy from the specular contraction block and concatenate it to the
    # current respective decoder layer of the expansive path
    x = concatenate([x, copy])

    # add simple 2D convolutions
    x = Conv2D(filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    
    if batch_normalization is True:
        x = BatchNormalization(axis=3)(x)

    if dropout_rate > 0:
        x = Dropout(rate = dropout_rate)(x)

    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)

    return x, filters



# Simple UNET
def Unet(input_size, num_classes:int, n_filters, dropout_rates:list, batch_normalization):
    
    input_layer = Input(input_size)

    copy1, p1, f1 = contraction_block(input_tensor=input_layer, num_filters=n_filters, dropout_rate=dropout_rates[0], batch_normalization=batch_normalization)
    copy2, p2, f2 = contraction_block(input_tensor=p1, num_filters=f1, dropout_rate=dropout_rates[0], batch_normalization=batch_normalization)
    copy3, p3, f3 = contraction_block(input_tensor=p2, num_filters=f2, dropout_rate=dropout_rates[0], batch_normalization=batch_normalization)
    copy4, p4, f4 = contraction_block(input_tensor=p3, num_filters=f3, dropout_rate=dropout_rates[0], batch_normalization=batch_normalization)

    x5, f5 = intermediate_block(input_tensor=p4, num_filters=f4, dropout_rate = dropout_rates[1])

    p6, f6 = expansive_block(copy=copy4, input_tensor=x5, num_filters=f5, dropout_rate = dropout_rates[2], batch_normalization = batch_normalization)
    p7, f7 = expansive_block(copy=copy3, input_tensor=p6, num_filters=f6, dropout_rate = dropout_rates[2], batch_normalization = batch_normalization)
    p8, f8 = expansive_block(copy=copy2, input_tensor=p7, num_filters=f7, dropout_rate = dropout_rates[2], batch_normalization = batch_normalization)
    p9, f9 = expansive_block(copy=copy1, input_tensor=p8, num_filters=f8, dropout_rate = dropout_rates[2], batch_normalization = batch_normalization)

    # Due to mirror-like shape of the UNET architecture, f9 == num_classes
    # since multiclass task use Softmax activation function
    output = Conv2D(filters=num_classes, kernel_size=(1, 1), activation='softmax')(p9)
    
    model = Model(inputs=[input_layer], outputs=[output])
    # model.compile(optimizer=Adam, loss=loss, metrics=metrics)
    # model.summary()
    return model




































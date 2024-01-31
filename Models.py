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


############################################################################################################################################################################
# Losses
# from segmentation_models.losses import DiceLoss, CategoricalFocalLoss
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true*K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1-y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss
    
    return focal_loss

# loss=categorical_focal_loss(gamma=2.0, alpha=0.25)

############################################################################################################################################################################
# Dice coefficient
# really useful in image segmentation tasks with imbalanced data
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

############################################################################################################################################################################
# BLOCKS
def contraction_block(input_tensor, num_filters, dropout_rate, batch_normalization):
    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input_tensor)      # strides default to 1
    
    # Batch normalization should be performed over channels after a convolution, 
    # In the following code axis is set to 3 as our inputs are of shape 
    # [None, height, width, channel]. Channel is axis=3.
    if batch_normalization is True:
        x = BatchNormalization(axis=3)(x)

    # Dropout rate in between the Conv layers -> we got better results
    if dropout_rate > 0:
        x = Dropout(rate = dropout_rate)(x)
    
    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)

    pooled = MaxPooling2D((2, 2))(x)            # OG paper proposed strides=2 but we leave them default to pool_size = 2
    return x, pooled




def intermediate_block(input_tensor, num_filters, dropout_rate):
    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input_tensor)      # strides default to 1
    if dropout_rate > 0:
        x = Dropout(rate = dropout_rate)(x)
    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    return x




def expansive_block(copy, input_tensor, num_filters, dropout_rate, batch_normalization):
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
    
    if batch_normalization is True:
        x = BatchNormalization(axis=3)(x)

    if dropout_rate > 0:
        x = Dropout(rate = dropout_rate)(x)

    x = Conv2D(num_filters, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)

    return x



# Simple UNET
def Unet(input_size, num_classes:int, dropout_rates:list, batch_normalization):
    
    input_layer = Input(input_size)

    copy1, p1 = contraction_block(input_tensor=input_layer, num_filters=32, dropout_rate=dropout_rates[0], batch_normalization=batch_normalization)
    copy2, p2 = contraction_block(input_tensor=p1, num_filters=64, dropout_rate=dropout_rates[0], batch_normalization=batch_normalization)
    copy3, p3 = contraction_block(input_tensor=p2, num_filters=128, dropout_rate=dropout_rates[1], batch_normalization=batch_normalization)
    copy4, p4 = contraction_block(input_tensor=p3, num_filters=256, dropout_rate=dropout_rates[1], batch_normalization=batch_normalization)

    x5 = intermediate_block(input_tensor=p4, num_filters=512, dropout_rate = dropout_rates[2])

    p6 = expansive_block(copy=copy4, input_tensor=x5, num_filters=256, dropout_rate = dropout_rates[1], batch_normalization = batch_normalization)
    p7 = expansive_block(copy=copy3, input_tensor=p6, num_filters=128, dropout_rate = dropout_rates[1], batch_normalization = batch_normalization)
    p8 = expansive_block(copy=copy2, input_tensor=p7, num_filters=64, dropout_rate = dropout_rates[0], batch_normalization = batch_normalization)
    p9 = expansive_block(copy=copy1, input_tensor=p8, num_filters=32, dropout_rate = dropout_rates[0], batch_normalization = batch_normalization)

    # Due to mirror-like shape of the UNET architecture, f9 == num_classes
    # since multiclass task use Softmax activation function
    output = Conv2D(filters=num_classes, kernel_size=(1, 1), activation='softmax')(p9)
    
    model = Model(inputs=[input_layer], outputs=[output])
    return model


# Autoencoder

# Attention

# Residual block: https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf

# ResUnet with Attention






















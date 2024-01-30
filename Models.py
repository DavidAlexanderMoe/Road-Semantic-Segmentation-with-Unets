# metrics
# losses
# blocchi convoluzionali semplici (encoder e decoder)
# blocchi residui
# attention (with gating signal)        https://arxiv.org/pdf/1804.03999.pdf
# autoencoder

# Note: Batch normalization should be performed over channels after a convolution, 
# In the following code axis is set to 3 as our inputs are of shape 
# [None, height, width, channel]. Channel is axis=3.
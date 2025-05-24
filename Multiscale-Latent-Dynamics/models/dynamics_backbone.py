import tensorflow as tf
from tensorflow import keras

# local modules
from models import layer


def latent_evolution_unet(spatial_shape, n_channel, n_out_features, n_base_features):
    """Build a U-Net using the Keras Functional API
    
    Args:
        spatial_shape (2-tupe):    (H, W) of the input tensor of shape (H, W, channels)
        n_channels (int):   number of input channels
        n_out_features (int):   number of final output channels
        n_base_features (int):    number of output channels for the first convolutional block

    Return:
        unet (Keras.Model):    U-Net 
    """
    input = keras.Input(shape = (spatial_shape[0], spatial_shape[1], n_channel))
    input = keras.layers.Dropout(rate=0.1)(input) 
    conv1 = layer.conv_block_down(input, 
                                   feat_dim = n_base_features,
                                   reps = 3, 
                                   kernel_size = 3,     
                                   mode='normal')                                         
    conv2 = layer.conv_block_down(conv1, 
                                   feat_dim = n_base_features*2,
                                   reps = 3,     
                                   kernel_size = 3,     
                                   mode = 'down')                                         
    conv3 = layer.conv_block_down(conv2, 
                                   feat_dim = n_base_features*4,
                                   reps = 3,     
                                   kernel_size = 3,     
                                   mode = 'down')                                         
    conv4 = layer.conv_block_up_w_concat(conv3, conv2, 
                                          feat_dim = n_base_features*2,
                                          reps = 3,  
                                          kernel_size = 3, 
                                          mode = 'up')                                          
    conv5 = layer.conv_block_up_w_concat(conv4, conv1, 
                                          feat_dim = n_base_features,
                                          reps = 3,
                                          kernel_size = 3,  
                                          mode = 'up')                                          
    output = layer.conv_block_up_wo_concat(conv5, 
                                            feat_dim = n_out_features,
                                            reps = 3,    
                                            kernel_size = 1,    
                                            mode = 'normal')    
                                                
    unet = keras.Model(input, output)

    return unet

class Dynamics(keras.Model):
    """Backbone model for the autoregressive functionals"""
    def __init__(self, model_name, model_hypers):
        """
        Args: 
            model_name (string):    specifies the architecture of the decoupled and coupled latent evolution mappings
                                    for the dynamics of latent-mean and latent-log-variance fields. 
                                    Currently, only `ResNet` architecture has been implemented.
            model_hypers (list):    list of hyperparameters of the correponding architecture
        """
        super(Dynamics, self).__init__()
        if model_name == 'resnet16':
            self.n_feats_map = model_hypers[0]
            self.backbone = resnet16(self.n_feats_map)
        self.dropout = tf.keras.layers.Dropout(rate=0.1) 

    def call(self, inputs):
        inputs = self.dropout(inputs)
        out = self.backbone(inputs)
        return out
class resnet_block(tf.keras.layers.Layer):
    def __init__(self, filters, n_blocks, stride=1, n_feats_map=None):
        super().__init__()
        self.n_blocks=n_blocks
        self.resnet_blocks = [None] * n_blocks
        for i in range(self.n_blocks - 1):
            self.resnet_blocks[i] = layer.resnet_unit(filters, stride=stride)
        self.resnet_blocks[-1] = layer.resnet_unit(filters, stride=stride, n_feats_map=n_feats_map) if n_feats_map is not None else layer.resnet_unit(filters, stride=stride)
        
    def call(self, inputs):
        x = self.resnet_blocks[0](inputs)
        for i in range(1, self.n_blocks):
            x = self.resnet_blocks[i](x)
        return x
class resnet16(tf.keras.layers.Layer):
    def __init__(self, n_feats_map):
        super().__init__()
        self.block1 = resnet_block(filters=64, n_blocks=1, stride=1)
        self.block2 = resnet_block(filters=128, n_blocks=1, stride=1)
        self.block3 = resnet_block(filters=128, n_blocks=1, stride=1)
        self.block4 = resnet_block(filters=256, n_blocks=1, stride=1, n_feats_map=n_feats_map) 
        self.conv1 = tf.keras.layers.Conv2D(n_feats_map, 1, strides=1, padding='same')
    
    def call(self, inputs):
        x1 = self.block1(inputs)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        out = self.conv1(x4)
        return out


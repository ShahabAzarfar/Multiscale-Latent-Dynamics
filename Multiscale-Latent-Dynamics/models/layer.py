import tensorflow as tf
from tensorflow import keras


class Sampling(tf.keras.layers.Layer):
    """Find a realization of a Gaussain random field"""
    def __init__(self):
        super(Sampling, self).__init__()

    def call(self, inputs):
        """
        Args:
            inputs (list):    list of the form [z_mean, z_log_var], where 
                              z_mean (tf.Tensor):   mean field of the Gaussian random field
                              z_log_var (tf.Tensor):    log-variance field of the Gaussian random field
        Return:
            z (tf.Tensor):    a realization of the Gaussain random field
        """
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.multiply(tf.exp(0.5 * z_log_var), epsilon)
        return z

# CNN-based building blocks of the VGG, U-Net and Res-Net architectures 
# used in the VAE and latent evolution autoregressive functionals, respectively. 

def conv_unit(feat_dim, kernel_size, x):
    x = tf.keras.layers.Conv2D(feat_dim, 
                                kernel_size, 
                                activation = tf.keras.layers.LeakyReLU(0.2), 
                                padding="same")(x)
    x = tf.keras.layers.Conv2D(feat_dim,
                                1,
                                activation = tf.keras.layers.LeakyReLU(0.2),
                                padding="same")(x)
    return x

def conv_block_down(x, feat_dim, reps, kernel_size, mode='normal'):
    if mode == 'down':
        x = tf.keras.layers.MaxPooling2D(2,2)(x)
    for _ in range(reps):
        x = conv_unit(feat_dim, kernel_size, x)                  
    return x

def conv_block_up_w_concat(x, x1, feat_dim, reps, kernel_size, mode='normal'):
    if mode == 'up':
        x = tf.keras.layers.UpSampling2D((2,2),interpolation='bilinear')(x)
    
    x = tf.keras.layers.Concatenate()([x,x1])
    for _ in range(reps):
        x = conv_unit(feat_dim, kernel_size, x)   
    return x

def conv_block_up_wo_concat(x, feat_dim, reps, kernel_size, mode='normal'):
    if mode == 'up':
        x = tf.keras.layers.UpSampling2D((2,2),interpolation='bilinear')(x)
    for _ in range(reps):
        x = conv_unit(feat_dim, kernel_size, x)   
    return x

class resnet_unit(tf.keras.layers.Layer):
    def __init__(self, filters, stride=1, n_feats_map=None): 
        super().__init__()
        if n_feats_map is not None: 
            self.shortcut = tf.keras.layers.Conv2D(n_feats_map, 1, strides=1, padding='same')
            self.c3 = tf.keras.layers.Conv2D(n_feats_map, 1, strides=1, padding='same')
        else: 
            self.shortcut = tf.keras.layers.Conv2D(filters * 4, 1, strides=1, padding='same')
            self.c3 = tf.keras.layers.Conv2D(filters * 4, 1, strides=1, padding='same')
        self.c1 = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same', activation='relu')
        self.c2 = tf.keras.layers.Conv2D(filters, 3, strides=1, padding='same', activation='relu')
        
    def call(self, inputs):
        shortcut = self.shortcut(inputs)
        c1 = self.c1(inputs)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        added = tf.keras.layers.ReLU()(tf.math.add( shortcut, c3 ))
        return added
    

# Complementary layer for the VAE to include the perception loss during its training
class Binary2RGB(tf.keras.layers.Layer):
    def __init__(self):
        super(Binary2RGB, self).__init__()

    def call(self, inputs):
        return tf.image.grayscale_to_rgb(inputs)

def inception(input_shape):
    inputs = keras.Input(shape = input_shape)
    rgb = Binary2RGB()(inputs)
    inception_v3 = tf.keras.applications.inception_v3.InceptionV3(include_top=False,
                                                       weights='imagenet',
                                                       input_shape = (160,256,3),   
                                                       pooling=max)
    feature = inception_v3(rgb)
    inception_model = keras.Model(inputs, feature)
    inception_model.trainable = False

    return inception_model

    
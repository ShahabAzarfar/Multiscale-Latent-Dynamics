import tensorflow as tf
from tensorflow import keras

# local modules
from models import layer

   
def vgg_encoder(input_shape, latent_dims = 4, n_base_features = 64):
    """Build a VGG-based variational encoder using the Keras Functional API
    
    Args:
        input_shape (3-tuple):    shape of input tensor (H, W, channels)
        latent_dims (int):    number of channels of the latent-mean and latent-log-variance fields
        n_base_features (int):    number of output channels of the first convolutional block of the VGG architecture
    """
    inputs = keras.Input(shape = input_shape)
    conv1 = layer.conv_block_down(inputs, 
                                  feat_dim = n_base_features,
                                  reps = 1,      
                                  kernel_size = 3,      
                                  mode = 'normal')                                          
    conv2 = layer.conv_block_down(conv1, 
                                  feat_dim = n_base_features*2,
                                  reps = 1,      
                                  kernel_size = 3,      
                                  mode = 'down')                                          
    conv3 = layer.conv_block_down(conv2, 
                                  feat_dim = n_base_features*4,
                                  reps = 2,      
                                  kernel_size = 3,      
                                  mode = 'down')                                          
    conv4 = layer.conv_block_down(conv3, 
                                  feat_dim = n_base_features*8,
                                  reps = 2,      
                                  kernel_size = 3,      
                                  mode = 'down')      
                                        
    z_mean = tf.keras.layers.Conv2D(latent_dims, 3, padding="same", name="z_mean")(conv4)
    z_log_var = tf.keras.layers.Conv2D(latent_dims, 3, padding="same", name="z_log_var")(conv4)
    z = layer.Sampling()([z_mean, z_log_var])

    encoder = keras.Model(inputs, [z_mean, z_log_var, z])
    return encoder

def vgg_decoder(input_shape, n_base_features = 64):
    """Build a (reversed) VGG-based variational decoder using the Keras Functional API
    
    Args:
        input_shape (3-tuple):    shape of encoded latent field (H, W, channels)
        n_base_features (int):    number of output channels of the first convolutional block of the VGG architecture
    """
    inputs = keras.Input(shape = input_shape)
    conv_in = tf.keras.layers.Conv2D(n_base_features*8, 3, 
                                     activation = tf.keras.layers.LeakyReLU(0.2), padding="same")(inputs)
    conv2 = layer.conv_block_up_wo_concat(conv_in, 
                                          feat_dim = n_base_features*8,
                                          reps = 2,      
                                          kernel_size = 3,      
                                          mode = 'up')                                                  
    conv3 = layer.conv_block_up_wo_concat(conv2, 
                                          feat_dim = n_base_features*4,
                                          reps = 1,      
                                          kernel_size = 3,      
                                          mode = 'up')                                                  
    conv4 = layer.conv_block_up_wo_concat(conv3, 
                                          feat_dim = n_base_features*2,
                                          reps = 1,      
                                          kernel_size = 3,      
                                          mode = 'up')                                                  
    conv5 = layer.conv_block_up_wo_concat(conv4, 
                                          feat_dim = n_base_features,
                                          reps = 1,      
                                          kernel_size = 3,      
                                          mode = 'normal')      
                                                
    conv_out = tf.keras.layers.Conv2D(1, 3, padding="same")(conv5)
    decoder = keras.Model(inputs, conv_out)
    return decoder

class VAE(keras.Model):
    """Variational auto-encoder with VGG-based encoder and decoder architecture"""
    def __init__(self, input_shape, loss_weights, **kwargs):
        """
        Args:
            input_shape (3-tuple):  shape of input tensor (H, W, channels)
            loss_weights (list):    weights for each loss term considered during training
        """
        super().__init__(**kwargs)
        
        self.in_shape = input_shape
        self.loss_weights = loss_weights

        # Initialize the encoder with VGG architecture
        self.encoder = vgg_encoder(input_shape=(self.in_shape[0], self.in_shape[1], 1))
        # Find shape of latent fields based on a sample input
        test_input = tf.random.uniform((5, self.in_shape[0], self.in_shape[1], 1))
        test_latent, _, _ = self.encoder(test_input) 
        self.latent_shape = (test_latent).numpy().shape[1:]
        # Initialize the decoder 
        self.decoder = vgg_decoder(input_shape=self.latent_shape)
        
        # training loss terms
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.edge_boost_loss_tracker = keras.metrics.Mean(name="edge_boost_loss")
        self.perceptual_loss_tracker = keras.metrics.Mean(name="perceptual_loss")
            
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.edge_boost_loss_tracker,
            self.perceptual_loss_tracker
        ]

    def compute_edge_boosting_weight(self, x_in):
        dy, dx = tf.image.image_gradients(x_in)
        G = tf.sqrt(dy**2+dx**2)
        normalized_g = (G - tf.reduce_min(G)) / (tf.reduce_max(G) - tf.reduce_min(G) + 1e-5)
        return normalized_g
        
    def edge_boosting_loss(self, y_pred, gt):
        weight = self.compute_edge_boosting_weight(gt)
        weighted_pred = tf.math.multiply(y_pred, weight)
        weighted_gt = tf.math.multiply(gt, weight)
        edge_boosting_mae = tf.keras.metrics.mean_absolute_error(weighted_gt, weighted_pred)
        edge_boosting_loss = tf.math.reduce_sum(edge_boosting_mae, axis = (0,1,2))
        return edge_boosting_loss
    
    def perceptual_loss(self, y_pred, gt):
        """perceptual loss based on extracted features using Inceptionv3 model"""
        pred_feature = layer.inception(input_shape=(self.in_shape[0], self.in_shape[1], 1))(y_pred)
        gt_feature = layer.inception(input_shape=(self.in_shape[0], self.in_shape[1], 1))(gt)
        return tf.keras.losses.MeanSquaredError(reduction = 'sum')(pred_feature, gt_feature)
    
    def train_step(self, data):
        data = tf.cast(data,dtype = tf.float32)
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.keras.losses.MeanSquaredError(reduction = 'sum')(reconstruction,data)
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = (tf.reduce_sum(kl_loss, axis=(1,2,3)))
            edge_boost_loss = self.edge_boosting_loss(reconstruction,data)
            perceptual_loss = self.perceptual_loss(reconstruction,data)
            total_loss = ((reconstruction_loss * self.loss_weights[0])
                            + (kl_loss * self.loss_weights[1])
                            + (edge_boost_loss * self.loss_weights[3])
                            + (perceptual_loss * self.loss_weights[4]))
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.edge_boost_loss_tracker.update_state(edge_boost_loss)
        self.perceptual_loss_tracker.update_state(perceptual_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "edge_boost_loss": self.edge_boost_loss_tracker.result(),
            "perceptual_loss": self.perceptual_loss_tracker.result(),
        }
    
    def predict(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        field_concat = tf.stack([data, reconstruction], axis=0)  
        latent_concat = tf.stack([z_mean, z_log_var, z], axis=0)
        return field_concat.numpy(), latent_concat.numpy()
    
    
    
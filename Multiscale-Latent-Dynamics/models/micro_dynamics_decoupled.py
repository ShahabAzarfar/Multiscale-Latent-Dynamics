import tensorflow as tf
from tensorflow import keras
import os

# local modules
from models.vae import vgg_encoder, vgg_decoder
from models.dynamics_backbone import latent_evolution_unet, Dynamics
from models.layer import Sampling
from utils.dir_hyper import vae_dir # directory where the pretrained weights for VAEs are saved
from utils.train_utils import state_norm


class MicroDynamicsDecoupled(keras.Model):
    """Learn the encoded decoupled evolution of each physical field over the latent space at micro-scale."""

    def __init__(
            self, 
            ts, 
            fields, 
            output_fields, 
            input_shape, 
            loss_weights, 
            dynamics_model, 
            **kwargs
            ):
        """
        Args:
            ts (int):   number of predicted micro-scale time-steps
            fields (list of strings):   list of the chosen names for the physicals fields involved in the multi-physics
                                        problem. For decoupled model, it is a list of length one.                            
            output_fields (list of 2-tuples):   list of the form [(field_idx, field_name)] specifying the physical fields
                                                whose reconstruction is considerd during training.
            input_shape (3-tuple):  shape of input tensor (H, W, channels)
            loss_weights (list):    weights for each loss term considered during training
            dynamics_model (string):    specifies the architecture considered for latant evolution autoregressive functionals
        """
        super(MicroDynamicsDecoupled, self).__init__(**kwargs)
        
        self.ts = ts
        self.fields = fields
        self.output_fields = output_fields 
        self.in_shape = input_shape
        self.loss_weights = loss_weights
        self.dynamics_model = dynamics_model
    
        # initialize encoder and decoder for each physical field and load the pretrained weights
        self.encoder_list = [None] * len(self.fields)
        for field_idx, field in enumerate(self.fields):
            setattr(self, f'{field}_encoder', vgg_encoder(input_shape=(self.in_shape[0], self.in_shape[1], 1)))
            self.encoder_list[field_idx] = getattr(self, f'{field}_encoder')
            getattr(self, f'{field}_encoder').load_weights(f"{vae_dir[field]['encoder']}")
            getattr(self, f'{field}_encoder').trainable = False
        
        test_input = tf.random.uniform((2, self.in_shape[0], self.in_shape[1], 1))
        test_latent, _, _ = getattr(self, f'{field}_encoder')(test_input) 
        self.latent_shape = (test_latent).numpy().shape[1:]
        for field in self.fields:
            setattr(self, f'{field}_decoder', vgg_decoder(input_shape=self.latent_shape))
            getattr(self, f'{field}_decoder').load_weights(f"{vae_dir[field]['decoder']}")
            getattr(self, f'{field}_decoder').trainable = False 
        
        # initialize latent dynamics model for latent-mean and latent-log-variance field.
        # Currently, only two architectures, i.e. U-Net and Res-Net has been implemented.
        if self.dynamics_model == 'unet':
            self.latent_evolution_mean = latent_evolution_unet((self.latent_shape[0], self.latent_shape[1]), 
                                                                self.latent_shape[2]*len(self.fields), 
                                                                self.latent_shape[2]*len(self.fields), 64)
        
            self.latent_evolution_var = latent_evolution_unet((self.latent_shape[0], self.latent_shape[1]), 
                                                                self.latent_shape[2]*len(self.fields)*2, 
                                                                self.latent_shape[2]*len(self.fields), 64)
        
        elif self.dynamics_model == 'resnet16':
            self.dyn_hypers = [self.latent_shape[2]*len(self.fields)] # [n_feats_map], i.e. number of output channels, as input for resnet16 class

            self.latent_evolution_mean = Dynamics(self.dynamics_model, self.dyn_hypers)
            self.latent_evolution_var = Dynamics(self.dynamics_model, self.dyn_hypers)

        # total training loss 
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

        # VAE reconstruction loss (Uncomment for trainable VAE) 
        # self.recon_loss_tracker = keras.metrics.Mean(name="recon_loss")  
        # VAE Kullback-Leibler loss (Uncomment for trainable VAE) 
        # self.KL_loss_tracker = keras.metrics.Mean(name="KL_loss") 
          
        # consistency_mean loss over latent space
        self.consistency_mean_loss_tracker = keras.metrics.Mean(name="consistency_mean_loss") 
        # consistency_var loss over latent space 
        self.consistency_var_loss_tracker = keras.metrics.Mean(name="consistency_var_loss")  
        # multi_step loss over latent space
        self.multi_step_loss_tracker = keras.metrics.Mean(name="multi_step_loss")  

        # validation losses
        self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")
        self.val_consistency_mean_loss_tracker = keras.metrics.Mean(name="val_consistency_mean_loss")
        self.val_consistency_var_loss_tracker = keras.metrics.Mean(name="val_consistency_var_loss")
        self.val_multi_step_loss_tracker = keras.metrics.Mean(name="val_multi_step_loss") 
    
    def vae_reconstruction(self, x_traj_gt, output_fields):
        """ Reconstruct physical fields of interest by the corresponding VAE 
        
        Args:
            x_traj_gt (tf.Tensor):  snapshot trajectory of evolving ground-truth fields of shape (N, H, W, channels, time-steps)  
            output_fields (list of 2-tuples):   list of the form [(field_idx, field_name)] specifying the physical fields
                                                of interest to be reconstructed

        Return:
            recon_x_traj (tf.Tensor):   reconstructed fields of shape (N, H, W, len(output_fields), time-steps)  
            z_mean_traj (tf.Tensor):    encoded latent-mean fields of shape (N, H, W, 4*len(output_fields), time-steps)  
            z_var_traj (tf.Tensor):    encoded latent-log-variance fields of shape (N, H, W, 4*len(output_fields), time-steps)  
        """
        z_mean_traj = [None] * x_traj_gt.shape[4]
        z_var_traj = [None] * x_traj_gt.shape[4]
        recon_x_traj = [None] * x_traj_gt.shape[4]

        for t in range(x_traj_gt.shape[4]):
            z_mean = [None] * len(self.fields)
            z_var = [None] * len(self.fields)
            recon_x = [None] * len(output_fields)
            i = 0
            for field_idx, field in enumerate(self.fields):
                latent_field_mean, latent_field_var, latent_field = getattr(self, f'{field}_encoder')(x_traj_gt[..., field_idx : field_idx + 1, t])
                z_mean[field_idx] = latent_field_mean
                z_var[field_idx] = latent_field_var
                for (_, f) in output_fields:
                    if field == f:
                        recon_field = getattr(self, f'{field}_decoder')(latent_field)
                        recon_x[i] = recon_field
                        i += 1
            
            z_mean = tf.concat(z_mean, axis=3)
            z_var = tf.concat(z_var, axis=3)
            recon_x = tf.concat(recon_x, axis=3)
            
            z_mean_traj[t] = z_mean
            z_var_traj[t] = z_var
            recon_x_traj[t] = recon_x
        
        z_mean_traj = tf.stack(z_mean_traj, axis=4) 
        z_var_traj = tf.stack(z_var_traj, axis=4) 
        recon_x_traj = tf.stack(recon_x_traj, axis=4) 
        
        return recon_x_traj, z_mean_traj, z_var_traj   

    def call(self, input):
        """
        Return the predicted evolution of latent fields and the corresponding physical fields in input space

        Args:
            input (list):   list of the form [init_x_gt, evol_step, output_fields], where
                                - init_x_gt (tf.Tensor):  GT-initial value for the corresponding field of shape (N, H, W, channels, 1)
                                - evol_step (int):    number of predicted micro-scale time-steps
                                - output_fields (list of 2-tuples):   list of the form [(field_idx, field_name)] specifying the physical fields
                                                                        of interest to be reconstructed
        
        Return: 
            x_evol_traj (tf.Tensor):    predicted fields evolution of shape (N, H, W, len(output_fields), time-steps) 
            z_mean_traj (tf.Tensor):    evolved latent-mean fields of shape (N, H, W, 4*len(output_fields), time-steps) 
            z_var_traj (tf.Tensor):    evolved latent-log-variance fields of shape (N, H, W, 4*len(output_fields), time-steps) 
        """
        init_x_gt, evol_step, output_fields = input[0], input[1], input[2] 
        z_mean_traj = [None] * (evol_step + 1)
        z_var_traj = [None] * (evol_step + 1)
        
        _ , init_z_mean, init_z_var = self.vae_reconstruction(init_x_gt, list(enumerate(self.fields)))
        current_z_mean, current_z_var = init_z_mean[..., 0], init_z_var[..., 0]
        z_mean_traj[0] = current_z_mean
        z_var_traj[0] = current_z_var

        # latent field evolution
        for t in range(1, evol_step + 1):
            evol_z_mean = self.latent_evolution_mean(current_z_mean)
            evol_z_var = self.latent_evolution_var(tf.keras.layers.Concatenate()([current_z_mean, current_z_var]))
            current_z_mean, current_z_var = evol_z_mean, evol_z_var                                                                           
            z_mean_traj[t] = current_z_mean
            z_var_traj[t] = current_z_var
        z_mean_traj = tf.stack(z_mean_traj, axis=4)
        z_var_traj = tf.stack(z_var_traj, axis=4)
        
        # decoding the evolved latent fields
        x_evol_traj = [None] * (evol_step + 1)
        for t in range(evol_step + 1):
            z_mean = z_mean_traj[..., t]
            z_var = z_var_traj[..., t]
            latent_field = Sampling()([z_mean, z_var])
            x_evol = getattr(self, f'{self.fields[0]}_decoder')(latent_field)
            x_evol_traj[t] = x_evol
        x_evol_traj = tf.stack(x_evol_traj, axis=4)
        
        return x_evol_traj, z_mean_traj, z_var_traj

    def KL_loss(self, z_mean, z_var):
        """ Compute the aggregated Kullback-Leibler divergence between a random Gaussian field with (z_mean, z_var) and the standard random Gaussian field
        
        Args:
            z_mean (tf.Tensor):    latent-mean field of shape (N, H, W, 4)
            z_var (tf.Tensor):    latent-log-variance field of shape (N, H, W, 4)

        Return:
            kl (tf.Tensor):    aggregated Kullback-Leibler divergence   
        """
        kl = tf.reduce_mean(-0.5 * (1 + z_var - tf.square(z_mean) - tf.exp(z_var)), axis=(1,2,3))

        return kl 

    def train_step(self, data):
        """
        Args:
            data (list):    list of the form [init_x_gt, x_traj_gt], where
                                - init_x_gt (tf.Tensor):  GT-initial value for the corresponding physical fields of shape (N, H, W, channels, 1)
                                - x_traj_gt (tf.Tensor):    GT trajectory of the corresponding physical field of shape (N, H, W, channels, time-steps)
        """
        init_x_gt, x_traj_gt = data
        # all the training loss terms
        total_loss = 0
        recon_loss = 0       
        kl_loss = 0      
        consistency_mean_loss = 0
        consistency_var_loss = 0
        multi_step_loss = 0
        criterion = tf.keras.losses.MeanSquaredError()
        
        with tf.GradientTape() as tape:
            # init_x_recon , init_z_mean, init_z_var = self.vae_reconstruction(init_x_gt, self.output_fields)
            # vae_x_traj, vae_z_mean_traj, vae_z_var_traj = self.vae_reconstruction(x_traj_gt, self.output_fields)
            # x_evol_traj, z_mean_traj, z_var_traj = self.call([init_x_gt, self.ts, self.output_fields])
            ###
            # The full set of variable names assigned to the function outputs, which are used in different loss terms,
            # are given in the above. Currently, the loss terms correpsonding to trainable VAE are commented.
              
            # init_x_recon, init_z_mean, init_z_var = self.vae_reconstruction(init_x_gt, self.output_fields)
            _ , vae_z_mean_traj, vae_z_var_traj = self.vae_reconstruction(x_traj_gt, self.output_fields)
            x_evol_traj, z_mean_traj, z_var_traj = self.call([init_x_gt, self.ts, self.output_fields])
            
            for field_idx, field in enumerate(self.fields):
                c0, c1 = (field_idx * self.latent_shape[2]), (( field_idx + 1) * self.latent_shape[2]) 
                # uncomment in the case of trainable VAE
                # recon_loss += criterion(init_x_gt[..., 0], init_x_recon[..., 0])
                # kl_loss += self.KL_loss(init_z_mean[..., c0:c1, 0], init_z_var[..., c0:c1, 0])
                                        
                for t in range(self.ts):
                    # recon_loss += criterion(x_traj_gt[..., t], vae_x_traj[..., t])
                    # kl_loss += self.KL_loss(vae_z_mean_traj[..., c0:c1, t], vae_z_var_traj[..., c0:c1, t])
                  
                    multi_step_loss += criterion(x_traj_gt[..., t], x_evol_traj[..., t+1])
                    
                    epsilon = 1e-6
                    mean_norm = state_norm(vae_z_mean_traj[..., c0:c1, t]) + epsilon
                    consistency_mean_loss += criterion(tf.divide(z_mean_traj[..., c0:c1, t+1], mean_norm), 
                                                       tf.divide(vae_z_mean_traj[..., c0:c1, t], mean_norm))
                                                        
                    var_norm = state_norm(vae_z_var_traj[..., c0:c1, t]) + epsilon
                    consistency_var_loss += criterion(tf.divide(z_var_traj[..., c0:c1, t+1], var_norm), 
                                                      tf.divide(vae_z_var_traj[..., c0:c1, t], var_norm))
                                                 
            # recon_loss = tf.cast(recon_loss, tf.float32)
            # kl_loss = tf.cast(kl_loss, tf.float32)
            consistency_mean_loss = tf.cast(consistency_mean_loss, tf.float32)
            consistency_var_loss = tf.cast(consistency_var_loss, tf.float32)
            multi_step_loss = tf.cast(multi_step_loss, tf.float32)
            ###
            total_loss = ((self.loss_weights[0] * recon_loss)
                        + (self.loss_weights[1] * kl_loss)
                        + (self.loss_weights[2] * consistency_mean_loss)
                        + (self.loss_weights[3] * consistency_var_loss) 
                        + (self.loss_weights[4] * multi_step_loss)) 
        ###    
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))    

        self.total_loss_tracker.update_state(total_loss)
        # self.recon_loss_tracker.update_state(recon_loss)
        # self.KL_loss_tracker.update_state(kl_loss)
        self.consistency_mean_loss_tracker.update_state(consistency_mean_loss)
        self.consistency_var_loss_tracker.update_state(consistency_var_loss)
        self.multi_step_loss_tracker.update_state(multi_step_loss)        
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            # "recon_loss": self.recon_loss_tracker.result(),
            # "KL_loss": self.KL_loss_tracker.result(),
            "consistency_mean_loss": self.consistency_mean_loss_tracker.result(),
            "consistency_var_loss": self.consistency_var_loss_tracker.result(),
            "multi_step_loss": self.multi_step_loss_tracker.result()
        }

    def test_step(self, data):
        init_x_gt, x_traj_gt = data
        # validation loss terms
        consistency_mean_loss = 0
        consistency_var_loss = 0
        multi_step_loss = 0
        criterion = tf.keras.losses.MeanSquaredError()

        _ , vae_z_mean_traj, vae_z_var_traj = self.vae_reconstruction(x_traj_gt, self.output_fields)
        x_evol_traj, z_mean_traj, z_var_traj = self.call([init_x_gt, self.ts, self.output_fields])
        
        for field_idx, field in enumerate(self.fields):
            c0, c1 = (field_idx * self.latent_shape[2]), (( field_idx + 1) * self.latent_shape[2]) 
            for t in range(self.ts):
                multi_step_loss += criterion(x_traj_gt[..., t], x_evol_traj[..., t+1])
                #
                epsilon = 1e-6
                mean_norm = state_norm(vae_z_mean_traj[..., c0:c1, t]) + epsilon
                consistency_mean_loss += criterion(tf.divide(z_mean_traj[..., c0:c1, t+1], mean_norm), 
                                                    tf.divide(vae_z_mean_traj[..., c0:c1, t], mean_norm))
                                                    
                var_norm = state_norm(vae_z_var_traj[..., c0:c1, t]) + epsilon
                consistency_var_loss += criterion(tf.divide(z_var_traj[..., c0:c1, t+1], var_norm), 
                                                    tf.divide(vae_z_var_traj[..., c0:c1, t], var_norm))
                                                
                                                
        
        consistency_mean_loss = tf.cast(consistency_mean_loss, tf.float32)
        consistency_var_loss = tf.cast(consistency_var_loss, tf.float32)
        multi_step_loss = tf.cast(multi_step_loss, tf.float32)
        
        total_loss = ((self.loss_weights[2] * consistency_mean_loss)
                    + (self.loss_weights[3] * consistency_var_loss) 
                    + (self.loss_weights[4] * multi_step_loss)) 
           
        self.val_total_loss_tracker.update_state(total_loss)
        self.val_consistency_mean_loss_tracker.update_state(consistency_mean_loss)
        self.val_consistency_var_loss_tracker.update_state(consistency_var_loss)
        self.val_multi_step_loss_tracker.update_state(multi_step_loss)        
        
        return {
            "total_loss": self.val_total_loss_tracker.result(),
            "consistency_mean_loss": self.val_consistency_mean_loss_tracker.result(),
            "consistency_var_loss": self.val_consistency_var_loss_tracker.result(),
            "multi_step_loss": self.val_multi_step_loss_tracker.result()
        }            

    def save_weight(self, ts, n_epochs, weight_dir):
        """Save weights of latent autoregressive functionals
        
        Args:
            ts (int):   number of predicted time-steps
            n_epochs (int):    number of training epochs corresponding to the weights
            weight_dir (string):    directory for saving weights
        """
        if not os.path.isdir(weight_dir):
            os.makedirs(weight_dir)
        self.latent_evolution_mean.save_weights(f'{weight_dir}/latent_evolution_mean_{ts}_ts_{n_epochs}_epoch.h5')
        self.latent_evolution_var.save_weights(f'{weight_dir}/latent_evolution_var_{ts}_ts_{n_epochs}_epoch.h5')
    
    def load_weight(self, ts, n_epochs, weight_dir):
        """Load weights of latent autoregressive functionals

        Args:
            ts (int):   number of predicted time-steps
            n_epochs (int):    number of training epochs corresponding to the weights
            weight_dir (string):    directory for loading weights
        """
        self.latent_evolution_mean.load_weights(f'{weight_dir}/latent_evolution_mean_{ts}_ts_{n_epochs}_epoch.h5')
        self.latent_evolution_var.load_weights(f'{weight_dir}/latent_evolution_var_{ts}_ts_{n_epochs}_epoch.h5')
    

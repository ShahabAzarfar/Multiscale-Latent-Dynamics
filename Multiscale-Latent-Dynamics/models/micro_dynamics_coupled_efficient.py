import tensorflow as tf
from tensorflow import keras
import os

# local modules
from models.vae import vgg_decoder
from models.dynamics_backbone import latent_evolution_unet, Dynamics
from models.layer import Sampling
from utils.train_utils import state_norm
from utils.dir_hyper import vae_dir, dyn_dir # directories where the pretrained weights for VAEs and decoupled dynamics are saved, respectively.



class MicroDynamicsCoupledEfficient(keras.Model):
    """Learn the vae-encoded coupled evolution of all physical fields together over the latent space at micro-scale.
    
    It has the same architecture as `MicroDynamicsCoupled` class, while the memory efficiency is improved by removing
    the encoder modules and taking the latent-mean and latent-log-variance fields directly as input.
    """

    def __init__(self, ts, fields, output_fields, latent_shape, loss_weights, dynamics_model):
        """
        Args:
            ts (int):   number of predicted micro-scale time-steps
            fields (list of strings):   list of the chosen names for the physicals fields involved in the multi-physics problem.                                                       
            output_fields (list of 2-tuples):   list of the form [(field_idx, field_name)] specifying the physical fields
                                                whose reconstruction is considered during training.
            latent_shape (3-tuple):  shape of latent field tensor (H, W, channels)
            loss_weights (list):    weights for each loss term considered during training
            dynamics_model (string):    specifies the architecture considered for latent evolution autoregressive functionals.
                                        Currently, it only accepts 'unet' or 'resnet16' corresponding to the 
                                        implemented U-Net and ResNet architectures.
        """
        super(MicroDynamicsCoupledEfficient, self).__init__()

        self.ts = ts
        self.fields = fields
        self.output_fields = output_fields 
        self.latent_shape = latent_shape  # (20, 32, 4) for single-void data
        self.loss_weights = loss_weights
        self.dynamics_model = dynamics_model
        
        # initialize latent dynamics autoregressive functionals
        if self.dynamics_model == 'unet':
            # decoupled dynamics models
            for field in self.fields: 
                setattr(self, f'{field}_decoupled_evol_mean', 
                        latent_evolution_unet((self.latent_shape[0], self.latent_shape[1]), 
                                                self.latent_shape[2], self.latent_shape[2], 64))

                setattr(self, f'{field}_decoupled_evol_var', 
                        latent_evolution_unet((self.latent_shape[0], self.latent_shape[1]), 
                                                            self.latent_shape[2]*2, self.latent_shape[2], 64))
                                               
            # coupled dynamics model    
            self.latent_evolution_mean = latent_evolution_unet((self.latent_shape[0], self.latent_shape[1]), 
                                                         self.latent_shape[2]*len(self.fields), 
                                                         self.latent_shape[2]*len(self.fields), 64)
        
            self.latent_evolution_var = latent_evolution_unet((self.latent_shape[0], self.latent_shape[1]), 
                                                                self.latent_shape[2]*len(self.fields)*2, 
                                                                self.latent_shape[2]*len(self.fields), 64)

        elif self.dynamics_model == 'resnet16':
            # decoupled dynamics models
            self.decouple_dyn_hypers = [self.latent_shape[2]] # [n_feats_map]
            for field in self.fields: 
                setattr(self, f'{field}_decoupled_evol_mean', Dynamics(self.dynamics_model, self.decouple_dyn_hypers))
                setattr(self, f'{field}_decoupled_evol_var', Dynamics(self.dynamics_model, self.decouple_dyn_hypers))
                
            # coupled dynamics model    
            self.dyn_hypers = [self.latent_shape[2]*len(self.fields)] # [n_feats_map]
            self.latent_evolution_mean = Dynamics(self.dynamics_model, self.dyn_hypers)
            self.latent_evolution_var = Dynamics(self.dynamics_model, self.dyn_hypers)

        else:
            raise ValueError("Currently, only the U-Net and ResNet-16 architectures are implemented.")
            
        # decoder corresponding to the output field           
        for field_idx, field in self.output_fields:
            setattr(self, f'{field}_decoder', vgg_decoder(input_shape=self.latent_shape))
            getattr(self, f'{field}_decoder').load_weights(f"{vae_dir[field]['decoder']}")
            getattr(self, f'{field}_decoder').trainable = False 
 
        # training loss terms
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.consistency_mean_loss_tracker = keras.metrics.Mean(name="consistency_mean_loss")
        self.consistency_var_loss_tracker = keras.metrics.Mean(name="consistency_var_loss")
        self.multi_step_loss_tracker = keras.metrics.Mean(name="multi_step_loss")
        # validation loss terms
        self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")
        self.val_consistency_mean_loss_tracker = keras.metrics.Mean(name="val_consistency_mean_loss")
        self.val_consistency_var_loss_tracker = keras.metrics.Mean(name="val_consistency_var_loss")
        self.val_multi_step_loss_tracker = keras.metrics.Mean(name="val_multi_step_loss")    
        
    def decoupled_evolution(self, z_mean, z_var):
        """Apply the decoupled dynamics autoregressive functional to the latent-mean and latent-log-vaiance fields
        
        Args:
            z_mean (tf.Tensor):    latent-mean field of shape (N, H, W, channels)
            z_var (tf.Tensor):    latent-log-variance field of shape (N, H, W, channels)

        Return:
            evol_z_mean (tf.Tensor):    evolved latent-mean field of shape (N, H, W, channels)
            evol_z_var (tf.Tensor):    evolved latent-log-variance field of shape (N, H, W, channels)
        """
        evol_z_mean = [None] * len(self.fields)
        evol_z_var = [None] * len(self.fields)

        for field_idx, field in enumerate(self.fields):
            c0, c1 = (field_idx * self.latent_shape[2]), (( field_idx + 1) * self.latent_shape[2]) 
            evol_latent_field_mean = getattr(self, f'{field}_decoupled_evol_mean')(z_mean[..., c0:c1])    
            evol_latent_field_var = getattr(self, f'{field}_decoupled_evol_var')(tf.keras.layers.Concatenate()([z_mean[..., c0:c1],
                                                                                                                z_var[..., c0:c1]]))
            evol_z_mean[field_idx] = evol_latent_field_mean
            evol_z_var[field_idx] = evol_latent_field_var    
        
        evol_z_mean = tf.concat(evol_z_mean, axis=3)
        evol_z_var = tf.concat(evol_z_var, axis=3)

        return evol_z_mean, evol_z_var

    def latent_evolution(self, in_z_mean, in_z_var):
        """Apply the full autoregressive functional to the latent-mean and latent-log-vaiance fields for one micro-scale time-step.
        
        Args:
            in_z_mean (tf.Tensor):    input latent-mean field of shape (N, H, W, channels)
            in_z_var (tf.Tensor):    input latent-log-variance field of shape (N, H, W, channels)

        Return:
            evol_z_mean (tf.Tensor):    evolved latent-mean field of shape (N, H, W, channels)
            evol_z_var (tf.Tensor):    evolved latent-log-variance field of shape (N, H, W, channels)
        """
        in_z_mean, in_z_var = self.decoupled_evolution(in_z_mean, in_z_var)
        evol_z_mean = self.latent_evolution_mean(in_z_mean)
        evol_z_var = self.latent_evolution_var(tf.keras.layers.Concatenate()([in_z_mean, in_z_var])) 

        return evol_z_mean, evol_z_var
    
    def call(self, input):
        """
        Return the predicted evolution of latent fields and the corresponding physical fields in input space

        Args:
            input (list):   list of the form [init_z_mean, init_z_var, evol_step, output_fields], where
                                - init_z_mean (tf.Tensor):  encoded-initial value for all the latent-mean fields of shape (N, H, W, channels, 1)
                                - init_z_var (tf.Tensor):  encoded-initial value for all the latent-log-variance fields of shape (N, H, W, channels, 1)
                                - evol_step (int):    number of predicted micro-scale time-steps
                                - output_fields (list of 2-tuples):   list of the form [(field_idx, field_name)] specifying the physical fields
                                                                      of interest to be reconstructed
        
        Return: 
            x_evol_traj (tf.Tensor):    predicted fields evolution of shape (N, H, W, len(output_fields), time-steps) 
            z_mean_traj (tf.Tensor):    evolved latent-mean fields of shape (N, H, W, 4*len(output_fields), time-steps) 
            z_var_traj (tf.Tensor):    evolved latent-log-variance fields of shape (N, H, W, 4*len(output_fields), time-steps) 
        """
        init_z_mean, init_z_var, evol_step, output_fields = input[0], input[1], input[2], input[3] 
        z_mean_traj = [None] * (evol_step + 1)
        z_var_traj = [None] * (evol_step + 1)
        
        z_mean, z_var = init_z_mean[..., 0], init_z_var[..., 0]
        z_mean_traj[0] = z_mean
        z_var_traj[0] = z_var

        # latent field evolution
        for t in range(1, evol_step + 1):
            z_mean, z_var = self.latent_evolution(z_mean, z_var)                                                                           
            z_mean_traj[t] = z_mean
            z_var_traj[t] = z_var
        z_mean_traj = tf.stack(z_mean_traj, axis=4)
        z_var_traj = tf.stack(z_var_traj, axis=4)
        
        # decoding the evolved latent fields
        x_evol_traj = [None] * (evol_step + 1)
        for t in range(evol_step + 1):
            x_evol = [None] * len(output_fields)
            for i, f in enumerate(output_fields):
                field_idx, field = f
                c0, c1 = (field_idx * self.latent_shape[2]), (( field_idx + 1) * self.latent_shape[2]) 
                z_mean = z_mean_traj[..., c0:c1, t]
                z_var = z_var_traj[..., c0:c1, t]
                latent_field = Sampling()([z_mean, z_var])
                field_evol = getattr(self, f'{field}_decoder')(latent_field)
                x_evol[i] = field_evol
            x_evol = tf.concat(x_evol, axis=3)
            x_evol_traj[t] = x_evol
        x_evol_traj = tf.stack(x_evol_traj, axis=4)

        return x_evol_traj, z_mean_traj, z_var_traj
    
    def train_step(self, data):
        """
        Args:
            data (list):    list of the form [init_z_mean, vae_z_mean_traj, init_z_var, vae_z_var_traj, x_traj_gt], where
                                - init_z_mean (tf.Tensor):  vae-encoded initial value for all the latent-mean fields of shape (N, H, W, channels, 1)
                                - vae_z_mean_traj (tf.Tensor):    vae-encoded latent-mean fields trajectory of shape (N, H, W, channels, time-steps)
                                - init_z_var (tf.Tensor):  vae-encoded initial value for all the latent-log-variance fields of shape (N, H, W, channels, 1)
                                - vae_z_var_traj (tf.Tensor):    vae-encoded latent-log-variance fields trajectory of shape (N, H, W, channels, time-steps)
                                - x_traj_gt (tf.Tensor):    GT physical fields trajectory of shape (N, H, W, channels, time-steps)
        """
        init_z_mean, vae_z_mean_traj, init_z_var, vae_z_var_traj, x_traj_gt = data
        # all the training loss terms
        total_loss = 0
        consistency_mean_loss = 0
        consistency_var_loss = 0
        multi_step_loss = 0
        criterion = tf.keras.losses.MeanSquaredError()
               
        with tf.GradientTape() as tape:
            x_evol_traj, z_mean_traj, z_var_traj = self.call([init_z_mean, init_z_var, self.ts, self.output_fields]) 
            # compute multi_step_loss
            for i, _ in enumerate(self.output_fields):
                for t in range(self.ts):
                    multi_step_loss += criterion(x_traj_gt[..., i:i+1, t], x_evol_traj[..., i:i+1, t+1])
            # compute consistency_mean_loss and consistency_var_loss        
            for field_idx, field in enumerate(self.fields):
                c0, c1 = (field_idx * self.latent_shape[2]), (( field_idx + 1) * self.latent_shape[2]) 
                for t in range(self.ts):
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
            
            total_loss = ((self.loss_weights[0] * consistency_mean_loss)
                        + (self.loss_weights[1] * consistency_var_loss) 
                        + (self.loss_weights[2] * multi_step_loss)) 
           
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))    

        self.total_loss_tracker.update_state(total_loss)
        self.consistency_mean_loss_tracker.update_state(consistency_mean_loss)
        self.consistency_var_loss_tracker.update_state(consistency_var_loss)
        self.multi_step_loss_tracker.update_state(multi_step_loss)        
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "consistency_mean_loss": self.consistency_mean_loss_tracker.result(),
            "consistency_var_loss": self.consistency_var_loss_tracker.result(),
            "multi_step_loss": self.multi_step_loss_tracker.result()
        }
    
    def test_step(self, data):
        init_z_mean, vae_z_mean_traj, init_z_var, vae_z_var_traj, x_traj_gt = data
        # validation loss terms
        consistency_mean_loss = 0
        consistency_var_loss = 0
        multi_step_loss = 0
        criterion = tf.keras.losses.MeanSquaredError()
                
        x_evol_traj, z_mean_traj, z_var_traj = self.call([init_z_mean, init_z_var, self.ts, self.output_fields]) 
        
        for i, f in enumerate(self.output_fields):
            field_idx, field = f 
            c0, c1 = field_idx, (field_idx + 1)
            for t in range(self.ts):
                multi_step_loss += criterion(x_traj_gt[..., c0:c1, t], x_evol_traj[..., i:i+1, t+1])
        
        for field_idx, field in enumerate(self.fields):
            c0, c1 = (field_idx * self.latent_shape[2]), (( field_idx + 1) * self.latent_shape[2]) 
            for t in range(self.ts):
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
        
        total_loss = ((self.loss_weights[0] * consistency_mean_loss)
                        + (self.loss_weights[1] * consistency_var_loss) 
                        + (self.loss_weights[2] * multi_step_loss))  
           
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

    def load_micro_decoupled_evol(self, pretrain_ts, n_epochs):
        """Load the pretrained weights for the autoregressive functionals of the decoupled micro-dynamics and freeze them

        Args:
            pretrain_ts (list):    list of ts_idx corresponding to each saved decoupled model
            n_epochs (list):   list of n_epochs corresponding to each saved decoupled model 
        """
        for field_idx, field in enumerate(self.fields):
            dir  = dyn_dir(self.dynamics_model)['micro_dyn']['decoupled'][field]
            dir = os.path.join(dir, f'weights_{pretrain_ts[field_idx]}_ts')
            path_mean = os.path.join(dir, f"latent_evolution_mean_{pretrain_ts[field_idx]}_ts_{n_epochs[field_idx]}_epoch.h5") 
            path_var = os.path.join(dir, f"latent_evolution_var_{pretrain_ts[field_idx]}_ts_{n_epochs[field_idx]}_epoch.h5")
            
            getattr(self, f'{field}_decoupled_evol_mean').load_weights(path_mean)
            getattr(self, f'{field}_decoupled_evol_mean').trainable = False
            
            getattr(self, f'{field}_decoupled_evol_var').load_weights(path_var)
            getattr(self, f'{field}_decoupled_evol_var').trainable = False

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
    ###
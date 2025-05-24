import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

def in_out_latent_dataset(z_mean_traj, z_var_traj, x_traj, sample_range, n_ts, batch_size, field_indices=[0]):
    """
    Return an input-output batched tf.dataset of latent fields for corriculum training
    of `MicroDynamicsCoupledEfficient` and `MesoDynamicsCoupledEfficient` models

    Args:
        z_mean_traj (np.array):    latent-mean fields trajectory of shape (N, H, W, channels, time-steps)  
        z_var_traj (np.array):    latent-log-variance fields trajectory of shape (N, H, W, channels, time-steps)  
        x_traj (np.array):    physical fields trajectory of shape (N, H, W, channels, time-steps)                                    
        sample_range (list):    the upper and lower bounds on sample sequnce
        n_ts (int):    the number of time-steps in output sequence 
        batch_size (int):   batch size
        field_indices (list):    list of `field_idx` of the output fields of interest considered during training
    
    Return:
        in_out_data (tf.dataset)
    """
    sweep = x_traj.shape[4] - n_ts
    init_z_mean = [None] *  sweep
    z_mean_out = [None] *  sweep
    init_z_var = [None] *  sweep
    z_var_out = [None] *  sweep
    x_gt_out = [None] *  sweep
    
    for t in range (sweep):
        init_z_mean[t] = z_mean_traj[sample_range[0]:sample_range[1], :,:,:, t:t+1]
        z_mean_out[t] = z_mean_traj[sample_range[0]:sample_range[1], :,:,:, t+1:t+1+n_ts]
        init_z_var[t] = z_var_traj[sample_range[0]:sample_range[1], :,:,:, t:t+1]
        z_var_out[t] = z_var_traj[sample_range[0]:sample_range[1], :,:,:, t+1:t+1+n_ts]
        x_gt_out[t] = np.concatenate([x_traj[sample_range[0]:sample_range[1], :,:, f:f+1, t+1:t+1+n_ts] for f in field_indices], axis=3)

    init_z_mean = (np.concatenate(init_z_mean, axis=0)).astype('float32')
    z_mean_out = (np.concatenate(z_mean_out, axis=0)).astype('float32')
    init_z_var = (np.concatenate(init_z_var, axis=0)).astype('float32')
    z_var_out = (np.concatenate(z_var_out, axis=0)).astype('float32')
    x_gt_out = (np.concatenate(x_gt_out, axis=0)).astype('float32')
    
    shuffle_buffer_size = init_z_mean.shape[0]
    in_out_data = tf.data.Dataset.from_tensor_slices((init_z_mean, z_mean_out, init_z_var, z_var_out, x_gt_out))
    in_out_data = in_out_data.shuffle(shuffle_buffer_size).batch(batch_size)
    
    return in_out_data

def in_out_dataset(x_traj, sample_range, n_ts, batch_size, decoupled=False, field_idx=0):
    """
    Return an input-output batched tf.dataset of physical fields for corriculum training
    of `MicroDynamicsDecoupled` and `MicroDynamicsCoupled` models

    Args:
        x_traj (np.array):    physical fields trajectory of shape (N, H, W, channels, time-steps)                                    
        sample_range (list):    the upper and lower bounds on sample sequnce
        n_ts (int):    the number of time-steps in output sequence 
        batch_size (int):   batch size
        decoupled (bool):    specifies whether the data is prepared for the decoupled model 
                             associated with a single field
        field_idx (int):    field_idx of the output field of interest in case decoupled=True
            
    Return:
        in_out_data (tf.dataset)
    """
    data_in = [] 
    data_out = []
    for t in range (x_traj.shape[4] - n_ts):
        if decoupled:
            data_in.append(x_traj[sample_range[0]:sample_range[1], :,:, field_idx:field_idx+1, t:t+1])
            data_out.append(x_traj[sample_range[0]:sample_range[1], :,:, field_idx:field_idx+1, t+1:t+1+n_ts])
        else:
            data_in.append(x_traj[sample_range[0]:sample_range[1], :,:,:, t:t+1])
            data_out.append(x_traj[sample_range[0]:sample_range[1], :,:,:, t+1:t+1+n_ts])
    data_in = np.concatenate(data_in, axis=0)
    data_out = np.concatenate(data_out, axis=0)
    data_in = data_in.astype('float32')
    data_out = data_out.astype('float32')
    shuffle_buffer_size = data_in.shape[0]
    
    in_out_data = tf.data.Dataset.from_tensor_slices((data_in, data_out))
    in_out_data = in_out_data.shuffle(shuffle_buffer_size).batch(batch_size)
    
    return in_out_data


class CheckPoint(keras.callbacks.Callback):
    """Keras Callback to save checkpoints periodically""" 

    def __init__(self, weight_dir, ckpt_period, spatial_scale):
        """
        Args:
            weight_dir (str):     directory to save checkpoints
            ckpt_period (int):    period to save checkpoints
            spatial_scale (string):     either `micro` or `meso`  
        """
        super().__init__()
        self.weight_dir = weight_dir
        self.ckpt_period = ckpt_period
        self.spatial_scale = spatial_scale

    def on_epoch_end(self, epoch, logs=None):
        if ((epoch+1) % self.ckpt_period) == 0:
            if self.spatial_scale == 'micro':
                    self.model.save_weight(self.model.ts, epoch+1, self.weight_dir)
            elif self.spatial_scale == 'meso':
                    self.model.save_weight(self.model.meso_ts, epoch+1, self.weight_dir)
                
            

def state_norm(state):
    """Compute the L2 norm of a batch of field snapshots
    
    Arg:
        state (np.array):   field snapshots of shape (N, H, W, channels)
    
    Return:
        norm (np.array):    L2 norms of shape (N, 1, 1, 1)
    """
    
    s = state.get_shape().as_list()
    norm = tf.reduce_sum(tf.norm(state, ord=2, axis=3), axis=[1,2])/(tf.cast(s[1]*s[2], tf.float32))
    for i in range(1,4):
        norm = tf.expand_dims(norm, axis=i)
    return norm 


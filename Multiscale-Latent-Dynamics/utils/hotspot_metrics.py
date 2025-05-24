import numpy as np

class EM_Metric:
    """Compute model performance based on four sensitivity metrics of energetic material"""

    def __init__(self, area_rescale_factor, time_step, hotspot_treshold):
        """
        Args:
            area_rescale_factor (float):    (1/micro_meter ** 2) area of one pixel in "micro_meter ** 2"
            time_step (float):    time-step in ground-truth dataset in nano-seconds 
            hotspot_treshold (float):     Temperature (K) threshold to distinguish between hotspot and non-hotspot area 
        """
        self.area_rescale_factor = area_rescale_factor 
        self.time_step = time_step 
        self.hotspot_treshold = hotspot_treshold  


    def hs_metrics_traj(self, T_traj):
        """Calculate the hotspot metrics, i.e. (T, A, T_dot, A_dot) for single case

        Args: 
            T_traj (np.array):   temperature snapshots for single case of shape [H, W, time-steps]

        Return:
            hs_metrics (dictionary):    dictionary of four arrays corresponding to evolution of each hotspot metric  
        """
        ts = T_traj.shape[-1]
        hs_A_traj = [None] * ts
        hs_T_traj = [None] * ts
        hs_metrics = {}

        # Calculate area and avg temperature of hotspots
        for t in range(ts):
            temp_i = T_traj[:, :, t]
            hs_mask = temp_i > self.hotspot_treshold
            hs_area = np.count_nonzero(hs_mask)

            hs_area_rescaled = hs_area * self.area_rescale_factor
            hs_A_traj[t] = hs_area_rescaled

            masked_temp_field = temp_i * hs_mask

            if hs_area == 0:
                avg_hs_T = 0.0
            else:
                avg_hs_T = np.sum(masked_temp_field) / hs_area
            hs_T_traj[t] =  avg_hs_T
        
        hs_A_traj = np.array(hs_A_traj)
        hs_T_traj = np.array(hs_T_traj)
        
        hs_T_dot_traj = (hs_T_traj[..., 1:] - hs_T_traj[..., 0:-1]) / self.time_step
        hs_A_dot_traj = (hs_A_traj[..., 1:] - hs_A_traj[..., 0:-1]) / self.time_step
        
        hs_metrics['T'] = hs_T_traj
        hs_metrics['A'] = hs_A_traj
        hs_metrics['T_dot'] = hs_T_dot_traj
        hs_metrics['A_dot'] = hs_A_dot_traj

        return hs_metrics

    def hs_metrics_traj_collective(self, T_traj_collective):
        """Calculate the hotspot metrics, i.e. (T, A, T_dot, A_dot), for all test samples
        
        Args: 
            T_traj_collective (np.array):   temperature snapshots for all test samples of shape [N, H, W, timesteps]

        Return:
            hs_metrics_collective (dictionary):    dictionary of four arrays corresponding to evolution of each hotspot metric
        """
        n_samples = T_traj_collective.shape[0]
        hs_T_traj_collective = [None] * n_samples
        hs_A_traj_collective = [None] * n_samples
        hs_T_dot_traj_collective = [None] * n_samples
        hs_A_dot_traj_collective = [None] * n_samples
        hs_metrics_collective = {}

        for sample in range(n_samples):
            sample_hs_metrics = self.hs_metrics_traj(T_traj=T_traj_collective[sample, ...])
            hs_T_traj_collective[sample] = sample_hs_metrics['T']
            hs_A_traj_collective[sample] = sample_hs_metrics['A']
            hs_T_dot_traj_collective[sample] = sample_hs_metrics['T_dot']
            hs_A_dot_traj_collective[sample] = sample_hs_metrics['A_dot']

        hs_T_traj_collective = np.stack(hs_T_traj_collective, axis=0) 
        hs_A_traj_collective = np.stack(hs_A_traj_collective, axis=0) 
        hs_T_dot_traj_collective = np.stack(hs_T_dot_traj_collective, axis=0) 
        hs_A_dot_traj_collective = np.stack(hs_A_dot_traj_collective, axis=0) 
        
        hs_metrics_collective['T'] = hs_T_traj_collective
        hs_metrics_collective['A'] = hs_A_traj_collective
        hs_metrics_collective['T_dot'] = hs_T_dot_traj_collective
        hs_metrics_collective['A_dot'] = hs_A_dot_traj_collective

        return hs_metrics_collective
    

    def hs_metrics_mean_std(self, T_traj_collective):
        """Calculate the mean-value and standard-deviation of the hotspot metrics, i.e. (T, A, T_dot, A_dot) 
        
        Args: 
            T_traj_collective (np.array):   temperature snapshots for all test samples of shape [N, H, W, timesteps]

        Return:
            mean_std (dictionary):    dictionary of four arrays corresponding to 'mean-value plus/minus standard-deviation'
                                      of each hotspot metric
        """
        hs_metrics_collective = self.hs_metrics_traj_collective(T_traj_collective)
        mean_std = {}
        for metric in list(hs_metrics_collective.keys()):
            mean = np.mean(hs_metrics_collective[metric], axis=0)
            std = np.std(hs_metrics_collective[metric], axis=0)
            mean_std[metric] = [mean, mean + std, mean - std]
        
        return mean_std
    
    def prediction_performance(self, T_traj_gt, T_traj_pred):
        """Compute the RMSE between ground-truth and predicted hotspot metrics, i.e. (T, A, T_dot, A_dot) 
        
        Args: 
            T_traj_gt (np.array):   ground-truth temperature snapshots for all test samples of shape [N, H, W, timesteps];  
            T_traj_pred (np.array):   predicted temperature snapshots for all test samples of shape [N, H, W, timesteps];  
        
        Return:
            measure_mean (dictionary):    dictionary of four mean-RMSE for (T, A, T_dot, A_dot)
            measure (dictionary):    dictionary of four RMSE-trajectory for (T, A, T_dot, A_dot)
        """
        hs_metrics_gt = self.hs_metrics_traj_collective(T_traj_gt)
        hs_metrics_pred = self.hs_metrics_traj_collective(T_traj_pred)
        measure = {}
        measure_mean = {}

        for metric in list(hs_metrics_gt.keys()):
            ts = hs_metrics_gt[metric].shape[-1]
            rmse_traj = np.empty(shape=(ts,))
            for t in range(ts):
                rmse_traj[t] = np.sqrt( np.mean( (hs_metrics_gt[metric][:, t] - hs_metrics_pred[metric][:, t])**2 ) )
                
            measure[metric] = [rmse_traj]
            rmse = np.sqrt( np.mean( (hs_metrics_gt[metric] - hs_metrics_pred[metric])**2 ) )
            measure_mean[metric] = [rmse]
        
        return measure_mean, measure

    def field_prediction_rmse(self, F_traj_gt, F_traj_pred):
        """compute the RMSE of field snapshots"""
        rmse = np.sqrt( np.mean( (F_traj_gt - F_traj_pred)**2 ) )
        return rmse


def T_rescale_LP(T_normal):
    """Scale the ground-truth and predicted temperature field of our model"""
    T_min, T_max = 300, 5000
    T_scale = ((T_max - T_min) * T_normal) + T_min
    return T_scale

def T_rescale_Pv2(T_normal):
    """Scale the ground-truth and predicted temperature field of PARCv2"""
    T_min, T_max = 300, 7300
    T_scale = ((T_max - T_min) * T_normal) + T_min
    return T_scale
    
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



def plot_pred_field(field, latent_fields, t0, ts, field_idx, spatial_scale,
                    t_init = 0, n_rows=6, mean_var='mean', model='Meso_Coupled'):
                     
    """
    Plot the predicted fields and latent fields.

    Args:
        field (np.array):   concatenated ground_truth and predicted fields 
        latent_fields (np.array):   concatenated latent-mean and latent-log-varince fields
        t0 (int):   starting time-point in the corresponding trajectory
        ts (int):   time-step
        field_idx (int):    either of [0, 1, 2] to choose between [temperature, pressure, microstructure]
        spatial_scale (str):    either 'micro' or 'meso'
        n_rows (int):   number of rows in the plot
        mean_var (str):    either 'mean' or 'var' to choose between latent-mean or latent-log-variance fields
        model (str):    prediction model
    
    Return:
        matplotlib figure         
    """
    if field_idx == 0: 
        field_min = 300  # min temperature (K)
        field_max = 5000  # max temperature (K)
        unit = "(K)"
    elif field_idx == 1:
        field_min = -2  # min pressure (GPa)
        field_max = 50  # max pressure (GPa)
        unit = "(GPa)"
    elif field_idx == 2:
        field_min = 0  
        field_max = 1  
        unit = ""
    
    if spatial_scale == 'micro':
        physical_ts = 0.172 # micro time-step in nano-seconds
    elif spatial_scale == 'meso':
        physical_ts = 0.395 # meso time-step in nano-seconds
      
    fields = ['T', 'P', '\mu']
    ylabel_ls = [f"Simulation \n ${{{fields[field_idx]}}}_\mathrm{{true}}$"]
    ylabel_ls.append(f"{model} \n $\widehat{{{fields[field_idx]}}}$")
    if mean_var == 'mean':
        ylabel_ls += ['$\overline{z}_{1}$', '$\overline{z}_{2}$', '$\overline{z}_{3}$', '$\overline{z}_{4}$']
        latent_idx = 0
    elif mean_var == 'var':
        ylabel_ls += ['$z^{\sigma}_{1}$', '$z^{\sigma}_{2}$', '$z^{\sigma}_{3}$', '$z^{\sigma}_{4}$']
        latent_idx = 1
    
    fig, ax = plt.subplots(n_rows, 5, figsize=(20, (n_rows*2)))
    plt.subplots_adjust(wspace=0.001, hspace=0.05, top=0.85)                               
    for i in range(n_rows):
        if i==0 or i ==1: 
            img = field[i,:,:,:,0]
            min, max = np.amin(field[0,:,:,:,0]), np.amax(field[0,:,:,:,0])
            min_val, max_val = field_min, field_max
        else:
            img = latent_fields[latent_idx, :, :, :, i-2] 
            min, max = np.amin(img), np.amax(img)
            min_val, max_val = min, max
            unit = ''
        for j in range(5):
            ax[i][j].clear()
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            t = int(t0 + (j * ts))
            ax[i][j].imshow(np.squeeze(img[t, :, :]), cmap='jet', vmin=min, vmax=max)
            ax[0][j].set_title(f't = {round(t * physical_ts, 2)} ns', fontsize=20)
            ax[0][j].set_title(f't = {t + t_init}', fontsize=20)
        ax[i][0].set_ylabel(ylabel_ls[i], fontsize=15)    
        norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="jet"), ax=ax[i][4])
        cbar.set_label(label=unit)
        cbar.ax.tick_params(labelsize=15)
    fig.tight_layout()
    
    return fig





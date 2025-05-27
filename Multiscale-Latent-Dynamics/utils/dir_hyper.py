import os

base_dir = '/home/user/Multiscale-Latent-Dynamics' # replace with your path
dir = os.path.join(base_dir, 'weights')


# Dictionary specifying directories to save and load weights of 
# the encoder and decoder corresponding to each physical field
vae_dir = {
        'temperature' : {'encoder' : os.path.join( dir, 'vae', 'temperature_encoder.h5' ),
                         'decoder' : os.path.join( dir, 'vae', 'temperature_decoder.h5' ) },
        'pressure' :    {'encoder' : os.path.join( dir, 'vae', 'pressure_encoder.h5' ),
                         'decoder' : os.path.join( dir, 'vae', 'pressure_decoder.h5' ) },
        'microstructure' : {'encoder' : os.path.join( dir, 'vae', 'microstructure_encoder.h5' ),
                         'decoder' : os.path.join( dir, 'vae', 'microstructure_decoder.h5' ) },
    }

def dyn_dir(dynamics_model):
    """Build a dictionary specifying directories to save and load weights of latent evolution autoregressive functionals
    
    Args:
        dynamics_model (string):    architecture considered for autoregressive functionals, e.g., `unet`
    
    Return:
        dic (dictionary):   dictionary of directories
    """
    dic = { 
    'micro_dyn' : {
        'decoupled' : {'temperature' : os.path.join( dir, 'micro_dyn', 'decoupled', 'temperature', dynamics_model ),
                      'pressure' : os.path.join( dir, 'micro_dyn', 'decoupled', 'pressure', dynamics_model ),
                      'microstructure' : os.path.join( dir, 'micro_dyn', 'decoupled', 'microstructure', dynamics_model )

        },
        'coupled' : os.path.join( dir, 'micro_dyn', 'coupled', dynamics_model )
                },
    'meso_dyn' : {
        'coupled' : os.path.join( dir, 'meso_dyn', 'coupled', dynamics_model )
                }
    }
    return dic

###
def mkdir_ifnot(dirs):
    if not os.path.isdir(dirs): 
        os.makedirs(dirs)
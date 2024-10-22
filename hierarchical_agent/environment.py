import numpy as np
import torch
import sys
from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import features

FLAGS = flags.FLAGS
FLAGS(sys.argv)

class StarCraft2Env:
    """
    A class to represent the StarCraft II environment.
                
    Attributes
    ----------
    features_to_select : list
        A list of feature indices to select from the observation.
        
    Methods
    -------
    create_env(interface_params, env_params):
        Static method to create and return a StarCraft II environment instance.
    preprocess_observation(obs):
        Preprocesses the observation by selecting relevant feature layers and 
        converting them to a tensor.
    """
    def __init__(self, features_index):
        self.features_to_select = features_index
            
    @staticmethod
    def create_env(
        interface_params,
        env_params,
    ):
        """
        Creates a StarCraft II environment with the specified interface and environment parameters.

        Args:
            interface_params (dict): A dictionary containing interface parameters such as 
                'screen_size', 'minimap', 'use_feature_units', and 'use_raw_units'.
            env_params (dict): A dictionary containing additional environment parameters 
                to be passed to the SC2Env constructor.

        Returns:
            SC2Env: An instance of the SC2Env class configured with the specified parameters.
        """
        env = sc2_env.SC2Env(
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen=interface_params['screen_size'], 
                    minimap=interface_params['minimap']
                ),
                use_feature_units=interface_params['use_feature_units'],
                use_raw_units=interface_params['use_raw_units'],
            ),
            **env_params,
        )
        return env
    
    def preprocess_observation(self, obs):
        """
        Preprocesses the observation by selecting relevant feature layers and converting them to a tensor.
        Args:
            obs (dict): A dictionary containing the observation data. It must have a key 'feature_screen' 
                        which is a list or array of feature layers.
        Returns:
            torch.Tensor: A tensor containing the selected feature layers, converted to float32.
        """
        feature_screen = obs.observation['feature_screen']
        
        # Select relevant feature layers
        feature_screen_selected = []
        for feature_index in self.features_to_select:
            feature_screen_selected.append(feature_screen[feature_index])
        
        return torch.tensor(np.array(feature_screen_selected), dtype=torch.float32)

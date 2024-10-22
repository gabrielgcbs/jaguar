import numpy as np
import random
import math
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import mse_loss
from torch.autograd import Variable
from models import DQNModel, DDQNModel
from collections import namedtuple, deque

try:
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'config', 'params.yaml')), "r") as fp:
        params = yaml.safe_load(fp)
except FileNotFoundError as e:
    raise Exception(e)

dqn_params = params['dqn']
ddqn_params = params['ddqn']

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def make_dir(path) -> None:
    """
    Creates a directory if it does not already exist.

    Args:
        path (str): The path of the directory to create.

    Returns:
        None
    """
    if not os.path.isdir(path):
        os.mkdir(path)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'over'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

LR_ddqn = ddqn_params['learning_rate']
MOMENTUM = ddqn_params['momentum']
GAMMA_ddqn = ddqn_params['gamma']
ACCUMULATION_STEPS = ddqn_params['accumulation_steps']
MIN_EPSILON = ddqn_params['min_epsilon']
EPSILON = ddqn_params['epsilon']
EPSILON_DISCOUNT_RATE = ddqn_params['epsilon_discount_rate']
DISCOUNT_FACTOR = ddqn_params['discount_factor']

class MacroAgent:
    """
    A class used to represent a MacroAgent for the strategic decision-making of Jaguar.
    Attributes
    ----------
    input_dim : int
        The dimension of the input state space.
    action_number : int
        The number of possible actions.
    epsilon : float
        The exploration rate for the epsilon-greedy strategy.
    memory : ReplayMemory
        The replay memory to store experiences.
    batch_size : int
        The size of the batch for training.
    goal_done : bool
        A flag indicating if the goal is achieved.
    Q_network : DDQNModel
        The Q-network for estimating action values.
    target_network : DDQNModel
        The target network for stable target value estimation.
    optimizer : torch.optim.Optimizer
        The optimizer for training the Q-network.
    Methods
    -------
    build_network():
        Builds the Q-network and target network.
    update_target_net():
        Updates the target network with the current Q-network parameters.
    update_Q_network(state, action, next_state, reward, terminal):
        Updates the Q-network based on a batch of experiences.
    select_strategy(state):
        Selects an action based on the current policy (epsilon-greedy).
    update_epsilon():
        Decreases the epsilon value for exploration.
    stop_epsilon():
        Stops exploration by setting epsilon to zero.
    restore_epsilon():
        Restores the previous epsilon value.
    set_training_mode():
        Sets the networks to training mode.
    set_evaluation_mode():
        Sets the networks to evaluation mode and stops exploration.
    set_pretrained_mode(freeze_layers):
        Sets the networks to pretrained mode and optionally freezes layers.
    save_model(step, path, map_name):
        Saves the Q-network and target network models to disk.
    load_model(step, path):
        Loads the Q-network and target network models from disk.
    """
    def __init__(
        self, 
        input_dim, 
        action_space,
        batch_size,
    ):
        self.action_number = action_space
        self.input_dim = input_dim
        self.epsilon = EPSILON
        self.memory = ReplayMemory(REPLAY__CAPACITY)
        self.build_network()
        self.batch_size = batch_size
        self.goal_done = False
        
    def build_network(self):
        """
        Builds the Q-network and target network for the agent using the DDQNModel.
        
        This method initializes the Q-network and target network with the given input dimensions
        and number of actions. It also sets up the optimizer for training the Q-network.
        
        Attributes:
            Q_network (DDQNModel): The Q-network used for estimating Q-values.
            target_network (DDQNModel): The target network used for stabilizing training.
            optimizer (torch.optim.Adam): The optimizer used for training the Q-network.
        """
        self.Q_network = DDQNModel(self.input_dim, self.action_number)
        self.target_network = DDQNModel(self.input_dim, self.action_number)
        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=LR_ddqn)
    
    def update_target_net(self):
        """
        Updates the target network by copying the state dictionary from the current Q-network.
        
        This method is typically used in deep reinforcement learning to periodically synchronize
        the target network with the current Q-network, ensuring that the target network remains
        a stable reference for calculating target values during training.
        """
        self.target_network.load_state_dict(self.Q_network.state_dict())
    
    def update_Q_network(self):
        """
        Updates the Q-network using a batch of transitions sampled from memory.
        This method performs the following steps:
        1. Checks if there are enough transitions in memory to sample a batch.
        2. Samples a batch of transitions from memory.
        3. Transposes the batch to convert it from a batch-array of Transitions to a Transition of batch-arrays.
        4. Concatenates the states, next states, actions, rewards, and terminal flags into batches.
        5. Sets the Q-network and target network to evaluation mode.
        6. Uses the current Q-network to evaluate the action with the highest Q-value for the next states.
        7. Converts the selected actions to one-hot encoding.
        8. Uses the target network to evaluate the Q-values for the next states and masks them with the one-hot encoded actions.
        9. Calculates the discounted Q-values and the target values.
        10. Sets the Q-network to training mode.
        11. Computes the Q-values for the current states and actions.
        12. Calculates the mean squared error loss between the computed Q-values and the target values.
        13. Performs backpropagation and updates the Q-network's weights using the optimizer.
        
        Returns:
            torch.Tensor: The loss value after the update.
        """
                
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))        
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        over_batch = torch.cat(batch.over)
        
        self.Q_network.eval()
        self.target_network.eval()
        
        # use current network to evaluate action argmax_a' Q_current(s', a')_
        action_new = self.Q_network.forward(next_state_batch).max(dim=1)[1].cpu().data.view(-1, 1)
        action_new_onehot = torch.zeros(self.batch_size, self.action_number)
        action_new_onehot = Variable(action_new_onehot.scatter_(1, action_new, 1.0))
        
        #### use target network to evaluate value y = r + discount_factor * Q_tar(s', a')
        
        next_q_values = self.target_network.forward(next_state_batch)
        masked_q_values = (next_q_values * action_new_onehot).sum(dim=1)
        # Ensure over_batch is broadcastable with masked_q_values
        discounted_q_values = masked_q_values * over_batch
        # Calculate the final target values
        y = (reward_batch + discounted_q_values * DISCOUNT_FACTOR).float() # to force float32 instead of double
        
        # regression Q(s, a) -> y
        self.Q_network.train()
        Q = (self.Q_network.forward(state_batch)*action_batch.unsqueeze(1)).sum(dim=1)
        loss = mse_loss(input=Q, target=y.detach())
        
        # backward optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.data

    def select_strategy(self, state):
        """
        Selects an action based on the current state using an epsilon-greedy strategy.

        Parameters:
        state (torch.Tensor): The current state of the environment.

        Returns:
        torch.Tensor: The selected action. If a random number is greater than epsilon,
                      the action with the highest expected reward is selected using the Q-network.
                      Otherwise, a random action is selected.
        """
        if np.random.random() > self.epsilon:
            with torch.no_grad():
                estimate = self.Q_network.forward(state).max(dim=1)
                return estimate[1].data[:1]
        else:
            return torch.tensor([np.random.randint(0, self.Q_network.fc2.out_features)])
    
    def update_epsilon(self):
        """
        Update the epsilon value by decrementing it with a predefined discount rate.
        
        This method reduces the epsilon value by a constant rate (EPSILON_DISCOUNT_RATE) 
        until it reaches a minimum threshold (MIN_EPSILON). Epsilon is typically used 
        in reinforcement learning algorithms to balance exploration and exploitation.

        Attributes:
            epsilon (float): The current epsilon value.
            MIN_EPSILON (float): The minimum allowable value for epsilon.
            EPSILON_DISCOUNT_RATE (float): The rate at which epsilon is decremented.
        """
        if self.epsilon > MIN_EPSILON:
            self.epsilon -= EPSILON_DISCOUNT_RATE
    
    def stop_epsilon(self):
        """
        Temporarily stops the exploration by setting epsilon to 0.

        This method saves the current value of epsilon to a temporary variable
        (epsilon_tmp) and then sets epsilon to 0. This can be used to stop
        exploration during certain phases of training or evaluation.
        """
        self.epsilon_tmp = self.epsilon        
        self.epsilon = 0        
    
    def restore_epsilon(self):
        """
        Restores the epsilon value to its temporary stored value.

        This method sets the epsilon attribute back to the value stored in epsilon_tmp.
        """
        self.epsilon = self.epsilon_tmp 

    def set_training_mode(self):
        """
        Sets the training mode for the Q-network and the target network.

        This method switches both the Q-network and the target network to training mode,
        which enables features like dropout and batch normalization to work in training mode.
        """
        self.Q_network.train()
        self.target_network.train()

    def set_evaluation_mode(self):
        """
        Sets the agent to evaluation mode.

        This method switches the Q-network and target network to evaluation mode,
        which typically disables certain layers like dropout and batch normalization
        that behave differently during training and evaluation. Additionally, it
        stops the epsilon-greedy exploration strategy, ensuring that the agent
        exploits its learned policy without further exploration.
        """
        self.Q_network.eval()
        self.target_network.eval()
        self.stop_epsilon()
        
    def set_pretrained_mode(self, freeze_layers):
        """
        Adjusts the model to operate in pretrained mode.

        If `freeze_layers` is True, the fully connected layers of both the Q_network 
        and the target_network are frozen to prevent further training. Additionally, 
        the learning rate of the optimizer is reduced to one-tenth of its current value.

        Args:
            freeze_layers (bool): A flag indicating whether to freeze the fully 
                                  connected layers of the networks.

        Prints:
            The new learning rate for the macro.
        """
        if freeze_layers:
            self.Q_network.freeze_fc_layers()
            self.target_network.freeze_fc_layers()
        new_learning_rate = self.optimizer.param_groups[0]['lr']/10
        print(f'New learning rate for macro: {new_learning_rate}')
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_learning_rate
        
    def save_model(self, step, path, map_name):
        """
        Saves the current state of the Q-network and target network to the specified path.

        Args:
            step (int): The current step or iteration number, used to create a unique directory for the model.
            path (str): The base directory where the model should be saved.
            map_name (str): The name of the map or environment, used as part of the saved file names.

        Returns:
            None
        """
        path_to_model = f"{path}run_{step}/"
        make_dir(path_to_model)
        self.Q_network.save(step, f"{path_to_model}high_Q_net.pt", self.optimizer, map_name)
        self.target_network.save(step, f"{path_to_model}high_target_net.pt", self.optimizer, map_name)
        
    def load_model(self, step, path):
        """
        Loads the model and target network from the specified path and step.
        Args:
            step (int): The step number to identify the specific model run.
            path (str): The base path where the model and target network files are stored.
        Returns:
            str: The name of the map loaded by the Q_network.
        """
        path_to_model = f"{path}run_{step}/"
        
        map_name = self.Q_network.load(f"{path_to_model}high_Q_net.pt")
        self.target_network.eval()      # Set to inference mode
        
        _ = self.target_network.load(f"{path_to_model}high_target_net.pt")
        self.target_network.eval()      # Set to inference mode
        return map_name

#######################
GAMMA_DQN = dqn_params['gamma']                     # Discount factor
EPS_START = dqn_params['eps_start']                 # Starting value of epsilon
EPS_END = dqn_params['eps_end']                     # Final value of epsilon
EPS_DECAY = dqn_params['eps_decay']                 # Controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = dqn_params['tau']                             # Update rate of the target network
LR_DQN = dqn_params['learning_rate']                # Learning rate of the ``AdamW`` optimizer
AMSGRAD = dqn_params['amsgrad']                     # Amsgrad parameter of ``AdamW`` optimizer
REPLAY__CAPACITY = dqn_params['replay_capacity']    # Replay buffer capacity to store previous steps

class MicroAgent:
    """
    A MicroAgent class implementing a Deep Q-Network (DQN) for tactical decision-making of Jaguar.
    Attributes:
        policy_net (DQNModel): The primary network used for selecting actions.
        target_net (DQNModel): The target network used for computing target Q-values.
        optimizer (torch.optim.AdamW): The optimizer for training the policy network.
        memory (ReplayMemory): The replay memory for storing past experiences.
        batch_size (int): The size of the batch used for training.
        steps_done (int): The number of steps taken in the environment.
        eval_mode (bool): A flag indicating if the agent is in evaluation mode.
    Methods:
        select_action(state, mask):
            Selects an action based on the current state and action mask using an epsilon-greedy policy.
        update(state, action, next_state, rewards, terminal):
            Updates the policy network based on the given transition (state, action, next_state, rewards, terminal).
        update_target_net():
            Soft updates the target network's weights.
        set_training_mode():
            Sets the agent to training mode.
        set_evaluation_mode():
            Sets the agent to evaluation mode.
        set_pretrained_mode(freeze_layers):
            Sets the agent to pretrained mode and optionally freezes certain layers.
        mask_actions(q_values, action_mask):
            Masks out invalid actions by adding a large negative number to their Q-values.
        save_model(step, path):
            Saves the current state of the policy and target networks to the specified path.
        load_model(step, path):
            Loads the state of the policy and target networks from the specified path.
    """
    def __init__(self, input_dim, action_space, batch_size):
        self.policy_net = DQNModel(input_dim, action_space).to(device)
        self.target_net = DQNModel(input_dim, action_space).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR_DQN, amsgrad=AMSGRAD)
        self.memory = ReplayMemory(REPLAY__CAPACITY)
        self.batch_size = batch_size
        self.steps_done = 0
        self.eval_mode = False
    
    def select_action(self, state, mask):
        """
        Selects an action based on the current state and action mask using an epsilon-greedy strategy.

        Parameters:
        state (np.ndarray): The current state of the environment.
        mask (np.ndarray): A binary mask indicating which actions are valid (1) or invalid (0).

        Returns:
        torch.Tensor: The selected action as a tensor with shape (1, 1).
        """
        sample = random.random()
        eps_threshold = 0
        if not self.eval_mode:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                tensor_state = torch.FloatTensor(state.flatten()).unsqueeze(0)
                q_values = self.policy_net(tensor_state)
                masked_q_values = self.mask_actions(q_values, mask)
                return masked_q_values.max(1).indices.view(1, 1)
        else:
            return np.random.choice(mask.nonzero().reshape(1, -1)[0])
    
    def update(self):
        """
        Perform one step of the optimization on the policy network.
        This method updates the policy network by sampling a batch of transitions
        from memory, computing the expected Q values, and optimizing the model
        using Huber loss.
        Steps:
        1. Check if there are enough transitions in memory to sample a batch.
        2. Sample a batch of transitions and transpose it.
        3. Reshape the states to match the input shape of the network layers.
        4. Compute a mask of non-final states and concatenate the batch elements.
        5. Compute Q(s_t, a) for the current states and actions.
        6. Compute V(s_{t+1}) for the next states using the target network.
        7. Compute the expected Q values.
        8. Compute the Huber loss between the predicted and expected Q values.
        9. Optimize the model by performing a backward pass and gradient clipping.
        
        Returns:
            loss (torch.Tensor): The computed loss value.
        """
        # Perform one step of the optimization (on the policy network)
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Pysc2 states are of shape (n_features, lines, rows)
        # so we need to reshape it to pass as inputs to the network layers
        # of shape (batch_size, n_features*lines*rows)
        reshape_size = (self.batch_size, self.policy_net.layer1.in_features)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        ).reshape(reshape_size[0], reshape_size[1])
        
        state_batch = torch.cat(batch.state).reshape(reshape_size[0], reshape_size[1])
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA_DQN) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss

    def update_target_net(self):
        """
        Perform a soft update of the target network's weights.

        This method updates the weights of the target network (self.target_net) 
        by blending them with the weights of the policy network (self.policy_net) 
        using a factor τ (TAU). The update rule is:
        
            θ' ← τ * θ + (1 − τ) * θ'
        
        where:
        - θ' represents the weights of the target network.
        - θ represents the weights of the policy network.
        - τ (TAU) is a blending factor between 0 and 1.

        The updated weights are then loaded back into the target network.
        """
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def set_training_mode(self):
        """
        Sets the policy and target networks to training mode.

        This method changes the mode of the policy network and the target network
        to training mode, which enables certain layers (e.g., dropout, batch normalization)
        to behave appropriately during training.
        """
        self.policy_net.train()
        self.target_net.train()
    
    def set_evaluation_mode(self):
        """
        Sets the policy and target networks to evaluation mode.

        This method changes the mode of the policy and target networks to evaluation mode,
        which affects certain layers like dropout and batch normalization. It also sets
        the `eval_mode` attribute to True, indicating that the agent is in evaluation mode.
        """
        self.policy_net.eval()
        self.target_net.eval()
        self.eval_mode = True
        
    def set_pretrained_mode(self, freeze_layers):
        """
        Adjusts the model to operate in pretrained mode.

        If `freeze_layers` is True, it freezes the later layers of both the 
        policy network and the target network. Additionally, it reduces the 
        learning rate of the optimizer by a factor of 10.

        Args:
            freeze_layers (bool): A flag indicating whether to freeze the later 
                      layers of the networks.
        """
        if freeze_layers:
            self.policy_net.freeze_later_layers()
            self.target_net.freeze_later_layers()
        new_learning_rate = self.optimizer.param_groups[0]['lr']/10
        print(f'New learning rate for micro: {new_learning_rate}')
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_learning_rate

    def mask_actions(self, q_values, action_mask):
        """
        Masks out invalid actions by adding a large negative number to their Q-values.

        Args:
            q_values (torch.Tensor): The Q-values for all actions.
            action_mask (torch.Tensor): A binary mask indicating valid (1) and invalid (0) actions.

        Returns:
            torch.Tensor: The Q-values with invalid actions masked out.
        """
        # Add a large negative number to masked-out actions
        masked_q_values = q_values + (action_mask - 1) * 1e9
        return masked_q_values

    def save_model(self, step, path):
        """
        Saves the current state of the policy and target networks to the specified path.

        Args:
            step (int): The current training step, used to create a unique directory for the saved models.
            path (str): The base directory where the model should be saved. A subdirectory named 'run_{step}' will be created.

        Returns:
            None
        """
        path_to_model = f"{path}run_{step}/"
        make_dir(path_to_model)
        self.policy_net.save(step, f"{path_to_model}lower_Q_net.pt", self.optimizer)
        self.target_net.save(step, f"{path_to_model}lower_target_net.pt", self.optimizer)

    def load_model(self, step, path):
        """
        Loads the policy and target networks from the specified path and sets them to inference mode.
        Args:
            step (int): The step number used to construct the path to the model files.
            path (str): The base path where the model files are stored.
        Returns:
            None
        """
        path_to_model = f"{path}run_{step}/"
        self.policy_net.load(f"{path_to_model}lower_Q_net.pt")
        self.policy_net.eval()      # Set to inference mode
        
        self.target_net.load(f"{path_to_model}lower_target_net.pt")
        self.target_net.eval()      # Set to inference mode

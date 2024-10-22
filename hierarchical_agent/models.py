import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """
    A base model class that extends nn.Module and provides methods to save and load model checkpoints.
    """

    def save(self, step, path, optimizer, map_name=None) -> None:
        """
        Saves the model state, optimizer state, and additional information to a specified path.

        Parameters:
        step (int): The current training step or epoch.
        path (str): The file path where the checkpoint will be saved.
        optimizer (torch.optim.Optimizer): The optimizer whose state will be saved.
        map_name (str, optional): Additional information to be saved in the checkpoint. Default is None.

        Returns:
            None
        """
        torch.save(
            {
                "step": step,
                "state_dict": self.state_dict(),
                "optimizer": optimizer.state_dict(),
                "map": map_name,
            },
            path,
        )

    def load(self, checkpoint_path, optimizer=None):
        """
        Loads the model state and optionally the optimizer state from a checkpoint file.

        Parameters:
        checkpoint_path (str): The file path from which the checkpoint will be loaded.
        optimizer (torch.optim.Optimizer, optional): The optimizer whose state will be loaded. Default is None.

        Returns:
        str or None: The additional information saved in the checkpoint, if available. Otherwise, returns None.
        """
        checkpoint = torch.load(checkpoint_path)
        step = checkpoint["step"]
        self.load_state_dict(checkpoint["state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint["map"] if "map" in checkpoint.keys() else None


""" DDQN
ref: https://github.com/chinancheng/DDQN.pytorch/tree/master
"""


class DDQNModel(BaseModel):
    """
    A Deep Double Q-Network (DDQN) model for reinforcement learning.
    Args:
        input_dim (tuple): The dimensions of the input (e.g., (channels, height, width)).
        action_num (int): The number of possible actions.
    Attributes:
        conv1 (nn.Sequential): The first convolutional layer followed by a ReLU activation.
        conv2 (nn.Sequential): The second convolutional layer followed by a ReLU activation.
        conv3 (nn.Sequential): The third convolutional layer followed by a ReLU activation.
        fc1 (nn.Sequential): The first fully connected layer followed by a ReLU activation.
        fc2 (nn.Linear): The second fully connected layer that outputs the action values.
    Methods:
        _get_conv_output(shape):
            Computes the output size of the convolutional layers given an input shape.
        forward(observation):
            Defines the forward pass of the network.
        freeze_conv_layers():
            Freezes the parameters of the convolutional layers to prevent them from being updated during training.
        freeze_fc_layers():
            Freezes the parameters of the fully connected layers to prevent them from being updated during training.
    """

    def __init__(self, input_dim, action_num):
        super(DDQNModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_dim[0], out_channels=16, kernel_size=(8, 8), stride=4
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
        )
        # Fully connected network
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=4 * 4 * 32, out_features=64), nn.ReLU()
        )
        self.fc2 = nn.Linear(in_features=64, out_features=action_num)
        self.float()

    def _get_conv_output(self, shape):
        """
        Computes the output size of the convolutional layers for a given input shape.

        Args:
            shape (tuple): The shape of the input tensor (excluding batch size).

        Returns:
            int: The size of the flattened output after passing through the convolutional layers.
        """
        # Create a dummy input with the given shape
        x = torch.zeros(1, *shape)
        # Pass it through the conv layers
        x = self.conv1(x)
        x = self.conv2(x)
        # Flatten the output
        return int(torch.flatten(x, 1).size(1))

    def forward(self, observation):
        """
        Perform a forward pass through the neural network.
        Args:
            observation (torch.Tensor): The input tensor representing the observation.
        Returns:
            torch.Tensor: The output tensor after passing through the network layers.
        """
        out = self.conv1(observation)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc1(out.view(-1, 4 * 4 * 32))
        out = self.fc2(out)

        return out

    def freeze_conv_layers(self):
        """
        Freezes the convolutional layers of the model.

        This method sets the `requires_grad` attribute of all parameters in the
        convolutional layers (conv1, conv2, conv3) to False, effectively freezing
        them during training. This means that the gradients will not be computed
        for these layers, and their weights will not be updated.

        Note:
            Only the last convolutional layer is not frozen.
        """
        # Freeze all layers except the last one
        conv_layers = [self.conv1, self.conv2, self.conv3]
        for layer in conv_layers:
            for param in layer.parameters():
                param.requires_grad = False

    def freeze_fc_layers(self):
        """
        Freezes the fully connected layers of the model.
        This method sets the `requires_grad` attribute of all parameters in the
        fully connected layers (fc1 and fc2) to False, effectively freezing them
        during training. This means that the gradients for these layers will not
        be computed, and their weights will not be updated.
        """
        # Freeze all layers except the last one
        fc_layers = [self.fc1, self.fc2]
        for layer in fc_layers:
            for param in layer.parameters():
                param.requires_grad = False


""" REFERENCE: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html """


class DQNModel(BaseModel):
    """
    A Deep Q-Network (DQN) model for reinforcement learning.
    Args:
        input_dim (int): The dimension of the input features.
        action_space (int): The number of possible actions.
    Attributes:
        layer1 (nn.Linear): The first fully connected layer.
        layer2 (nn.Linear): The second fully connected layer.
        layer3 (nn.Linear): The output layer that maps to the action space.
    Methods:
        forward(x):
            Performs a forward pass through the network.
        freeze_early_layers():
            Freezes the parameters of the early layers (layer1 and layer2) to prevent them from being updated during training.
        freeze_later_layers():
            Freezes the parameters of the later layer (layer3) to prevent it from being updated during training.
    """

    def __init__(self, input_dim, action_space):
        super(DQNModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_space)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        """
        Perform a forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Output tensor after passing through the network layers.
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def freeze_early_layers(self):
        """
        Freezes all layers except the last one in the model.

        This method sets the `requires_grad` attribute of all parameters in
        the early layers (layer1 and layer2) to False, effectively freezing
        them during training. This is useful when you want to fine-tune only
        the later layers of the model while keeping the early layers fixed.
        """
        # Freeze all layers except the last one
        layers = [self.layer1, self.layer2]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

    def freeze_later_layers(self):
        """
        Freezes the parameters of all layers except the last one in the model.

        This method sets the `requires_grad` attribute of the parameters in the specified
        layers to `False`, preventing them from being updated during training. This is
        typically used to fine-tune a pre-trained model by only training the last layer(s).

        Note:
            Currently, this method only freezes `self.layer3`. If additional layers need
            to be frozen, they should be added to the `layers` list.
        """
        # Freeze all layers except the last one
        layers = [self.layer3]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

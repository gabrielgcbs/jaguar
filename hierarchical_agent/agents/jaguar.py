import numpy as np
import torch
from typing import Tuple, List
from pysc2.lib import actions, units
from policy import MacroAgent, MicroAgent
from config import avail_actions

_PLAYER_SELF = 1
_PLAYER_ALLY = 2
_PLAYER_NEUTRAL = 3
_PLAYER_ENEMY = 4
_COMMAND_CENTER = units.Terran.CommandCenter

class Grid:
    """
    A class to represent a Grid.
    Attributes:
        zone (int): The zone identifier for the grid.
        x_min (int): The minimum x-coordinate of the grid.
        y_min (int): The minimum y-coordinate of the grid.
        x_max (int): The maximum x-coordinate of the grid.
        y_max (int): The maximum y-coordinate of the grid.
        x (int): The central x-coordinate of the grid, calculated as the midpoint of x_min and x_max.
        y (int): The central y-coordinate of the grid, calculated as the midpoint of y_min and y_max.
    Methods:
        __repr__(): Returns a string representation of the Grid object.
    """
    def __init__(self, zone, x_min, y_min, x_max, y_max):
        self.zone = zone
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.x = (x_max+x_min)//2
        self.y = (y_max+y_min)//2
        
    def __repr__(self) -> str:
        return f"Zone {self.zone}\nX: {self.x}\nY: {self.y}"

class Jaguar:
    """
    Initialize the Jaguar agent with the given parameters.
    Args:
        mid_input_dim (int): Dimension of the input for the macro agent.
        lower_input_dim (int): Dimension of the input for the micro agent.
        screen_size (tuple): Size of the screen to be used for the grid.
        grid_resize_factor (float): Factor by which the grid should be resized.
        maps (dict): Dictionary containing map information.
        current_map (str): Name of the current map.
        batch_size (int): Batch size for training the agents.
    Attributes:
        grid_resize_factor (float): Factor by which the grid is resized.
        last_unit_position (NoneType): Placeholder for the last unit position.
        current_strategy (NoneType): Placeholder for the current strategy.
        current_micro_action (NoneType): Placeholder for the current micro action.
        last_action (NoneType): Placeholder for the last action.
        max_actions_queue (int): Maximum number of actions in the queue.
        strategies (list): List of strategies for the current map.
        alliances (list): List of alliances for the current map.
        action_space (list): List of possible actions.
        strategy_space_size (int): Size of the strategy action space.
        lower_action_space_size (int): Size of the lower-level action space.
        upper_policy (MacroAgent): Macro agent policy.
        lower_policy (MicroAgent): Micro agent policy.
        map_name (str): Name of the current map.
    """
    def __init__(
        self,
        mid_input_dim,
        lower_input_dim,
        screen_size,
        grid_resize_factor,
        maps,
        current_map,
        batch_size,
    ):
        # Settings
        self.grid_resize_factor = grid_resize_factor
        self.last_unit_position = None
        self.current_strategy = None
        self.current_micro_action = None
        self.last_action = None
        self.max_actions_queue = 5
        self._set_grid(screen_size)     # Define grid to map the feature screen

        self.strategies = self._define_map_strategies(maps, current_map)
        self.alliances = self._define_map_alliances(maps, current_map)
        self.action_space = self._get_action_space()
        
        self.strategy_space_size = self._get_action_space_size(self.strategies)
        self.lower_action_space_size = self._get_action_space_size(self.strategies, level=1)
        
        # Policies
        self.upper_policy = MacroAgent(mid_input_dim, self.strategy_space_size, batch_size)
        self.lower_policy = MicroAgent(lower_input_dim, self.lower_action_space_size, batch_size)
        
        self.map_name = current_map

    @staticmethod
    def _define_map_strategies(maps_list, current_map):
        """
        Defines the strategies for the given map based on the available strategies.

        Args:
            maps_list (dict): A dictionary where keys are map names and values are dictionaries containing map details, including 'strategies'.
            current_map (str): The name of the current map for which strategies need to be defined.

        Returns:
            list: A list of strategies that are applicable to the current map.
        """
        strategy_cat = maps_list[current_map]['strategies']
        strategies = []
        for strategy in avail_actions.STRATEGIES:
            if strategy.category in strategy_cat:
                strategies.append(strategy)
        return strategies
    
    @staticmethod
    def _define_map_alliances(maps_list: List, current_map: str) -> List[int]:
        """
        Defines the alliances for a given map.

        Args:
            maps_list (list): A list of maps where each map is a dictionary containing map details.
            current_map (str): The name of the current map in the maps_list.

        Returns:
            list: A list of alliances for the current map, where each alliance is represented by a constant.
        """
        map_alliaces = maps_list[current_map]['alliances']
        alliances = []
        alliances_dict = {
            'self': _PLAYER_SELF,
            'ally': _PLAYER_ALLY,
            'neutral': _PLAYER_NEUTRAL,
            'enemy': _PLAYER_ENEMY,
        }
        for alliance in map_alliaces:
            alliances.append(alliances_dict[alliance])
        return alliances
                
    def _get_action_space(self) -> np.ndarray:
        """
        Generates the action space for the agent by aggregating micro actions from all strategies.

        This method iterates through each strategy in the agent's strategies list, collects all 
        micro actions, removes duplicates, sorts them, and returns the resulting action space 
        as a numpy array.

        Returns:
            np.ndarray: A sorted numpy array containing the unique micro actions from all strategies.
        """
        action_space = []
        for strategy in self.strategies:
            action_space += strategy.micro_actions
        return np.array(sorted(set(action_space)))
    
    def _get_action_space_size(self, space, level=0) -> int:
        """
        Calculate the size of the action space.

        Parameters:
        space (list): The list of actions available at the current level.
        level (int, optional): The hierarchical level of the action space. Defaults to 0.

        Returns:
        int: The size of the action space. If level is 0, returns the length of the space list.
             Otherwise, returns the number of unique micro actions across all strategies.
        """
        if level == 0:
            return len(space)
        else:
            action_space = []
            for strategy in self.strategies:
                action_space += strategy.micro_actions
            return len(set(action_space))
            
    def _actions_in_queue(self, feature_units) -> bool:
        """
        Checks if any unit in the feature_units list has fewer actions in its queue than the maximum allowed.

        Args:
            feature_units (list): A list of units, where each unit is represented as a dictionary containing its attributes.

        Returns:
            bool: True if any unit belonging to the player has fewer actions in its queue than the maximum allowed, False otherwise.
        """
        for unit in feature_units:
            if unit['alliance'] == _PLAYER_SELF:
                return unit['order_length'] < self.max_actions_queue
    
    def choose_mid_strategy(self, state):
        """
        Selects a mid-level strategy based on the given state.

        This method uses the upper policy to select a strategy based on the provided state.
        It then maps the strategy to corresponding actions, sets the action mask, and updates
        the current strategy.

        Args:
            state: The current state used to select the strategy.

        Returns:
            The selected strategy.
        """
        strategy = self.upper_policy.select_strategy(state)
        self.current_strategy = strategy
        self.upper_policy.strategy_actions = self.map_strategy_actions(strategy)
        self.action_mask = self.set_mask()
        return strategy

    def choose_micro_action(self, state, available_actions, feature_units):
        """Selects and executes an appropriate micro action based on availability and context.

        Args:
            available_actions: A dictionary or module containing available action functions.
            feature_units: A list of feature units in the environment.

        Returns:
            A tuple containing the executed action function and the action index,
            or (actions.FUNCTIONS.no_op(), 0) if no action is applicable.
        """
        self.last_action = self.current_micro_action if self.current_micro_action != avail_actions._NO_OP else self.last_action
        
        # INITIALIZE MAP WITH SELECTED COMMAND CENTER
        if self.map_name == 'BuildMarines' and self.last_action is None:
            action = avail_actions._SELECT_POINT
            point = self.find_point(feature_units)
            self.current_micro_action = action
            return action('select', point), avail_actions.ACTIONS.index(action)
        
        action_index = self.lower_policy.select_action(state, self.action_mask)
        action = self.map_action(action_index)
        
        # Initialize the current action as no_op
        self.current_micro_action = actions.FUNCTIONS.no_op
        
        if not self._is_action_available(action.id, available_actions) or not self._is_action_available(avail_actions.ACTIONS.index(action), self.upper_policy.strategy_actions) or not self._is_in_action_list(action.id):
            return actions.FUNCTIONS.no_op(), 0  # Early return for unavailable actions
        
        if action == avail_actions._SELECT_ARMY:     
            self.current_micro_action = action       
            return action('select'), action_index
        
        if action == avail_actions._SELECT_UNIT:     
            self.current_micro_action = action
            unit_index = self.find_unit(feature_units)    
            return action('select', unit_index), action_index
                
        if action == avail_actions._SELECT_POINT:
            point = self.find_point(feature_units)
            if point is None:
                return actions.FUNCTIONS.no_op(), 0
            self.current_micro_action = action  
            return action('select', point), action_index

        if action == avail_actions._ATTACK_SCREEN:
            attack_action = self.attack(action, feature_units)
            
            # Check if there is an enemy or if coordinates are within screen range
            if attack_action is None or min(attack_action.arguments[1]) < 0:
                return actions.FUNCTIONS.no_op(), 0
            self.current_micro_action = action
            return attack_action, action_index
        
        if action.id == avail_actions._MOVE_SCREEN.id and (self.last_action != avail_actions._ATTACK_SCREEN or 
                                                           self.current_strategy == avail_actions.retreat.id):
                if self._actions_in_queue(feature_units):
                    self.current_micro_action = action
                    return self.move_to_position(action), action_index

        return actions.FUNCTIONS.no_op(), 0  # No other matching actions, return no-op
        
    def set_mask(self) -> torch.Tensor:
        """
        Creates a mask for available actions based on the upper policy's strategy actions.

        This method initializes an action mask with zeros and sets the indices of available actions
        (those present in the upper policy's strategy actions) to one.

        Returns:
            torch.Tensor: A tensor representing the action mask, where available actions are marked with 1.
        """
        action_mask = torch.zeros(self.lower_action_space_size)
        available_actions_index = np.nonzero(np.isin(self.action_space, self.upper_policy.strategy_actions))[0]
        action_mask[available_actions_index] = 1
        return action_mask
    
    def map_action(self, action_index):
        """
        Maps an action index to its corresponding action.

        Args:
            action_index (int): The index of the action in the action space.

        Returns:
            The action corresponding to the given action index.
        """
        return avail_actions.ACTIONS[self.action_space[action_index]]
    
    def find_command_center(self, feature_units) -> Tuple[int, int]:
        """
        Finds the coordinates of the command center from a list of feature units.

        Args:
            feature_units (list): A list of units with their features.

        Returns:
            Tuple[int, int]: The (x, y) coordinates of the command center if found.
        """
        for unit in feature_units:
            if unit.unit_type == _COMMAND_CENTER:
                return (unit.x, unit.y)
    
    def find_point(self, feature_units):
        """
        Determines the point of interest based on the current map name.

        Args:
            feature_units (list): A list of feature units available in the current context.

        Returns:
            The point of interest based on the map name. If the map name is 'BuildMarines',
            it returns the result of the `find_command_center` method.
        """
        if self.map_name == 'BuildMarines':
            return self.find_command_center(feature_units)
    
    def find_unit(self, feature_units) -> int:
        """
        Selects a random unit from the player's units.

        Args:
            feature_units (list): A list of feature units from which player units are to be identified.

        Returns:
            int: The index of a randomly selected player unit.
        """
        player_units = self.get_player_units(feature_units)
        return np.random.randint(0, len(player_units))
    
    def count_current_enemies(self, feature_units) -> int:
        """
        Counts the number of enemy units in the given feature units.

        Args:
            feature_units (list): A list of feature units, where each unit is a dictionary
                      containing unit attributes, including 'alliance'.

        Returns:
            int: The count of enemy units.
        """
        count_enemy = 0
        for unit in feature_units:
            if unit['alliance'] == _PLAYER_ENEMY:
                count_enemy += 1
        return count_enemy
    
    def set_num_enemies(self, feature_units):
        """
        Sets the number of enemies based on the provided feature units.

        Args:
            feature_units (list): A list of feature units from which the number of enemies will be counted.
        """
        self.num_enemies = self.count_current_enemies(feature_units)
    
    def check_win(self, feature_units=None, units_defeated=None) -> bool:
        """
        Determines if the win condition has been met.

        For the 'FindAndDefeatZerglings' map, the win condition is met if the number of units defeated is greater than or equal to 25.
        For other maps, the win condition is met if the current number of enemies is greater than the initial number of enemies.

        Args:
            feature_units (optional): The current feature units in the game.
            units_defeated (optional): The number of units defeated so far.

        Returns:
            bool: True if the win condition is met, False otherwise.
        """
        # For FindAndDefeatZerglings map only
        if self.map_name == 'FindAndDefeatZerglings':
            return units_defeated >= 25
        # For other maps
        current_enemies = self.count_current_enemies(feature_units)
        return current_enemies > self.num_enemies

    def train_upper_policy(self, state, strategy, next_state, rewards, terminal) -> float:
        """
        Trains the upper policy network using the provided state, strategy, next state, rewards, and terminal flag.

        Args:
            state (object): The current state of the environment.
            strategy (object): The strategy or action taken in the current state.
            next_state (object): The state of the environment after the action is taken.
            rewards (float): The reward received after taking the action.
            terminal (bool): A flag indicating whether the episode has ended.

        Returns:
            float: The loss value after updating the Q-network.
        """
        loss = self.upper_policy.update_Q_network(state)
        return loss

    def train_lower_policy(self) -> float:
        """
        Trains the lower-level policy using the provided state, action, next state, rewards, and terminal flag.

        Args:
            state (object): The current state of the environment.
            action (object): The action taken in the current state.
            next_state (object): The state of the environment after the action is taken.
            rewards (float): The reward received after taking the action.
            terminal (bool): A flag indicating whether the episode has ended.

        Returns:
            float: The loss value after updating the lower-level policy.
        """
        loss = self.lower_policy.update()
        return loss

    def update_memory(self, state, action, next_state, reward, terminal, lower):
        """
        Updates the memory of the agent's policy based on the given parameters.

        Args:
            state (torch.Tensor): The current state of the environment.
            action (torch.Tensor): The action taken by the agent.
            next_state (torch.Tensor): The state of the environment after the action is taken.
            reward (float): The reward received after taking the action.
            terminal (bool): A flag indicating whether the episode has ended.
            lower (bool): A flag indicating whether to update the lower policy's memory or the upper policy's memory.
        """
        if lower:
            self.lower_policy.memory.push(state.unsqueeze(0), action, next_state.unsqueeze(0), reward, terminal)
        else:
            self.upper_policy.memory.push(state.unsqueeze(0), action, next_state.unsqueeze(0), reward, terminal)

    def update_lower_target_net(self):
        """
        Updates the target network of the lower-level policy.

        This method calls the `update_target_net` function of the `lower_policy` 
        to synchronize the target network with the current policy network. This 
        is typically done to stabilize training in reinforcement learning by 
        periodically updating the target network with the weights of the current 
        policy network.
        """
        self.lower_policy.update_target_net()
            
    def update_upper_target_net(self):
        """
        Updates the target network of the upper policy.

        This method calls the `update_target_net` method of the `upper_policy` 
        to synchronize the target network with the current policy network.
        """
        self.upper_policy.update_target_net()

    def _set_grid(self, screen_size):
        """
        Sets up the grid for the world map by dividing the screen into zones.
        Args:
            screen_size (int): The size of the screen or world map.
        Attributes:
            zones (list): A list of zone names generated from letters.
            grid (dict): A dictionary mapping zone names to Grid objects.
        The method performs the following steps:
        1. Calculates the size of each zone by dividing the screen size by the grid resize factor.
        2. Generates zone names using letters starting from 'A'.
        3. Creates a dictionary mapping each zone name to a Grid object, which defines the boundaries of each zone.
        """
        # Define the size of the world map
        zone_size = screen_size // self.grid_resize_factor  # Divide m x m map into n x n zones

        # Define the zones names
        self.zones = [
            chr(i) for i in range(ord('A'), ord('A') + self.grid_resize_factor**2)
        ]
        # Map the zones
        index = 0
        zones_obj = dict()
        for i in range(self.grid_resize_factor):
            for j in range(self.grid_resize_factor):
                zone_id = self.zones[index]
                index += 1
                zones_obj[zone_id] = Grid(
                    zone_id,
                    i * zone_size,
                    j * zone_size,
                    (i + 1) * zone_size,
                    (j + 1) * zone_size,
                )
        self.grid = zones_obj
        
    def action_available(self, action, available_actions_list):
        """
        Determines if a given action is available based on a list of available actions.
        Args:
            action (Action): The action to check for availability.
            available_actions_list (list): A list of available actions to check against.
        Returns:
            bool: True if the action is available in the provided lists, False otherwise.
        """
        for available_actions in available_actions_list:
            if not self._is_action_available(action, available_actions):
                return False
        
        # If the action is in available lists, check if it is in the actions list
        return self._is_in_action_list(action.id)
                    
    def get_player_units(self, feature_units):
        """
        Filters and returns the units that belong to the player.

        Args:
            feature_units (list): A list of units to filter from.

        Returns:
            list: A list of units that belong to the player.
        """
        player_units = [unit for unit in feature_units if unit.alliance == _PLAYER_SELF]
        return player_units
    
    def is_army_attacking(self, feature_units):
        """
        Determines if any player unit is currently attacking.

        This method checks the player's units to see if any of them have an active order and their weapon is on cooldown,
        which indicates that the unit is attacking.

        Args:
            feature_units (list): A list of feature units from which player units will be filtered.

        Returns:
            bool: True if any player unit is attacking, False otherwise.
        """
        player_units = self.get_player_units(feature_units)
        for unit in player_units:
            if unit.order_length > 0 and unit.weapon_cooldown > 0:
                return True
        return False
        
    def update_health(self, feature_units):
        """
        Updates the health information of the player's units.

        Args:
            feature_units (list): A list of feature units from which player units are extracted.

        Updates:
            self.prev_health (dict): A dictionary mapping unit tags to their health values.
        """
        units = self.get_player_units(feature_units)
        self.prev_health = {unit.tag: unit.health for unit in units}    
    
    def is_taking_damage(self, feature_units):
        """
        Determines if any player unit is currently taking damage.

        Args:
            feature_units (list): A list of feature units from which player units are extracted.

        Returns:
            bool: True if any player unit's current health is less than its previous health, indicating it is taking damage. False otherwise.
        """
        player_units = self.get_player_units(feature_units)
        for unit in player_units:
            if unit.tag in self.prev_health:
                return unit.health < self.prev_health[unit.tag]
        return False
                
    def find_target(self, feature_units):
        """
        Finds the first enemy unit on the screen from a list of feature units.

        Args:
            feature_units (list): A list of units with their features.

        Returns:
            tuple: The (x, y) coordinates of the first enemy unit found.
            None: If no enemy unit is found.
        """
        # Find the first enemy on the screen
        for unit in feature_units:
            if unit.alliance == _PLAYER_ENEMY:
                return (unit.x, unit.y)
        return None
                
    def move_to_position(self, action):
        """
        Moves the agent to a specified position on the screen based on the given action.

        Args:
            action (Action): An action object that contains the action name, which includes the zone ID.

        Returns:
            FunctionCall: A FunctionCall object that represents the move action to the specified screen coordinates.
        """
        zone = action.action_name[-1]   # Get the zone ID
        zone_to_move = self.grid[zone]
        x, y = (zone_to_move.x, zone_to_move.y)
        return actions.FUNCTIONS.Move_screen('queued', (x, y))

    def attack(self, action, feature_units):
        """
        Executes an attack action on a target if a valid target is found.

        Args:
            action (callable): A function that performs the attack action. It should accept two arguments:
                - A string indicating the timing of the action (e.g., 'now').
                - The coordinates of the target.
            feature_units (list): A list of feature units from which the target will be determined.

        Returns:
            The result of the action function if a target is found, otherwise None.
        """
        target_coor = self.find_target(feature_units)
        if target_coor is not None:
            return action('now', target_coor)
        return None

    def map_strategy_actions(self, id):
        """
        Maps the strategy actions for a given strategy ID.

        Args:
            id (int): The identifier of the strategy.

        Returns:
            list: A sorted list of micro actions associated with the strategy.
        """
        return sorted(self.strategies[id].micro_actions)

    def map_strategy(self, strategy_index):
        """
        Maps a given strategy index to its corresponding strategy.

        Args:
            strategy_index (int): The index of the strategy to be mapped.

        Returns:
            Strategy: The strategy corresponding to the given index.
        """
        return self.strategies[strategy_index]

    def train_mode(self):
        """
        Sets the training mode for both the upper and lower policies.

        This method enables the training mode for the upper policy and the lower policy,
        allowing them to update their parameters during the training process.
        """
        self.upper_policy.set_training_mode()
        self.lower_policy.set_training_mode()
        
    def eval_mode(self):
        """
        Sets the upper and lower policies to evaluation mode.

        This method changes the mode of both the upper and lower policies to evaluation mode,
        which is typically used during the testing or validation phase.
        """
        self.upper_policy.set_evaluation_mode()
        self.lower_policy.set_evaluation_mode()
        print("**** Set to evaluation mode: ON ****")
    
    def enable_pretrained_mode(self, freeze_layers):
        """
        Enables the pretrained mode for both upper and lower policies.

        Args:
            freeze_layers (bool): If True, the layers of the policies will be frozen, 
                                  meaning they will not be updated during training.
        """
        self.upper_policy.set_pretrained_mode(freeze_layers)
        self.lower_policy.set_pretrained_mode(freeze_layers)
    
    def save(self, step, path):
        """
        Save the current state of the agent's policies.

        This method attempts to save both the upper and lower policies of the agent
        to the specified path. If an error occurs during the saving process, an exception
        is raised with the error details.

        Args:
            step (int): The current step or iteration number to be saved.
            path (str): The directory path where the models should be saved.

        Raises:
            Exception: If there is an issue with saving the models, an exception
                       is raised with the corresponding error message.
        """
        try:
            self.upper_policy.save_model(step, path, self.map_name)
            self.lower_policy.save_model(step, path)
            print("MODEL SAVED SUCCESSFULLY")
        except Exception as e:
            raise Exception(f'Could not save agent due to {e}')
    
    def load(self, step, path):
        """
        Loads the agent's policies from the specified path at a given step.

        Args:
            step (int): The step number to load the model from.
            path (str): The file path where the model is stored.

        Returns:
            str: The name of the map loaded by the upper policy.

        Raises:
            Exception: If the agent could not be loaded due to an error.
        """
        try:
            map_name = self.upper_policy.load_model(step, path)
            _ = self.lower_policy.load_model(step, path)
            print('Agent loaded successfully')
            return map_name
        except Exception as e:
            raise Exception(f'Could not load agent due to {e}')

    @staticmethod
    def map_strategy_ids(strategies) -> List[int]:
        """
        Maps a list of strategy objects to their corresponding IDs.

        Args:
            strategies (list): A list of strategy objects, each containing an 'id' attribute.

        Returns:
            list: A list of IDs extracted from the strategy objects.
        """
        ids = []
        for strategy in strategies:
            ids.append(strategy.id)
        return ids

    @staticmethod
    def _is_action_available(action_id, available_actions) -> bool:
        """
        Check if a given action is available.

        Args:
            action_id (int): The ID of the action to check.
            available_actions (list): A list of available action IDs.

        Returns:
            bool: True if the action is available, False otherwise.
        """
        return action_id in available_actions

    @staticmethod
    def _is_in_action_list(action_id) -> bool:
        """
        Check if the given action ID is in the list of available action IDs.

        Args:
            action_id (int): The ID of the action to check.

        Returns:
            bool: True if the action ID is in the list of available action IDs, False otherwise.
        """
        return action_id in avail_actions.ACTIONS_IDS

    @staticmethod
    def map_action_old(action_index, action_list):
        """
        Maps an action index to its corresponding action in the action list.

        Args:
            action_index (int): The index of the action to be mapped.
            action_list (list): The list of possible actions.

        Returns:
            The action corresponding to the given index in the action list.
        """
        return action_list[action_index]

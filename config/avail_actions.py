from pysc2.lib import actions
import yaml
import os

with open(os.path.join(os.path.dirname(__file__), "params.yaml"), "r") as fp:
    params = yaml.safe_load(fp)['agent']

_GRID_RESIZE_FACTOR = params['grid_resize_factor']

_NO_OP = actions.FUNCTIONS.no_op#.id                     # 0
_SELECT_POINT = actions.FUNCTIONS.select_point#.id       # 2
_SELECT_ARMY = actions.FUNCTIONS.select_army#.id         # 7
_SELECT_UNIT = actions.FUNCTIONS.select_unit#.id         # 5
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen#.id     # 12
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen#.id         # 331
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen
_BUILD_COMMAND_CENTER = actions.FUNCTIONS.Build_CommandCenter_screen
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick

# STRATEGY CATEGORIES
_COMBAT = 'combat'
_COLLECT = 'collect'
_PRODUCTION = 'production'

class PositionActions:
    """
    PositionActions class represents an action associated with a specific position.

    Attributes
    ----------
        id (int): The identifier for the position.
        action_name (str): The name of the action associated with the position.

    Methods
    ----------
        __init__(id, action):
            Initializes the PositionActions instance with the given id and action name.
        
        __repr__():
            Returns a string representation of the PositionActions instance.
    """
    def __init__(self, id, action):
        self.id = id
        self.action_name = action
    def __repr__(self):
        return f'id: {self.id} action: {self.action_name}'

_move_actions = [
    PositionActions(
        _MOVE_SCREEN.id, 
        f'MOVE_TO_{chr(i)}'
    )
    for i in range(ord('A'), ord('A') + _GRID_RESIZE_FACTOR**2)
]

_build_barracks_positions = [
    PositionActions(
        _BUILD_BARRACKS.id, 
        f'BUILD_BARRACKS_AT_{chr(i)}'
    )
    for i in range(ord('A'), ord('A') + _GRID_RESIZE_FACTOR**2)
]

_build_supply_positions = [
    PositionActions(
        _BUILD_SUPPLY_DEPOT.id, 
        f'BUILD_SUPPLY_DEPOT_AT_{chr(i)}'
    )
    for i in range(ord('A'), ord('A') + _GRID_RESIZE_FACTOR**2)
]

ACTIONS = [
    _NO_OP,
    _SELECT_POINT,
    _SELECT_UNIT,
    _SELECT_ARMY,
    _ATTACK_SCREEN,
    _TRAIN_MARINE,
    _TRAIN_SCV,
] + _move_actions + _build_barracks_positions + _build_supply_positions

ACTIONS_IDS = [
    _NO_OP.id,
    _SELECT_POINT.id,
    _SELECT_UNIT.id,
    _SELECT_ARMY.id,
    _ATTACK_SCREEN.id,
    _MOVE_SCREEN.id,
    _BUILD_BARRACKS.id,
    _BUILD_SUPPLY_DEPOT.id,
    _BUILD_COMMAND_CENTER.id,
    _TRAIN_MARINE.id,
    _TRAIN_SCV.id,
]

############################################################################
class Strategy:
    """
    A class to represent a strategy with a name, id, and category, 
    and to manage its associated micro actions.
    
    Attributes
    -----------
    name : str
        The name of the strategy.
    id : int
        The unique identifier for the strategy.
    category : str
        The category to which the strategy belongs.
    micro_actions : list
        A list to store micro actions associated with the strategy.
        
    Methods
    --------
    add_actions(action):
        Adds a micro action to the strategy.
    __repr__():
        Returns a string representation of the strategy, including its name and the number of micro actions.
    """
    def __init__(self, name: str, id: int, category: str):
        self.name = name
        self.id = id
        self.category = category
        self.micro_actions = []
    
    def add_actions(self, action):
        """
        Adds a new action to the list of micro actions.

        Args:
            action: The action to be added to the micro actions list.
        """
        self.micro_actions.append(action)

    def __repr__(self):
        return f"{self.name} No. actions: {len(self.micro_actions)}"

def create_strategy(name, id, category, actions):
    """
    Creates a strategy with the given name, id, category, and actions.

    Args:
        name (str): The name of the strategy.
        id (int): The unique identifier for the strategy.
        category (str): The category of the strategy.
        actions (list): A list of action names to be added to the strategy.

    Returns:
        Strategy: An instance of the Strategy class with the specified attributes and actions.
    """
    strategy = Strategy(name, id, category)
    for action in actions:
        strategy.add_actions(ACTIONS.index(action))
    return strategy


# COMBAT
attack_actions = [
    _SELECT_ARMY,
    _ATTACK_SCREEN,
]
    
retreat_actions = [
    _NO_OP,
    _SELECT_ARMY,    
] + _move_actions

# PRODUCTION
collect_actions = [
   _SELECT_ARMY,
   _SELECT_UNIT,
] + _move_actions

train_actions = [
    _SELECT_POINT,
    _TRAIN_MARINE,
    _TRAIN_SCV,
]

build_actions = [
    _SELECT_POINT,
    _SELECT_ARMY,
] + _build_barracks_positions + _build_supply_positions

attack = create_strategy('attack', 0, _COMBAT, attack_actions)
retreat = create_strategy('retreat', 1, _COMBAT, retreat_actions)
collect = create_strategy('collect_gas', 2, _COLLECT, collect_actions)
build_units = create_strategy('build_units', 3, _PRODUCTION, build_actions)
train_units = create_strategy('train_units', 4, _PRODUCTION, train_actions)

STRATEGIES = [
    attack,             # COMBAT
    retreat,            # COMBAT
    collect,        # PRODUCTION
]

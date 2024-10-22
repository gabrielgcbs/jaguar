import json
import logging
from hierarchical_agent.agents.jaguar import Jaguar

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_action_space_size(space, level=0) -> int:
    """
    Calculate the size of the action space.

    Parameters:
    space (list): A list representing the action space.
    level (int, optional): The level of granularity for calculating the size. 
                           Defaults to 0. If level is 0, the function returns 
                           the length of the space. If level is greater than 0, 
                           the function calculates the size based on the 
                           micro_actions attribute of each strategy in the space.

    Returns:
    int: The size of the action space.
    """
    if level == 0:
        return len(space)
    else:
        size = 0
        for strategy in space:
            size += len(strategy.micro_actions)
        return size
    
def get_result(obs, map_name=None, units_defeated=None) -> int:
    """
    Computes episode result. Only usefull for combat maps
    return: win = 1
            draw = 0
            defeat = -1
    """
    # For FindAndDefeatZerglings map only
    if map_name == 'FindAndDefeatZerglings':
        if units_defeated >= 25:
            return 1        # Win
        if obs['player']['army_count'] == 0:
            return -1       # Defeat
        return 0
    
    # For other maps
    if obs['player']['army_count'] == 0:
        return -1       # Defeat
    
    enemies = [e for e in obs['feature_units'] if e.alliance == 4]
    return 1 if len(enemies) == 0 else 0

def compute_score(wins: int, draws: int, eps: int) -> float:
    """
    
    Compute the score as the formula number of wins + half the number of draws,  
    divided by the total number of evaluation episodes
    
    Parameters:
    wins: number of wins
    draws: number of draws
    eps: total number of episodes
    
    Returns:
    score [float]: the computed score
    
    Reference: https://ceur-ws.org/Vol-2862/paper28.pdf
    """
    score = (wins+(draws/2))/eps
    return score    

def get_score(rewards: list):
    """
    Calculate the score based on the list of rewards.

    Parameters:
    rewards [list]: A list of integers representing rewards. 

    Returns:
    score [float]: The computed score based on the number of wins, draws, and total episodes.

    Note:
        This function relies on an external function `compute_score` to calculate the final score.
    """
    num_wins =  rewards.count(1)
    num_draws = rewards.count(0)
    num_eps = len(rewards)
    score = compute_score(num_wins, num_draws, num_eps)
    return score

def save_results(
    agent: Jaguar, 
    step: int, 
    train_rewards: list,
    eval_rewards: list,
    train_ep_results: list,
    eval_ep_results: list,
    upper_loss: list,
    lower_loss: list,
    time_taken: float,
    map_name: str,
    model_path: str,
    train_results_path: str,
    eval_results_path: str,
    is_eval: bool,
    used_pretrained_model: bool,
) -> None:
    """
    Save the results of training and evaluation of the agent.
    
    Parameters:
    agent (Jaguar): The agent being trained or evaluated.
    step (int): The current step or episode number.
    train_rewards (list): List of rewards obtained during training.
    eval_rewards (list): List of rewards obtained during evaluation.
    train_ep_results (list): List of episode results during training.
    eval_ep_results (list): List of episode results during evaluation.
    upper_loss (list): List of upper loss values during training.
    lower_loss (list): List of lower loss values during training.
    time_taken (float): Time taken for the training/evaluation in minutes.
    map_name (str): Name of the map used for training/evaluation.
    model_path (str): Path to save the trained model.
    train_results_path (str): Path to save the training results.
    eval_results_path (str): Path to save the evaluation results.
    is_eval (bool): Flag indicating if the current run is an evaluation.
    used_pretrained_model (bool): Flag indicating if a pretrained model was used.
        
    Returns:
    None
    """
    filename = 'results_' + str(step) + '.json'
    
    results_scheme = {
        'run': step,
        'rewards': None,
        'ep_results': None,
        'upper_loss': None,
        'lower_loss': None,
        'score': None,
        'trained_at': map_name,
        'evaluated_at': None,
        'original_map': None,
        'time': time_taken, # in min
    }
    
    eval_score = get_score(eval_ep_results)
    
    # TODO: CHANGE MAP_NAME
    if used_pretrained_model:
        results_scheme['original_map'] = map_name
    
    eval_results = results_scheme.copy()    
    eval_results['rewards'] = eval_rewards
    eval_results['ep_results'] = eval_ep_results
    eval_results['score'] = eval_score
    eval_results['evaluated_at'] = agent.map_name

    if not is_eval:
        train_score = get_score(train_ep_results)
        train_results = results_scheme.copy()
        train_results['rewards'] = train_rewards
        train_results['ep_results'] = train_ep_results
        train_results['upper_loss'] = upper_loss
        train_results['lower_loss'] = lower_loss
        train_results['score'] = train_score
        
        agent.save(step, model_path)
        logger.info("SUCCESSFULLY SAVED AGENT")

        # Saving training results
        with open(train_results_path + filename, 'w') as fp:
            json.dump(train_results, fp, indent=4)
            
    # Saving evaluation results
    with open(eval_results_path + filename, 'w') as fp:
        json.dump(eval_results, fp, indent=4)
    logger.info("RESULTS SAVED")
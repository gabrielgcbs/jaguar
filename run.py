import torch
import yaml
import os
import time
from copy import copy
from hierarchical_agent.environment import StarCraft2Env
from hierarchical_agent.agents.jaguar import Jaguar
from utils import get_result, save_results

# Load params
with open(os.path.join(os.path.dirname(__file__), "config", "params.yaml"), "r") as fp:
    params = yaml.safe_load(fp)

with open(
    os.path.join(os.path.dirname(__file__), "config", "avail_maps.yaml"), "r"
) as fp:
    maps = yaml.safe_load(fp)

agent_params = params["agent"]
run_params = params["run"]
env_params = params["env"]
agent_interface_params = params["agent_interface_format"]

# Define constants
_SCREEN_SIZE = agent_interface_params["screen_size"]
_MAP_NAME = env_params["map_name"]

_GRID_RESIZE_FACTOR = agent_params["grid_resize_factor"]
_BATCH_SIZE = agent_params["batch_size"]
_FEATURES_INDEX = agent_params["features_index"]

_RUN = run_params["run"]
_EPISODES = run_params["episodes"]

_EVAL_WINDOW = run_params["eval_window"]
_NUM_EVALS = run_params["num_evals"]

_TRIES = run_params["tries"]
_STEPS_TO_CHANGE_STRATEGY = run_params["steps_to_change_strategy"]
_STEPS_TO_TRAIN_MACRO = run_params["steps_to_train_macro"]
_EVAL_MODE = run_params["eval_mode"]
_USE_PRETRAINED_MODEL = run_params["use_pretrained_model"]
_FREEZE_LAYERS = run_params["freeze_layers"]
_RUN_TO_LOAD = run_params["load_from_run"]
_MODEL_PATH = run_params["model_save_path"]
_TRAIN_RESULTS_PATH = run_params["results_train_save_path"]
_EVAL_RESULTS_PATH = run_params["results_eval_save_path"]


def run_loop():
    saved_results = False
    upper_losses = []
    lower_losses = []
    rewards = []
    eval_rewards = []

    ep_results = []
    eval_ep_results = []

    mid_input_dim = (len(_FEATURES_INDEX), _SCREEN_SIZE, _SCREEN_SIZE)
    lower_input_dim = len(_FEATURES_INDEX) * _SCREEN_SIZE * _SCREEN_SIZE

    agent = Jaguar(
        mid_input_dim,
        lower_input_dim,
        screen_size=_SCREEN_SIZE,
        grid_resize_factor=_GRID_RESIZE_FACTOR,
        maps=maps,
        current_map=_MAP_NAME,
        batch_size=_BATCH_SIZE,
    )

    map_trained = _MAP_NAME
    if _EVAL_MODE:  # Inference mode
        map_trained = agent.load(_RUN_TO_LOAD, _MODEL_PATH)
        agent.eval_mode()

    if _USE_PRETRAINED_MODEL:
        agent.load(_RUN_TO_LOAD, _MODEL_PATH)
        agent.enable_pretrained_mode(_FREEZE_LAYERS)
        map_trained = agent.map_name

    episodes = (
        range(_EPISODES + _NUM_EVALS * _EVAL_WINDOW)
        if not _EVAL_MODE
        else range(_EPISODES)
    )
    train_episodes = 0
    eval_episodes = 0
    episode = 0

    num_tries = _TRIES
    current_try = 0
    won = False
    loaded_env = False
    test_mode = False
    sc2_env = StarCraft2Env(_FEATURES_INDEX)

    eps_to_start_test = (_EPISODES) // _NUM_EVALS if not _EVAL_MODE else -1
    env = None  # Initialize env variable
    start = time.time()

    print(f"Run {_RUN}")
    # Attempt to open the game. Sometimes the connection is lost
    while current_try < num_tries:
        print(f"Opening the game... try {current_try+1}")
        try:
            if not loaded_env:
                env = sc2_env.create_env(agent_interface_params, env_params)
                loaded_env = True

            while episode < len(episodes):
                print(f"Episode {episode+1}")
                actions_taken = 0
                reward = 0
                cum_reward = 0
                units_defeated = 0
                done = False
                count_strategies = 0

                if (train_episodes == eps_to_start_test) or (
                    test_mode and eval_episodes < _EVAL_WINDOW
                ):
                    print("***** EVALUATION MODE ENABLED *****")
                    test_mode = True
                    eval_episodes += 1
                    train_episodes = 0
                    agent.eval_mode()
                elif not _EVAL_MODE:
                    print("TRAINING...")
                    test_mode = False
                    eval_episodes = 0
                    train_episodes += 1
                    agent.train_mode()

                obs = env.reset()
                obs = obs[0]
                state = sc2_env.preprocess_observation(obs)

                while not done:
                    if actions_taken % _STEPS_TO_CHANGE_STRATEGY == 0:
                        # TRAIN UPPER AGENT
                        if not (_EVAL_MODE or test_mode) and actions_taken > 0:
                            print("Training strategy...")
                            agent.update_memory(
                                states_pair[0],
                                mid_strategy_index,
                                states_pair[1],
                                torch.tensor([upper_reward]),
                                torch.tensor([done]),
                                lower=False,
                            )
                            if (count_strategies + 1) % _STEPS_TO_TRAIN_MACRO == 0:
                                agent.update_upper_target_net()
                                count_strategies = 0
                            upper_loss = agent.train_upper_policy(
                                states_pair[0],
                                mid_strategy_index,
                                states_pair[1],
                                torch.tensor([upper_reward]),
                                torch.tensor([done]),
                            )
                            if upper_loss is not None:
                                upper_losses.append(
                                    upper_loss.item()
                                )  # from tensor to float
                            agent.upper_policy.update_epsilon()

                        # Choose strategy
                        mid_strategy_index = agent.choose_mid_strategy(state)
                        _ = agent.map_strategy(mid_strategy_index)
                        upper_reward = 0
                        count_strategies += 1

                    # if not agent.mid_policy.goal_done:
                    available_actions = obs.observation["available_actions"]
                    feature_units = obs.observation["feature_units"]

                    agent.set_num_enemies(feature_units)

                    micro_action, micro_action_index = agent.choose_micro_action(
                        state, available_actions, feature_units
                    )
                    micro_action_index = torch.tensor(
                        [[micro_action_index]], dtype=torch.long
                    )  # change from int to tensor
                    # print(micro_action)
                    obs = env.step([micro_action])[0]
                    next_state = sc2_env.preprocess_observation(obs)
                    done = torch.tensor(obs.last())
                    # Immediate reward
                    reward = obs.reward

                    if _MAP_NAME == "FindAndDefeatZerglings" and reward == 1:
                        units_defeated += 1

                    reward -= 0.1
                    cum_reward += reward
                    upper_reward += reward
                    states_pair = (state, next_state)
                    # Go for the next state
                    state = copy(next_state)

                    next_feature_units = obs.observation["feature_units"]

                    agent.update_health(next_feature_units)
                    if not done:
                        won = (
                            agent.check_win(feature_units=next_feature_units)
                            if _MAP_NAME != "FindAndDefeatZerglings"
                            else agent.check_win(units_defeated=units_defeated)
                        )

                    # TRAIN LOWER AGENT
                    if not (_EVAL_MODE or test_mode):
                        agent.update_memory(
                            states_pair[0],
                            micro_action_index,
                            states_pair[1],
                            torch.tensor([reward]),
                            done,
                            lower=True,
                        )
                        # Perform one step of the optimization (on the policy network)
                        lower_loss = agent.train_lower_policy()
                        agent.update_lower_target_net()
                        if lower_loss is not None:
                            lower_losses.append(lower_loss.item())

                    actions_taken += 1

                    if won:
                        # Add the value 1 to count as a victory
                        (
                            ep_results.append(1)
                            if not (_EVAL_MODE or test_mode)
                            else eval_ep_results.append(1)
                        )
                        units_defeated = 0

                    if done:
                        print("done")

                if test_mode or _EVAL_MODE:
                    eval_rewards.append(
                        cum_reward.item()
                    )  # converting from np.int32 to int
                    print(f"Episode reward: {cum_reward}")
                    eval_ep_results.append(
                        get_result(obs.observation, _MAP_NAME, units_defeated)
                    )
                else:
                    rewards.append(cum_reward.item())  # converting from np.int32 to int
                    print(f"Episode reward: {cum_reward}")
                    ep_results.append(
                        get_result(obs.observation, _MAP_NAME, units_defeated)
                    )

                episode += 1

            env.close()
            end = time.time()
            time_taken = round((end - start) / 60, 2)
            loaded_env = True
            save_results(
                agent,
                _RUN,
                rewards,
                eval_rewards,
                ep_results,
                eval_ep_results,
                upper_losses,
                lower_losses,
                time_taken,
                map_trained,
                _MODEL_PATH,
                _TRAIN_RESULTS_PATH,
                _EVAL_RESULTS_PATH,
                _EVAL_MODE,
                _USE_PRETRAINED_MODEL,
            )
            saved_results = True

            print("***Finished***")
            print(f"Time taken: {round(time_taken/60, 2)} h")
            break

        except Exception as e:
            current_try += 1
            loaded_env = False
            print(f"Exception occurred during episode {episode+1}: {e}")
            if env:
                env.close()
            continue

    try:
        if not saved_results:
            end = time.time()
            time_taken = round((end - start) / 60, 2)
            save_results(
                agent,
                _RUN,
                rewards,
                eval_rewards,
                ep_results,
                eval_ep_results,
                upper_losses,
                lower_losses,
                time_taken,
                map_trained,
                _MODEL_PATH,
                _TRAIN_RESULTS_PATH,
                _EVAL_RESULTS_PATH,
                _EVAL_MODE,
                _USE_PRETRAINED_MODEL,
            )
    except Exception as e:
        raise Exception(f"Could not save game results: {e}")


if __name__ == "__main__":
    run_loop()

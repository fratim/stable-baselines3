import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gym
import numpy as np

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3 import video

import wandb

import time

def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    video_recorder = None,
    step_training = -1,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """

    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    x_positions_last = []

    positions_0_means = []
    positions_1_means = []
    positions_2_means = []

    positions_0_medians = []
    positions_1_medians = []
    positions_2_medians = []

    positions_0_stds = []
    positions_1_stds = []
    positions_2_stds = []

    positions_0_maxs = []
    positions_1_maxs = []
    positions_2_maxs = []



    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)

    assert n_envs == 1
    video_recorder.init(enabled=True)

    step = 0

    positions_0 = []
    positions_1 = []
    positions_2 = []

    while (episode_counts < episode_count_targets).any():

        actions, states = model.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic)

        observations, rewards, dones, infos = env.step(actions)

        positions_0.append(observations[0][0])
        positions_1.append(observations[0][1])
        positions_2.append(observations[0][2])

        current_rewards += rewards
        current_lengths += 1

        if step % 2 == 0:
            video_recorder.record(env.envs[0].env)
            print(f"video step {step} saved")
        step += 1

        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:

                    video_recorder.save(f'{episode_counts.sum()}.mp4')
                    video_recorder.init(enabled=True)

                    x_positions_last.append(infos[0]["x_position"])

                    positions_0_means.append(np.mean(positions_0))
                    positions_0_medians.append(np.median(positions_0))
                    positions_0_stds.append(np.std(positions_0))
                    positions_0_maxs.append(np.max(positions_0))

                    positions_1_means.append(np.mean(positions_1))
                    positions_1_medians.append(np.median(positions_1))
                    positions_1_stds.append(np.std(positions_1))
                    positions_1_maxs.append(np.max(positions_1))

                    positions_2_means.append(np.mean(positions_2))
                    positions_2_medians.append(np.median(positions_2))
                    positions_2_stds.append(np.std(positions_2))
                    positions_2_maxs.append(np.max(positions_2))

                    positions_0 = []
                    positions_1 = []
                    positions_2 = []

                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0



        if render:
            env.render()

    video_recorder.save(f'{step_training}.mp4')

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"

    # wandb.log({
    #     "eval/position_0_mean": np.mean(positions_0_means),
    #     "eval/position_0_median": np.mean(positions_0_medians),
    #     "eval/position_0_std": np.mean(positions_0_stds),
    #     "eval/position_0_max": np.mean(positions_0_maxs),
    #     "eval/position_1_mean": np.mean(positions_1_means),
    #     "eval/position_1_median": np.mean(positions_1_medians),
    #     "eval/position_1_std": np.mean(positions_1_stds),
    #     "eval/position_1_max": np.mean(positions_1_maxs),
    #     "eval/position_2_mean": np.mean(positions_2_means),
    #     "eval/position_2_median": np.mean(positions_2_medians),
    #     "eval/position_2_std": np.mean(positions_2_stds),
    #     "eval/position_2_max": np.mean(positions_2_maxs),
    # }
    # )

    if return_episode_rewards:
        return episode_rewards, x_positions_last
    return mean_reward, std_reward

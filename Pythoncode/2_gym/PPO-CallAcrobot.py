import os
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw

from IPython.display import display, Markdown

from stable_baselines3 import PPO

from stable_baselines3.common import results_plotter
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.monitor import load_results

import torch
import gym
import gymnasium as gym

import warnings

from time import time

ALG = 'PPO'
ENV = 'Acrobot'
ENVNAME = f'{ENV}-v1'
PROCNUM = min(24, os.cpu_count())

warnings.filterwarnings("ignore", category=DeprecationWarning)

enableTraining = 1

def agent_eval(agent, n_eval_episodes=25, seed=42, verbose=1):
    eval_env = make_vec_env(ENVNAME, n_envs=PROCNUM, seed=seed)
    reward_length = evaluate_policy(agent,
                                    eval_env, 
                                    return_episode_rewards=True,
                                    n_eval_episodes=n_eval_episodes,
                                    deterministic=True)
    for i, p in enumerate(["mean", "length"]):
        mean = np.mean(reward_length[i])
        if verbose:
            sem = st.sem(reward_length[i])
            mean_CI = st.t.interval(0.95, df=len(reward_length[i])-1, loc=mean, scale=sem) 
        
            print(f"mean_{p}={mean:.3f}, SEM={sem}, CI=[{mean_CI[0]:.3f}:{mean_CI[1]:.3f}]")

    return np.mean(reward_length[0])

def agent_eval_single(agent):
    env = gym.make(ENVNAME,render_mode='human')
    #reward_length = evaluate_policy(agent,
    #                                env, 
    #                                return_episode_rewards=True,
    #                                n_eval_episodes=n_eval_episodes,
    #                                deterministic=True)
    for episode in range(100):
        obs,_ = env.reset()
        for i in range(1000):
            action, _state = agent.predict(obs)
            obs, reward, done, info,_ = env.step(action)
            env.render()
            if done:
              obs,_ = env.reset()



def _save_frames_as_gif(frames, path="./", filename="gym_animation.gif", fps=60):
    imageio.mimwrite(os.path.join(path, filename), frames, fps=fps)

def _add_text_info(frame, episode_num, step, reward):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), 
                f"Episode: {episode_num+1}\nStep: {step}\nCurrent reward: {reward}", 
                fill=text_color)
    return im

def save_trained_agent_gif(agent, path="./", filename="trained_agent.gif", n=1, fps=60):
    env = gym.make(ENVNAME, render_mode="rgb_array")
    
    frames = []
    for i in range(n):
        R = 0
        observation, _ = env.reset()        
        for t in range(500):
            action, _ = agent.predict(observation, None, None, True)
            observation, reward, terminated, _, _ = env.step(action)
            R += reward
            frame = env.render()
            frames.append(_add_text_info(frame, episode_num=i, step=t, reward=R))
            if terminated:
                frames.extend([_add_text_info(frame, episode_num=i, step=t, reward=R)]*25)
                break
    
    env.close()

    _save_frames_as_gif(frames, path=path, filename=filename, fps=fps)

class RandomAgent(object):
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self._rng = np.random.default_rng(None)

    def predict(self, observations, state, episode_start, deterministic):
        rng = self.rng if deterministic else self._rng
        if len(observations.shape) == 1:
            return rng.integers(4, size=1)[0], None
        else: 
            return rng.integers(4, size=observations.shape[0]), None

def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    fig = plt.figure(title, figsize=(10, 5))
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    min_x, max_x = min(x), max(x)
    plt.scatter(x, y, s=1)
    y = moving_average(y, window=100)
    # Truncate x
    x = x[len(x) - len(y):]

    plt.plot(x, y, color="black")
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " (Smoothed)")
    plt.xlim(min_x, max_x)
    plt.show()
    
class SaveOnBestTrainingRewardCallback(BaseCallback):
    # идея в том, что во время ролоутов монитор скидывает результаты в заданную папку
    # в конце ролаута чекаются только новые эпизоды (за последний ролаут)
    # если метрика повышается, то сохраняем новые веса
    # после последнего шага нет возможности сделать такой же шаг (хотя можно выкруутиться путем нового ролаута), поэтому
    # просто сравниваются итоговые веса и промежуточные (тут как-то мне не нравится что-то, но в первом приближении пусть так будет)
    def __init__(self, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf
        self.done_eps = 0

    def _check(self):
        results = load_results(self.log_dir)
        new_eps = len(results) - self.done_eps
        self.done_eps = len(results)
        x, y = ts2xy(results, 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-new_eps:])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose > 0:
                    print(f"{GPUorCPU}-{ALG}-{ENVNAME}_New best mean reward: {mean_reward:.2f}, eval_episodes: {new_eps}, total_steps: {self.num_timesteps}, total_episodes: {self.done_eps}")
                    self.model.save(self.save_path)

    def _init_callback(self) -> None:
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # if self.n_calls % self.check_freq == 0:
        #   self._check()
        return True
    
    def _on_training_end(self) -> None:
        interim_mean = agent_eval(self.model.load(self.save_path), verbose=0)
        final_mean = agent_eval(self.model, verbose=0)

        if interim_mean > final_mean:
            print(f"Interim model was better.")
            self.model.set_parameters(self.save_path)

        self.model.save(self.log_dir)

    def _on_rollout_end(self) -> None:
        self._check()


#agent_eval(RandomAgent(42));

#save_trained_agent_gif(RandomAgent(42), filename="random_agent.gif", n=5, fps=60)
#display(Markdown(f"<img src='random_agent.gif'>"))

model_dir = f"../models/01_{ENV}/{ALG}-{ENV}-{PROCNUM}-{TOTAL_STEP}-v1_baseline"

env = make_vec_env(ENVNAME, n_envs=PROCNUM, monitor_dir=model_dir)
model = PPO("MlpPolicy", env, device=GPUorCPU, verbose=0, tensorboard_log="../tb_logs/")

if enableTraining == 1:
    
    exist = os.path.exists('{}.zip'.format(model_dir))
    if exist == True:
        model = PPO.load('{}.zip'.format(model_dir), env)

    t1 = time()

    callback = SaveOnBestTrainingRewardCallback(log_dir=model_dir)
    model.learn(total_timesteps=int(TOTAL_STEP), callback=callback, tb_log_name="first_run_2m")
    t2 = time()
    elapsed = t2-t1;
    print('elapsed=%f' % elapsed)

    plot_results(model_dir)

    #agent_eval(model)
    agent_eval_single(model)

    save_trained_agent_gif(model, filename=f"{ALG}_{ENV}_baseline.gif", n=5, fps=60)

else:
    env = gym.make(ENVNAME, render_mode='human')
    model = PPO.load('{}.zip'.format(model_dir), env)
    agent_eval_single(model)






























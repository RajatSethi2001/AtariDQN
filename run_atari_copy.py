"""Train PPO with any Atari2600 env.

Author: Jinwoo Park
Email: www.jwpark.co.kr@gmail.com
"""
import numpy as np
import gym
import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from ray.tune.logger import pretty_print
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models import ModelCatalog

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv3D, MaxPool2D, MaxPool3D, AvgPool2D, AvgPool3D, Flatten, Dropout

class KerasModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(KerasQModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.inputs = Input(shape=obs_space.shape)
        conv_1 = Conv2D(16, (3, 3), strides=(2, 2), activation="relu")(self.inputs)
        pool_1 = AvgPool2D(pool_size=(2, 2))(conv_1)
        drop_1 = Dropout(0.1)(pool_1)
        conv_2 = Conv2D(16, (3, 3), strides=(2, 2), activation="relu")(drop_1)
        pool_2 = MaxPool2D(pool_size=(2,2))(conv_2)
        drop_2 = Dropout(0.1)(pool_2)
        flatten = Flatten()(drop_2)
        dense_1 = Dense(64, activation="relu")(flatten)
        dense_2 = Dense(64, activation="relu")(dense_1)
        drop_3 = Dropout(0.2)(dense_2)
        layer_out = Dense(num_outputs, activation="relu")(drop_3)
        value_out = Dense(1, activation=None)(drop_3)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
	
    def forward(self, input_dict, state, seq_lens):
        input_values = tf.cast(input_dict["obs"], tf.float32)
        model_out, self._value_out = self.base_model(input_values)
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

class KerasQModel(DistributionalQTFModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        super(KerasQModel, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        self.inputs = Input(shape=obs_space.shape)
        conv_1 = Conv2D(4, (3, 3), strides=(1, 1), activation="relu")(self.inputs)
        pool_1 = AvgPool2D(pool_size=(2, 2))(conv_1)
        drop_1 = Dropout(0.1)(pool_1)
        conv_2 = Conv2D(16, (3, 3), strides=(4, 4), activation="relu")(drop_1)
        pool_2 = MaxPool2D(pool_size=(2, 2))(conv_2)
        drop_2 = Dropout(0.1)(pool_2)
        flatten = Flatten()(drop_2)
        dense_1 = Dense(64, activation="relu")(flatten)
        dense_2 = Dense(64, activation="relu")(dense_1)
        drop_3 = Dropout(0.2)(dense_2)
        layer_out = Dense(num_outputs, activation=None)(drop_3)
        pre_value = Dense(128, activation="relu")(drop_3)
        value_out = Dense(1, activation=None)(pre_value)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])
	
    def forward(self, input_dict, state, seq_lens):
        input_values = tf.cast(input_dict["obs"], tf.float32)
        model_out, self._value_out = self.base_model(input_values)
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
          
class ModelTrainer:
    def __init__(self, model=0, env="Breakout-v0", n_workers=4):
        ray.init()
        ModelCatalog.register_custom_model("keras_model", KerasModel)
        ModelCatalog.register_custom_model("keras_q_model", KerasQModel)
        if model == 0:
            model_type = "keras_model"
        else:
            model_type = "keras_q_model"

        config = {
            "env": env,
            "lr": 1e-3,
            "num_workers": n_workers,
            "model": {"custom_model": model_type},
            "framework": "tf2",
        }
        self.train_config = ppo.DEFAULT_CONFIG.copy()
        self.train_config.update(config)

    def train(self, n_iters):
        """Train the agent with n_iters iterations."""
        print("Start training.")
        agent = ppo.PPOTrainer(config=self.train_config, env=self.train_config["env"])
        for _ in range(n_iters):
            result = agent.train()
            print(pretty_print(result))
            checkpoint_path = agent.save()
            print(f"Checkpoint saved in {checkpoint_path}")

    def continue_train(self, n_iters, checkpoint_path):
        """Train the agent from a checkpoint with n_iters iterations."""
        print("Continue training.")
        agent = ppo.PPOTrainer(config=self.train_config, env=self.train_config["env"])
        agent.restore(checkpoint_path)
        for _ in range(n_iters):
            result = agent.train()
            print(pretty_print(result))
            checkpoint_path = agent.save()
            print(f"Checkpoint saved in {checkpoint_path}")        

    def evaluate(self, checkpoint_path, render=False):
        print("Start evaluation.")
        """Evaluate the agent with a single iteration."""
        eval_config = self.train_config
        eval_config["evaluation_interval"] = None
        eval_config["num_workers"] = 0
        eval_config["explore"] = False
        
        agent = ppo.PPOTrainer(config=eval_config, env=eval_config["env"])
        agent.restore(checkpoint_path)
        print(f"Checkpoint loaded from {checkpoint_path}")

        env = gym.make(eval_config["env"], render_mode="human" if render else None)
        env = wrap_deepmind(env)
        obs = env.reset()
        print(f"Created env for {eval_config['env']}")

        done = False
        score = 0.0
        while not done:
            action = agent.compute_single_action(obs)
            obs, reward, done, _ = env.step(action)
            score += reward

        env.close()
        print(f"Evaluation score: {score}")

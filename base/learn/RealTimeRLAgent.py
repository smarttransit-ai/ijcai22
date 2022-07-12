import ast
import os
import random
from collections import deque
from datetime import datetime, timedelta

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from common.constant.constants import RESULT_DIR
from common.util.common_util import logger, create_dir


class RealTimeRLAgentBase(object):
    def __init__(self, state_space, memory_file_name=None):
        self.state_space = state_space
        self.memory_file_name = memory_file_name
        if self.memory_file_name is not None:
            create_dir(self.memory_file_name)
            self.memory_file = open(self.memory_file_name, "w+")
            self.memory_file.write("state;action;reward;done\n")
            self.memory_file.close()
        else:
            self.memory_file = None
        self.batch_size = 32
        self.epsilon = 1
        self.epsilon_min = .01
        self.epsilon_decay = .995
        self.memory = deque(maxlen=100000)
        self.model = None

    def build_model(self):
        raise NotImplementedError

    def build_model_by_experiences(self, experience_file, idx=-1):
        raise NotImplementedError

    def load_model(self, model_file_name, weight_file_name=None):
        raise NotImplementedError

    def save_model(self, model_file_name, weight_file_name=None):
        raise NotImplementedError

    def remember(self, state, action, reward, done=False):
        state = np.reshape(state, (1, self.state_space))
        self.memory.append((state, action, reward, done))
        # store the data to memory
        if self.memory_file is not None:
            # only write to file if memory file already presents
            self.memory_file = open(self.memory_file_name, "a+")
            self.memory_file.write(f"{list(state[0])};{action};{reward};{done}\n")
            self.memory_file.flush()
            os.fsync(self.memory_file.fileno())
            self.memory_file.close()
        self.replay()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        mini_batch = self.memory
        states = np.array([i[0] for i in mini_batch])
        rewards = np.array([i[2] for i in mini_batch])

        states = np.squeeze(states)
        self.model.fit(states, rewards)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        features, targets = state
        if np.random.rand() <= self.epsilon:
            idx = random.randint(0, len(features) - 1)
        else:
            choices = self.model.predict(features)
            idx = np.argmax(choices)
        return features[idx], targets[idx]

    def act_eval(self, state):
        features, targets = state
        choices = self.model.predict(features)
        idx = np.argmax(choices)
        return features[idx], targets[idx]

    def train(self, env_dict, args):
        start = datetime.now()
        _loss = []
        all_dates = []
        _start_date = datetime(2021, 1, 1, 0, 0, 0)
        _end_date = datetime(2021, 7, 1, 0, 0, 0)
        while _start_date != _end_date:
            _date = _start_date.strftime("%Y-%m-%d")
            if _date != "2021-04-04":
                all_dates.append(_date)
            _start_date = _start_date + timedelta(days=1)

        if not isinstance(args.features, list):
            features = ast.literal_eval(args.features)
        else:
            features = args.features

        features_size = len(features)
        all_features = ['pdh', 'ph', 'dh', 'h', 'et', 'ed', 'ts', 'er']
        if features_size == 0:
            str_features = all_features
        else:
            str_features = []
            for k, feature in enumerate(all_features):
                if k in features:
                    str_features.append(feature)

        logger.info(f"Training with features: {str_features}")

        episode = int(args.no_of_episodes)
        train_env_type = str(args.train_env_type)

        env_class = env_dict[train_env_type]
        env = env_class(args)

        idx = 0
        random.seed(int(args.random_seed))
        random.shuffle(all_dates)
        dates = all_dates[:episode]

        create_dir(f"{RESULT_DIR}/{args.agency}/models")

        for e, date in enumerate(dates):
            # each day is an episode
            args.date = date
            env = env_class(args)
            state = env.reset()
            score = 0
            done = False
            while not done:
                state_feature, action = self.act(state)
                reward, next_state, done = env.step(action)
                score += reward
                self.remember(state_feature, action, reward, done=done)
                state = next_state
            if done:
                print(f"episode: {e + 1}/{episode}, score: {score}, date: {date}")
            _loss.append(score)
            idx += 1
            self.save_model(
                f"{RESULT_DIR}/{args.agency}/models/{env.prefix}_model_{idx}.h5",
                f"{RESULT_DIR}/{args.agency}/models/{env.prefix}_weights_{idx}.h5"
            )
        self.save_model(
            f"{RESULT_DIR}/{args.agency}/models/{env.prefix}_model.h5",
            f"{RESULT_DIR}/{args.agency}/models/{env.prefix}_weights.h5"
        )
        print("Time taken: ", (datetime.now() - start).total_seconds())


class RealTimeRLAgentNN(RealTimeRLAgentBase):
    def __init__(self, state_space, memory_file_name=None):
        super(RealTimeRLAgentNN, self).__init__(state_space, memory_file_name)
        self.gamma = .95
        self.learning_rate = 0.001
        self.model = self.build_model()
        logger.info("running RL-Agent (NN)")

    def build_model(self):
        """
        :return: build the model and return model
        """
        model = Sequential()
        model.add(InputLayer(input_shape=(self.state_space,)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def build_model_by_experiences(self, experience_file, idx=-1):
        self.model = self.build_model()
        import pandas as pd
        df = pd.read_csv(experience_file, delimiter=";")
        df["state"] = df["state"].apply(ast.literal_eval)
        x = df["state"].to_list()
        y = df["reward"].to_list()
        size = len(x)
        if idx > 0:
            step_size = int(size / 5)
            x = x[idx * step_size:(idx + 1) * step_size]
            y = y[idx * step_size:(idx + 1) * step_size]
        self.model.fit(x, y, epochs=1, verbose=0)

    def replay(self):
        if len(self.memory) < 2:
            return
        mini_batch = self.memory
        states = np.array([i[0] for i in mini_batch])
        rewards = np.array([i[2] for i in mini_batch])

        states = np.squeeze(states)
        self.model.fit(states, rewards, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, model_file_name, weight_file_name=None):
        """
        :param model_file_name: model file name
        :param weight_file_name: weight file name
        Load the model and load the weights for the model
        """
        logger.info(f"Loading Model from : {model_file_name}, Loading Weights from : {weight_file_name}")
        self.model = load_model(model_file_name)
        self.model.load_weights(weight_file_name)

    def save_model(self, model_file_name, weight_file_name=None):
        """
        :param model_file_name: model file name
        :param weight_file_name: weight file name

        save the model and save the weights of the model
        """
        if self.model is not None:
            self.model.save(model_file_name)
            self.model.save_weights(weight_file_name)

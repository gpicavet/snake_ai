import random
from collections import deque

import numpy as np
import torch

from Game import Game, Direction
from model import LinearQNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
BOARD_SIZE = 12

dirMap = {1: Direction.LEFT, 2: Direction.RIGHT, 0: Direction.UP}

#if torch.backends.mps.is_available():
#    torch.set_default_device('mps')

class Agent:

    def __init__(self, input_size: int, output_size: int):
        self.epsilon = 1
        self.n_games = 0
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = LinearQNet(input_size, 256, output_size)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        if self.epsilon < np.random.random():
            #print("predict")
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            return torch.argmax(prediction).item()
        else:
            #print("random")
            return random.randint(0, 2)

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    game = Game(BOARD_SIZE, BOARD_SIZE)
    game.start()

    agent = Agent(len(game.get_state()), len(dirMap))

    while True:
        # get old state
        state_old = np.array(game.get_state(), dtype=float)
        #print(state_old)

        # get move
        move = agent.get_action(state_old)

        # perform move and get new state
        game.dir(dirMap[move])
        reward, done, score = game.step()
        state_new = np.array(game.get_state(), dtype=float)

        move_arr = [0, 0, 0]
        move_arr[move] = 1

        # train short memory
        agent.train_short_memory(state_old, move_arr, reward, state_new, done)

        # remember
        agent.remember(state_old, move_arr, reward, state_new, done)

        if done:

            # train long memory, plot result
            game.start()
            agent.n_games += 1
            agent.epsilon = (agent.epsilon - 1e-3) if agent.epsilon > 1e-2 else 1e-2
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            #plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()

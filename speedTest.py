"""
This program measures the speed of a training using RLlib with minimal package requirements.

Manually installed packages in this order :
 - gymnasium
 - ray[rllib]
 - torch
 - tqdm
"""

# Imports needed for time measurement
import time  # Used to have definitive result
from tqdm import tqdm  # Used in for loop for time preview in case N_ITER is set too high

# Imports permitting environment definition
import gymnasium  # we use gymnasium env
from gymnasium import spaces  # Permits to define observation and action spaces
from gymnasium.envs.toy_text.frozen_lake import generate_random_map  # Used to generate maps for maze

# Imports needed for maths
import numpy as np

# Needed to initialize and configure ray
import ray  # ray is the library in which rllib is contained
from ray.rllib.algorithms.dqn import DQNConfig  # We use DQN algorithm
from ray.rllib.connectors.env_to_module import FlattenObservations  # Needed for one hot encoding



# Number of training iteration to be done.
N_ITER = 20




# Maze Environment
class FrozenLake3Env(gymnasium.Env) :
    def __init__(self, options = None) :
        super(FrozenLake3Env, self).__init__()
        self.map = options['lake_map']
        self.width = len(self.map[0])
        self.height = len(self.map)
        self.cell_values = {'F' : 0, 'S' : 0, 'H' : 1, 'G' : 2, 'W' : 3}
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([self.width * self.height, 4, 4, 4, 4])
        self.state = 0

        for x in range(self.width) :
            for y in range(self.height) :
                if self.map[y][x] == 'S' :
                    self.state = self._coordinateToState(x, y)
                    break
        

    
    def reset(self, seed = None, options = None) :
        super().reset(seed = seed)
        self.state = 0

        for x in range(self.width) :
            for y in range(self.height) :
                if self.map[y][x] == 'S' :
                    self.state = self._coordinateToState(x, y)
                    break


        left, down, right, up = self._getNeighbors(self.state)
        return np.array([self.state, left, down, right, up]), {}
    
    def step(self, action) :
        x, y = self._stateToCoordinate(self.state)
        if action == 0 : # Left
            if x > 0 :
                x -= 1
        elif action == 1 : # Down
            if y < self.height -1 :
                y += 1
        elif action == 2 : # Right
            if x < self.width -1 :
                x += 1
        elif action == 3 : # Up
            if y > 0 :
                y -= 1
        
        self.state = self._coordinateToState(x, y)
        
        result = self.map[y][x]
        done = result in ['H', 'G']
        if result == 'G' :
            reward = 1
        elif result == 'H' :
            reward = -1
        else :
            reward = -1/(self.width * self.height)

        left, down, right, up = self._getNeighbors(self.state)
        return np.array([self.state, left, down, right, up]), reward, done, False, {}
    
    
    def _stateToCoordinate(self, state) :
        x = state % self.width
        y = state // self.width
        return x, y
    
    def _coordinateToState(self, x, y) :
        return self.width*y + x

    def _getNeighbors(self, state) :
        x, y = self._stateToCoordinate(state)
        if x == 0 :  # Retrieve left cell
            left = self.cell_values['W']
        else :
            left = self.cell_values[self.map[y][x-1]]

        if y == self.height -1 :  # Retrieve down cell
            down = self.cell_values["W"]
        else :
            down = self.cell_values[self.map[y+1][x]]

        if x == self.width -1 :   # Retrieve right cell
            right = self.cell_values["W"]
        else :
            right = self.cell_values[self.map[y][x+1]]

        if y == 0 :   # Retrieve up cell
            up = self.cell_values["W"]
        else :
            up = self.cell_values[self.map[y-1][x]]

        return left, down, right, up


    
    def render(self) :
        x, y = self._stateToCoordinate(self.state)
        toPrint = ''
        for i in range(self.height) :
            toPrintRow = ''
            row = self.map[i]
            for j in range(self.width) :
                if (j, i) == (x, y):
                    toPrintRow += '.'
                else :
                    toPrintRow += row[j]
            toPrint += toPrintRow + '\n'
        print(toPrint)


# Initialize ray
ray.init()


# Create configuration of future algorithm
config = (
    DQNConfig()
    .environment(
        FrozenLake3Env, # Specify our env
        env_config = {'lake_map' : generate_random_map(10)}, # Provide maze's map
    )
    
    .env_runners(
        env_to_module_connector=lambda env: FlattenObservations(), # Enable one hot encoding
    )
)


# Config verifications
config.framework("torch")
config.validate()


# Get algorithm from config
algo = config.build_algo()


# Display first part
print('----')
print("Starting Training Loop")
print("Chosen number of iterations :", N_ITER)


# Training loop
start = time.time()
for _ in tqdm(range(N_ITER)) :
    results = algo.train()
end = time.time()


# Display time
print(f"Time spent : {end - start} seconds with {N_ITER} iterations.")


# Stop everything
algo.stop()
ray.shutdown()
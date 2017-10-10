import numpy as np
from collections import namedtuple
import copy
import matplotlib.pyplot as plt
import multiprocessing
import matplotlib.lines as mlines

class gridWorld:
    def __init__(self, grid):
        self.grid = grid
        self.actions = np.array(['up', 'down', 'left', 'right'])
        self.states = np.array(range(0,self.grid.shape[0] * self.grid.shape[1]))
        self.rewards = np.zeros(grid.shape)
        self.rewards[grid == '#'] = -10
        self.rewards[grid == '*'] =0
        self.rewards[grid == 'o'] = -1
        self.Trans = np.ndarray([self.grid.size, self.actions.size, 2], dtype='f2,u2,i2,u2')
        self.previous_grid_state_robot = "o"
        self.previous_grid_location_robot = []

    def reset(self):
        self.previous_grid_state_robot = "o"
        self.previous_grid_location_robot = []
        self.grid = copy.deepcopy(grid)


    def gridGenerator(self):
        self.rewards[self.grid == "#"] = -100
        self.rewards[self.grid == "*"] = 10
        self.rewards[self.grid == "o"] = -1
        self.rewards[self.grid == "r#"] = -100
        self.rewards[self.grid == "r*"] = 10

    def transitionFunc(self):
        action_possibilities = [None] * (self.grid.shape[0] * self.grid.shape[1])

        for i in range(self.grid.shape[0]):

            for j in range(self.grid.shape[1]):

                if i > 0 and i != self.grid.shape[0] - 1:
                    action_possibilities[self.grid.shape[1] * i + j] = (['up', 'down'])

                elif i > 0 and i == self.grid.shape[0] - 1:
                    action_possibilities[self.grid.shape[1] * i + j] = (['up',None])

                else:
                    action_possibilities[self.grid.shape[1] * i + j] = ([None,'down'])

                if j > 0 and j != self.grid.shape[1] - 1:
                    action_possibilities[self.grid.shape[1] * i + j].append('left')
                    action_possibilities[self.grid.shape[1] * i + j].append('right')

                elif j > 0 and j == self.grid.shape[1] - 1:
                    action_possibilities[self.grid.shape[1] * i + j].append('left')
                    action_possibilities[self.grid.shape[1] * i + j].append(None)
                else:
                    action_possibilities[self.grid.shape[1] * i + j].append(None)
                    action_possibilities[self.grid.shape[1] * i + j].append('right')

        grid = self.grid.reshape(self.grid.shape[0] * self.grid.shape[1])
        rewards=self.rewards.reshape(self.grid.shape[0] * self.grid.shape[1])
        for s, state in enumerate(grid):
            if state=='*':
                done=True
            else:
                done=False
            for a, action in enumerate(action_possibilities[s]):
                if action != None:
                    prob = 1.0
                    if action == 'left':
                        next_state = s - 1
                    elif action == 'right':
                        next_state = s + 1
                    elif action == 'up':
                        next_state = s - self.grid.shape[1]
                    elif action == 'down':
                        next_state = s + self.grid.shape[1]

                    self.Trans[s, a, 0] = (prob, next_state, rewards[next_state],done)
                    self.Trans[s, a, 1] = (1.0 - prob, s,rewards[s],done)
                else:
                    next_state = s
                    self.Trans[s, a, 0] = (1.0, next_state,rewards[next_state],done)
                    self.Trans[s, a, 1] = (0.0, s, rewards[s],done)
                if done:
                    self.Trans[s, a, 0] = (1.0, s, 0, done)
                    self.Trans[s, a, 1] = (0.0, s, 0, done)
        return self.Trans

    def validateState(self, state):
        obstacles = []
        goal=[]
        for obstacles_x, obstacles_y in zip(*np.where(self.grid == "#")):
            obstacles.append(self.grid.shape[1]*obstacles_x+obstacles_y)
        for goal_x, goal_y in zip(*np.where(self.grid == "*")):
            goal.append(self.grid.shape[1]*goal_x+goal_y)
        if state.static_location in obstacles or state.static_location in goal:

            return False
        else:
            return True

    def initializeState(self):
        state_to_location = np.reshape(np.arange(16), grid.shape)
        while True:
            robot_location = np.random.choice([i for i in range(grid.size)], p=[1./env.grid.size]*env.grid.size)
            state = State(robot_location)
            if env.validateState(state):
                break
        for i,j in zip(*np.where(state_to_location == state)):
            initialState = (i,j)

        self.grid[initialState[0], initialState[1]] = 'r'
        return state

    def executeAction(self, state, action):
        max_action_prob = 1.0
        state_to_location = np.reshape(np.arange(16), grid.shape)

        for i, j in zip(*np.where(state_to_location == state.static_location)):
            self.previous_grid_location_robot = [i, j]


        for i, j in zip(*np.where(state_to_location == state.static_location)):
            current_location = [i, j]

        if action == 2:
            if current_location[1] > 0:
                next_state = np.random.choice([state.static_location - 1, state.static_location], p=[max_action_prob, 1. - max_action_prob])
            else:
                next_state = state.static_location
        elif action == 3:
            if current_location[1] == 0 or (current_location[1] > 0 and current_location[1] != self.grid.shape[1] - 1):
                next_state = np.random.choice([state.static_location + 1, state.static_location], p=[max_action_prob, 1. - max_action_prob])
            else:
                next_state = state.static_location
        elif action == 0:
            if current_location[0] > 0:
                next_state = np.random.choice([state.static_location - self.grid.shape[1], state.static_location], p=[max_action_prob, 1. - max_action_prob])
            else:
                next_state = state.static_location
        elif action == 1:
            if current_location[0] == 0 or (current_location[0] > 0 and current_location[0] != self.grid.shape[0] - 1):
                next_state = np.random.choice([state.static_location + self.grid.shape[1], state.static_location], p=[max_action_prob, 1. - max_action_prob])
            else:
                next_state = state.static_location
        else:
            next_state = state.static_location

        for i, j in zip(*np.where(state_to_location == next_state)):
            robot_location = (i, j)

        #
        # if next_state < 0 or next_state>env.grid.size-1:
        #     next_state = state.static_location


        ########  Reset Previous States ##############################
        self.grid[self.previous_grid_location_robot[0], self.previous_grid_location_robot[1]] = self.previous_grid_state_robot

        ######## Backup Next State ###################################
        self.previous_grid_location_robot = [robot_location[0], robot_location[1]]

        self.previous_grid_state_robot = self.grid[robot_location[0], robot_location[1]]


        ######### Set Next State as Current State ####################
        if self.grid[robot_location[0], robot_location[1]] == "*":
            self.grid[robot_location[0], robot_location[1]] = "r*"
            done = True
        elif self.grid[robot_location[0], robot_location[1]] == "#":
            self.grid[robot_location[0], robot_location[1]] = "r#"
            done = False
        else:
            self.grid[robot_location[0], robot_location[1]] = "r"
            done = False

        env.gridGenerator()
        rewards = self.rewards.reshape(self.grid.shape[0] * self.grid.shape[1])

        return State(next_state), rewards[next_state], done



# Main Program Area
grid = np.array([['o', 'o', 'o', '*'],
                 ['#', '#', 'o', '#'],
                 ['o', 'o', 'o', 'o'],
                 ['o', 'o', 'o', 'o']], dtype='S4')

env = gridWorld(grid)

State=namedtuple("State", ["static_location"])
policy={}
for i in range(grid.size):
    policy[State(i)] = [0.25, 0.25, 0.25, 0.25]




# policy = valueIterationAlgo(env)

# print policy
# grid_policy=np.zeros(grid.size,dtype=env.actions.dtype)
# for i in range(grid.size):
#    print env.actions[policy[i]==1]
#
# print ("{0},\n{1}").format(policy,V)


# while True:
#     env.reset()
#     # action=input("What do you want me to do?")
#     #
#     # state,reward,done=env.executeAction(state,action)
#     # print (state,reward,done)
#     # print env.grid
#     x=input("Press Enter for next state")
#     state=env.initializeState()
#     print state
#     print env.grid
#     print env.rewards





import numpy as np
import copy
from collections import namedtuple
import algorithms
class gridWorld:

    def __init__(self, grid):
        self.grid = copy.deepcopy(grid)
        self.actions = np.array(['up', 'down', 'left', 'right', 'wait'])
        self.states = np.array(range(0, self.grid.shape[0] * self.grid.shape[1]))
        self.rewards = np.zeros(self.grid.shape)


    def reset(self):
        self.previous_grid_state_robot = "o"
        self.previous_grid_state_dynobj = "o"
        self.previous_grid_location_dynobj = []
        self.previous_grid_location_robot = []
        self.grid = copy.deepcopy(grid)

    def gridGenerator(self):

        self.rewards[self.grid == "#"] = -80
        self.rewards[self.grid == "*"] = 0
        self.rewards[self.grid == "o"] = -1
        self.rewards[self.grid == "^"] = -10
        self.rewards[self.grid == "r#"] = -80
        self.rewards[self.grid == "r^"] = -80
        self.rewards[self.grid == "r*"] = 0
        self.rewards[self.grid == "r<>"] = -80



    def validateState(self, state):
        obstacles = []
        goal=[]
        for obstacles_x, obstacles_y in zip(*np.where(self.grid == "#")):
            obstacles.append(self.grid.shape[1]*obstacles_x+obstacles_y)
        for goal_x, goal_y in zip(*np.where(self.grid == "*")):
            goal.append(self.grid.shape[1]*goal_x+goal_y)
        if state.static_location in obstacles or state.dynamic_location in obstacles or state.static_location==state.dynamic_location or state.static_location in goal or state.dynamic_location in goal:

            return False
        else:
            return True

    def initializeState(self):
        state_to_location = np.reshape(np.arange(16), grid.shape)
        while True:
            robot_location = np.random.choice([i for i in range(grid.size)], p=[1./env.grid.size]*env.grid.size)
            # dynamic_object_location = np.random.choice([i for i in range(grid.size)], p=[1./env.grid.size]*env.grid.size)
            dynamic_object_location = 2
            state = State(robot_location, dynamic_object_location)
            if env.validateState(state):
                break
        for i,j in zip(*np.where(state_to_location == state.static_location)):
            initialRobotState = (i,j)

        for i,j in zip(*np.where(state_to_location == state.dynamic_location)):
            initialDynamicState = (i,j)

        self.grid[initialRobotState[0], initialRobotState[1]] = 'r'
        self.grid[initialDynamicState[0], initialDynamicState[1]] = '^'
        return state

    def dynamicObjectGenerator(self, dynamic_location):
        state_to_location = np.reshape(np.arange(16), grid.shape)
        if dynamic_location == 14:
            next_location = (0, 0)
        else:
            for i, j in zip(*np.where(state_to_location == dynamic_location)):
                (locations_x, locations_y)=(i,j)
            obstacles_x, obstacles_y = np.where(self.grid == "#")
            obstacles = zip(obstacles_x, obstacles_y)
            action_possibilities = []
            if locations_x > 0 and locations_x != self.grid.shape[0] - 1:
                if (locations_x + 1, locations_y) in obstacles and (locations_x - 1, locations_y) in obstacles:
                    action_possibilities = ([None, None])
                elif (locations_x + 1, locations_y) in obstacles:
                    action_possibilities = (['up', None])
                elif (locations_x - 1, locations_y) in obstacles:
                    action_possibilities = ([None, 'down'])
                else:
                    action_possibilities = (['up', 'down'])

            elif locations_x > 0 and locations_x == self.grid.shape[0] - 1:
                if (locations_x - 1, locations_y) in obstacles:
                    action_possibilities = ([None, None])
                else:
                    action_possibilities = (['up', None])

            else:
                if (locations_x + 1, locations_y) in obstacles:
                    action_possibilities = ([None, None])
                else:
                    action_possibilities = ([None, 'down'])


            if locations_y > 0 and locations_y != self.grid.shape[1] - 1:
                if (locations_x, locations_y + 1) in obstacles and (locations_x,locations_y-1) in obstacles:
                    action_possibilities.append(None)
                    action_possibilities.append(None)
                elif (locations_x, locations_y + 1) in obstacles:
                    action_possibilities.append('right')
                    action_possibilities.append(None)
                elif (locations_x, locations_y - 1) in obstacles:
                    action_possibilities.append(None)
                    action_possibilities.append('left')
                else:
                    action_possibilities.append('left')
                    action_possibilities.append('right')

            elif locations_y > 0 and locations_y == self.grid.shape[1] - 1:
                if (locations_x, locations_y - 1) in obstacles:
                    action_possibilities.append(None)
                    action_possibilities.append(None)
                else:
                    action_possibilities.append('left')
                    action_possibilities.append(None)
            else:
                if (locations_x, locations_y + 1) in obstacles:
                    action_possibilities.append(None)
                    action_possibilities.append(None)
                else:
                    action_possibilities.append(None)
                    action_possibilities.append('right')

            if 'down' in action_possibilities:
                next_action ='down'
            else:
                next_action = 'right'

            # action_probabilities = []
            # for i, action in enumerate(action_possibilities):
            #     if action == None:
            #         action_probabilities.append(0.01)
            #     elif action == 'up':
            #         action_probabilities.append(0.02)
            #     elif action == 'down':
            #         action_probabilities.append(0.8)
            #     elif action == 'left':
            #         action_probabilities.append(0.08)
            #     elif action == 'right':
            #         action_probabilities.append(0.1)


            # next_action=np.random.choice(action_possibilities, p=[action_probability/sum(action_probabilities) for action_probability in action_probabilities])
            if next_action == 'up':
                next_location = (locations_x - 1, locations_y)
            elif next_action == 'down':
                next_location = (locations_x + 1, locations_y)
            elif next_action == 'left':
                next_location = (locations_x, locations_y - 1)
            elif next_action == 'right':
                next_location = (locations_x, locations_y + 1)
            elif next_action == None:
                next_location = (locations_x, locations_y)

        next_state = self.grid.shape[1]*next_location[0]+next_location[1]

        return next_location, next_state

    def executeAction(self, state, action):
        max_action_prob = 1.0
        state_to_location = np.reshape(np.arange(16), grid.shape)


        for i, j in zip(*np.where(state_to_location == state.static_location)):
            self.previous_grid_location_robot = [i, j]
        for i, j in zip(*np.where(state_to_location == state.dynamic_location)):
            self.previous_grid_location_dynobj = [i, j]

        for i,j in zip(*np.where(state_to_location == state.static_location)):
            current_location = [i,j]


        if action == 2:
            if current_location[1] > 0:
                next_state = np.random.choice([state.static_location - 1, state.static_location], p=[max_action_prob, 1.-max_action_prob])
            else:
                next_state = state.static_location
        elif action == 3:
            if current_location[1] == 0 or (current_location[1] > 0 and current_location[1] != self.grid.shape[1] - 1):
                next_state = np.random.choice([state.static_location + 1, state.static_location], p=[max_action_prob, 1.-max_action_prob])
            else:
                next_state = state.static_location
        elif action == 0:
            if current_location[0] > 0:
                next_state = np.random.choice([state.static_location - self.grid.shape[1], state.static_location], p=[max_action_prob, 1.-max_action_prob])
            else:
                next_state = state.static_location
        elif action == 1:
            if current_location[0] == 0 or (current_location[0] > 0 and current_location[0] != self.grid.shape[0] - 1):
                next_state = np.random.choice([state.static_location + self.grid.shape[1], state.static_location], p=[max_action_prob, 1.-max_action_prob])
            else:
                next_state = state.static_location
        elif action == 4:
            next_state = state.static_location
        else:
            next_state = state.static_location


        dynamic_location, dynamic_state = self.dynamicObjectGenerator(state.dynamic_location)
        for i, j in zip(*np.where(state_to_location == next_state)):
            robot_location = (i, j)


        if np.array_equal(self.previous_grid_location_robot,[dynamic_location[0],dynamic_location[1]]) and np.array_equal(self.previous_grid_location_dynobj,[robot_location[0],robot_location[1]]):
            interchange = True
        else:
            interchange = False
        #
        # if next_state < 0 or next_state>env.grid.size-1:
        #     next_state = state.static_location


        ########  Reset Previous States ##############################
        self.grid[self.previous_grid_location_robot[0], self.previous_grid_location_robot[1]] = self.previous_grid_state_robot
        self.grid[self.previous_grid_location_dynobj[0], self.previous_grid_location_dynobj[1]] = self.previous_grid_state_dynobj

        ######## Backup Next State ###################################
        self.previous_grid_location_robot = [robot_location[0], robot_location[1]]
        self.previous_grid_location_dynobj = [dynamic_location[0], dynamic_location[1]]
        self.previous_grid_state_robot = self.grid[robot_location[0], robot_location[1]]
        self.previous_grid_state_dynobj = self.grid[dynamic_location[0], dynamic_location[1]]


        ######### Set Next State as Current State ####################
        if interchange:
            self.grid[robot_location[0], robot_location[1]] = "r<>"
            self.grid[dynamic_location[0], dynamic_location[1]] = "^"
            done = False
        elif self.grid[robot_location[0], robot_location[1]] == "*":
            self.grid[robot_location[0], robot_location[1]] = "r*"
            self.grid[dynamic_location[0], dynamic_location[1]] = "^"
            done = True
        elif self.grid[robot_location[0], robot_location[1]] == "#":
            self.grid[robot_location[0], robot_location[1]] = "r#"
            self.grid[dynamic_location[0], dynamic_location[1]] = "^"
            done = False
        elif robot_location[0] == dynamic_location[0] and robot_location[1] == dynamic_location[1]:
            self.grid[robot_location[0], robot_location[1]] = "r^"
            done = False
        else:
            self.grid[robot_location[0], robot_location[1]] = "r"
            self.grid[dynamic_location[0], dynamic_location[1]] = "^"
            done = False

        env.gridGenerator()
        rewards = self.rewards.reshape(self.grid.shape[0] * self.grid.shape[1])

        return State(next_state,dynamic_state), rewards[next_state], done



grid = np.array([['o', 'o', 'o', '*'],
                 ['#', '#', 'o', '#'],
                 ['o', 'o', 'o', 'o'],
                 ['o', 'o', 'o', 'o']], dtype= 'S4')

State=namedtuple("State", ["static_location", "dynamic_location"])
policy = {}
for i in range(grid.size):
    for j in range(grid.size):
        policy[State(i, j)] = [0.2, 0.2, 0.2, 0.2, 0.2]
env = gridWorld(grid)

final_policy = algorithms.monteCarloPredictor(env, policy, 0.2, 90000)
















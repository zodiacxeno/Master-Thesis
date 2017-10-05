import numpy as np
import copy
from collections import namedtuple
class gridWorld:

    def __init__(self, grid):
        self.grid = copy.deepcopy(grid)
        self.actions = np.array(['up', 'down', 'left', 'right'])
        self.states = np.array(range(0, self.grid.shape[0] * self.grid.shape[1]))
        self.rewards = np.zeros(self.grid.shape)
        self.gridGenerator()

    def reset(self):
        self.grid = copy.deepcopy(grid)

    def gridGenerator(self):

        self.rewards[self.grid == '#'] = -10
        self.rewards[self.grid == '*'] = 0
        self.rewards[self.grid == 'o'] = -1
        self.rewards[self.grid == '^'] = -10

        self.previous_grid_state_robot = 'o'
        self.previous_grid_state_dynobj = 'o'
        self.previous_grid_location_dynobj = []
        self.previous_grid_location_robot = []

    def validateState(self, state):
        obstacles = []
        goal=[]
        for obstacles_x, obstacles_y in zip(*np.where(self.grid == '#')):
            obstacles.append(self.grid.shape[1]*obstacles_x+obstacles_y)
        for goal_x, goal_y in zip(*np.where(self.grid== '*')):
            goal.append(self.grid.shape[1]*goal_x+goal_y)
        if state.static_location in obstacles or state.dynamic_location in obstacles or state.static_location==state.dynamic_location or state.static_location in goal or state.dynamic_location in goal:

            return False
        else:
            return True

    def dynamicObjectGenerator(self, dynamic_location):
        state_to_location = np.reshape(np.arange(16), grid.shape)
        for i, j in zip(*np.where(state_to_location == dynamic_location)):
            (locations_x, locations_y)=(i,j)
        obstacles_x, obstacles_y = np.where(self.grid == '#')
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

        action_probabilities = []
        for i, action in enumerate(action_possibilities):
            if action == None:
                action_probabilities.append(0.0)
            elif action == 'up':
                action_probabilities.append(0.1)
            elif action == 'down':
                action_probabilities.append(0.7)
            elif action == 'left':
                action_probabilities.append(0.1)
            elif action == 'right':
                action_probabilities.append(0.1)

        next_action=np.random.choice(action_possibilities, p=[action_probability/sum(action_probabilities) for action_probability in action_probabilities])
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

    def executeAction(self,state,action):
        max_action_prob=0.8
        state_to_location = np.reshape(np.arange(16), grid.shape)
        rewards=self.rewards.reshape(self.grid.shape[0]*self.grid.shape[1])

        for i, j in zip(*np.where(state_to_location == state.static_location)):
            self.previous_grid_location_robot = [i,j]
        for i, j in zip(*np.where(state_to_location == state.dynamic_location)):
            self.previous_grid_location_dynobj = [i,j]


        if state == '*':
            done = True
        else:
            done = False

        if action == 'left':
            next_state = np.random.choice([state.static_location - 1, state.static_location], p=[max_action_prob, 1.-max_action_prob])
        elif action == 'right':
            next_state = np.random.choice([state.static_location + 1, state.static_location], p=[max_action_prob, 1.-max_action_prob])
        elif action == 'up':
            next_state = np.random.choice([state.static_location - self.grid.shape[1], state.static_location], p=[max_action_prob, 1.-max_action_prob])
        elif action == 'down':
            next_state = np.random.choice([state.static_location + self.grid.shape[1], state.static_location], p=[max_action_prob, 1.-max_action_prob])
        else:
            next_state = state.static_location

        dynamic_location, dynamic_state=self.dynamicObjectGenerator(state.dynamic_location)
        for i, j in zip(*np.where(state_to_location == next_state)):
            robot_location=(i,j)

        self.grid[self.previous_grid_location_robot[0], self.previous_grid_location_robot[1]] = self.previous_grid_state_robot
        self.grid[self.previous_grid_location_dynobj[0], self.previous_grid_location_dynobj[1]] = self.previous_grid_state_dynobj

        self.previous_grid_location_robot = [robot_location[0], robot_location[1]]
        self.previous_grid_location_dynobj = [dynamic_location[0], dynamic_location[1]]
        self.previous_grid_state_robot = self.grid[robot_location[0], robot_location[1]]
        self.previous_grid_state_dynobj = self.grid[dynamic_location[0], dynamic_location[1]]


        self.grid[dynamic_location[0], dynamic_location[1]] = '^'
        self.grid[robot_location[0], robot_location[1]] = 'r'


        return State(next_state,dynamic_state), rewards[next_state], done

def monteCarloPredictor(env,policy,epsilon,discount_factor=0.5):
    Q = {}
    returns_value = {}
    returns_count = {}
    for i in range(grid.size):
        for j in range(grid.size):
            Q[State(i, j)] = [0] * 4
            returns_value[State(i, j)] = [0] * 4
            returns_count[State(i, j)] = [0] * 4
    for i in range(0, 1000):
        env.reset()


        # returns_value = np.zeros([env.grid.size**2, env.actions.size])
        # returns_count = np.zeros([env.grid.size**2, env.actions.size])

        while True:
            robot_location = np.random.choice([i for i in range(grid.size)], p=[1./env.grid.size]*env.grid.size)
            dynamic_object_location = np.random.choice([i for i in range(grid.size)], p=[1./env.grid.size]*env.grid.size)
            state = State(robot_location, dynamic_object_location)
            if env.validateState(state):
                break

        episode = []
        print "--------------------------------------Start of Episode ----------------------------------------------"
        for t in range(100):
            action = policy[state]
            next_state, reward, done = env.executeAction(state, action)
            episode.append((state, action, reward))
            print env.grid
            if done:
                break
            state = next_state
        states_in_episode = set([tuple(x[0]) for x in episode])
        for state in states_in_episode:
            first_occurrence_id = next(i for i, x in enumerate(episode) if x[0] == state)
            G = sum(x[2]*discount_factor**i for i, x in enumerate(episode[first_occurrence_id:]))
            returns_value[state][action] += G
            returns_count[state][action] += 1
            Q[state][action] = returns_value[state][action]/returns_count[state][action]
            # print Q[state]

        for i in range(grid.size):
            for j in range(grid.size):
                epsilon_greedy_prob = (np.eye(env.actions.size)[np.argmax(Q[State(i, j)])]*(1-epsilon))+(epsilon/env.actions.size)
                policy[State(i, j)]=np.random.choice([k for k in range(4)], p=epsilon_greedy_prob)

        print "--------------------------------------- End of Episode -------------------------------------------------"

    return policy

        # for i in range(grid.size):
        #     for j in range(grid.size):
        #         policy[State(i, j)]=np.argmax(Q[State(i, j)])


# def simulate(grid, start_location,final_policy):
#     state_to_location = np.reshape(np.arange(16), grid.shape)
#     robot_location=start_location
#     for i in range(0,100):
#         env = gridWorld(grid)
#         env.grid[robot_location] = 'r'
#         env.grid[np.where(env.grid == '^')] = 'o'
#         env.grid[env.dynamic_location[0], env.dynamic_location[1]] = '^'
#         print env.grid
#
#         state = State(int(env.grid.shape[1]*robot_location[0]+robot_location[1]), int(env.grid.shape[1] * env.dynamic_location[0] + env.dynamic_location[1]))
#         action=env.actions[np.argmax(final_policy[state])]
#         robot_next_state, reward, done = env.executeAction(int(env.grid.shape[1]*robot_location[0]+robot_location[1]), action)
#         env.grid[robot_location] = 'o'
#         for i,j in zip(*np.where(state_to_location==robot_next_state)):
#             robot_location=(i,j)
#         if done:
#             break




grid = np.array([['o', 'o', 'o', '*'],
                 ['#', '#', 'o', '#'],
                 ['o', 'o', 'o', 'o'],
                 ['o', 'o', 'o', 'o']])

State=namedtuple("State", ["static_location", "dynamic_location"])
policy = {}
for i in range(grid.size):
    for j in range(grid.size):
        policy[State(i, j)] = np.random.choice([k for k in range(4)], p=[0.25, 0.25, 0.25, 0.25])
env = gridWorld(grid)
final_policy = monteCarloPredictor(env, policy, 0.2)
print final_policy
# simulate(grid,(3,0),final_policy)














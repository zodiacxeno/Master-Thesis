import numpy as np
from collections import namedtuple
class gridWorld:
    def __init__(self, grid):
        self.grid = grid
        self.actions = np.array(['up', 'down', 'left', 'right'])
        self.states = np.array(range(0, self.grid.shape[0] * self.grid.shape[1]))
        self.rewards = np.zeros(self.grid.shape)
        self.gridGenerator()
        self.dynamic_location=self.dynamicObjectGenerator()

    def gridGenerator(self):

        self.rewards[self.grid == '#'] = -10
        self.rewards[self.grid == '*'] = 0
        self.rewards[self.grid == 'o'] = -1
        self.rewards[self.grid == '^'] = -10

    def dynamicObjectGenerator(self):
        locations_x, locations_y = np.where(self.grid == '^')
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
            if i!=len(action_possibilities)-1:
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
            else:
                action_probabilities.append(1.-sum(action_probabilities))

        next_action=np.random.choice(action_possibilities, p=action_probabilities)
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

        return next_location

    def executeAction(self,state,action):
        rewards=self.rewards.reshape(self.grid.shape[0]*self.grid.shape[1])
        if state == '*':
            done = True
        else:
            done = False

        if action == 'left':
            next_state = state - 1
        elif action == 'right':
            next_state = state + 1
        elif action == 'up':
            next_state = state - self.grid.shape[1]
        elif action == 'down':
            next_state = state + self.grid.shape[1]
        else:
            next_state = state

        return next_state, rewards[next_state], done

def monteCarloPredictor(grid,policy,epsilon,discount_factor=0.5):
    Q = {}
    returns_value = {}
    returns_count = {}
    for i in range(grid.size):
        for j in range(grid.size):
            Q[State(i, j)] = [0] * 4
            returns_value[State(i, j)] = [0] * 4
            returns_count[State(i, j)] = [0] * 4
    for i in range(0, 1000):
        env = gridWorld(grid)
        env.grid[np.where(env.grid == '^')] = 'o'
        env.grid[env.dynamic_location[0], env.dynamic_location[1]] = '^'
        print env.grid

        # returns_value = np.zeros([env.grid.size**2, env.actions.size])
        # returns_count = np.zeros([env.grid.size**2, env.actions.size])

        x=np.random.choice([i for i in range(grid.size)], p=[1./env.grid.size]*env.grid.size)
        state=State(x, int(env.grid.shape[1]*env.dynamic_location[0]+env.dynamic_location[1]))
        episode=[]
        for t in range(100):
            action = policy[state]
            robot_next_state, reward, done = env.executeAction(state.static_location,action)
            episode.append((state, action, reward))
            if done:
                break
            state = State(robot_next_state, int(env.grid.shape[1]*env.dynamic_location[0]+env.dynamic_location[1]))

        for state, action, reward in episode:
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

    return policy

        # for i in range(grid.size):
        #     for j in range(grid.size):
        #         policy[State(i, j)]=np.argmax(Q[State(i, j)])


def simulate(grid, start_location,final_policy):
    state_to_location = np.reshape(np.arange(16), grid.shape)
    robot_location=start_location
    for i in range(0,100):
        env = gridWorld(grid)
        env.grid[robot_location] = 'r'
        env.grid[np.where(env.grid == '^')] = 'o'
        env.grid[env.dynamic_location[0], env.dynamic_location[1]] = '^'
        print env.grid

        state = State(int(env.grid.shape[1]*robot_location[0]+robot_location[1]), int(env.grid.shape[1] * env.dynamic_location[0] + env.dynamic_location[1]))
        action=env.actions[np.argmax(final_policy[state])]
        robot_next_state, reward, done = env.executeAction(int(env.grid.shape[1]*robot_location[0]+robot_location[1]), action)
        env.grid[robot_location] = 'o'
        for i,j in zip(*np.where(state_to_location==robot_next_state)):
            robot_location=(i,j)
        if done:
            break




grid = np.array([['o', 'o', '^', '*'],
                 ['#', '#', 'o', '#'],
                 ['o', 'o', 'o', 'o'],
                 ['o', 'o', 'o', 'o']])

State=namedtuple("State", ["static_location", "dynamic_location"])
policy = {}
for i in range(grid.size):
    for j in range(grid.size):
        policy[State(i, j)] = np.random.choice([k for k in range(4)], p=[0.25, 0.25, 0.25, 0.25])

final_policy = monteCarloPredictor(grid, policy, 0.2)
print final_policy
simulate(grid,(3,0),final_policy)














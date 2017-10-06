import numpy as np
from collections import namedtuple
import copy
import matplotlib.pyplot as plt

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

def valueIterationAlgo(env,discount_factor=0.7,theta=0.0001):
    V=np.zeros(env.states.size)
    P=env.transitionFunc()
    policy=np.zeros([env.states.size,env.actions.size])
    V_trend=[]
    while True:
        delta = 0
        for s in env.states:
            action_values = np.zeros(env.actions.size)
            for a, actions in enumerate(env.actions):
                for prob,next_state,reward,done in P[s,a]:
                    action_values[a]+=prob*(reward+discount_factor*V[next_state])

            V_new=max(action_values)
            policy[s]=np.eye(env.actions.size)[np.argmax(action_values)]
            delta=max(delta,np.abs(V_new-V[s]))
            V[s]=V_new
        V_trend.append(sum(V))
        print V.reshape(grid.shape)
        if delta<theta:
           return policy, V_trend, V
           break


def monteCarloPredictor(env, policy, epsilon, num_episodes, discount_factor=1.0):
    Q = dict(policy)
    returns_value = dict(policy)
    returns_count = dict(policy)
    V={}
    V_trend=[]
    for key in policy:
        Q[key] = [0] * env.actions.size
        returns_value[key] = [0] * env.actions.size
        returns_count[key] = [0] * env.actions.size

    for i in range(0, num_episodes):
        env.reset()
        state = env.initializeState()
        # print env.grid
        episode = []
        print ("Episode: {0}".format(i))
        # print "--------------------------------------Start of Episode ----------------------------------------------"
        for t in range(100):
            action = np.random.choice([k for k in range(4)], p=policy[state])
            # print state, env.actions[action]
            next_state, reward, done = env.executeAction(state, action)
            # print next_state
            episode.append((state, action, reward))
            # print env.grid
            if done:
                break
            state = next_state



        # Version 1: Only first state,action pair in episode is considered for updating the Q-value table
        states_in_episode = set([episode[0][0]])
        G = sum(x[2] * discount_factor ** i for i, x in enumerate(episode[0:]))
        returns_value[episode[0][0]][episode[0][1]] += G
        returns_count[episode[0][0]][episode[0][1]] += 1
        Q[episode[0][0]][episode[0][1]] = returns_value[episode[0][0]][episode[0][1]] / returns_count[episode[0][0]][episode[0][1]]



        # Version 2: Subsequent episodes are considered but with first-visit states
        # states_in_episode = set([(x[0]) for x in episode])
        # for state in states_in_episode:
        #     first_occurrence_id = next(i for i, x in enumerate(episode) if x[0] == state)
        #     G = sum(x[2]*discount_factor**i for i, x in enumerate(episode[first_occurrence_id:]))
        #     returns_value[state][episode[first_occurrence_id][1]] += G
        #     returns_count[state][episode[first_occurrence_id][1]] += 1
        #     Q[state][episode[first_occurrence_id][1]] = returns_value[state][episode[first_occurrence_id][1]]/returns_count[state][episode[first_occurrence_id][1]]
        # # #     # print Q[state]
        #
        for state in states_in_episode:
            epsilon_greedy_prob = (np.eye(env.actions.size)[np.argmax(Q[state])]*(1-epsilon))+(epsilon/env.actions.size)
            policy[state] = epsilon_greedy_prob

        for state, Q_values in Q.items():
            V[state]=sum([Q_values[val]*policy[state][val] for val in range(len(Q_values))])

        V_trend.append(sum(V.values()))


        # print "--------------------------------------- End of Episode -------------------------------------------------"


    return policy,V_trend, V

def comparePolicies(env,policy1,policy2):
    grid_policy_mc = np.zeros(policy1.__len__(), dtype='S8')
    grid_policy_val = np.zeros(policy2.__len__(), dtype='S8')
    for i in range(16):
        if np.argmax(policy1[State(i)]) == np.argmax(policy2[i]):
            print True
        else:
            print False

        grid_policy_mc[i] = env.actions[np.argmax(policy1[State(i)])]

        grid_policy_val[i] = env.actions[np.argmax(policy2[i])]


    print "Monte Carlo: "
    print np.reshape(grid_policy_mc,env.grid.shape)
    print "Value Iteration:"
    print np.reshape(grid_policy_val, env.grid.shape)


def plotTrend(algorithm,subplot):
    policy, V_trend, V = algorithm
    print policy
    plt.figure(1)
    plt.subplot(subplot)
    plt.plot(V_trend)
    plt.ylabel('V values')
    plt.xlabel('Iteration Number')


def testAlgoParam():
    num_iters = np.linspace(1000, 50000, 50)
    epsilon = np.linspace(0.0, 1.0, 10)
    V_trend = []
    for num in num_iters:
        _, _, V = monteCarloPredictor(env, policy, 0.1, int(num))
        V_trend.append(V)

    plt.figure(1)
    plt.subplot(121).set_title("Iteration Performance")
    plt.xlabel('Number of Iterations')
    plt.ylabel('V Value')
    plt.plot(V)

    for eps in epsilon:
        _, _, V = monteCarloPredictor(env, policy, eps, 10000)
        V_trend.append(V)

    plt.figure(1)
    plt.subplot(122).set_title("Exploration Performance")
    plt.xlabel('Epsilon')
    plt.ylabel('V Value')
    plt.plot(V)

    plt.show()

grid = np.array([['o', 'o', 'o', '*'],
                 ['#', '#', 'o', '#'],
                 ['o', 'o', 'o', 'o'],
                 ['o', 'o', 'o', 'o']], dtype='S4')

env = gridWorld(grid)

State=namedtuple("State", ["static_location"])
policy={}
for i in range(grid.size):
    policy[State(i)] = [0.25, 0.25, 0.25, 0.25]

# policy_mc, _ = monteCarloPredictor(env, policy, 0.1, 100)
plotTrend(monteCarloPredictor(env, policy, 0.1, 1000),subplot=121)
plotTrend(valueIterationAlgo(env), subplot=122)
plt.show()
testAlgoParam()







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





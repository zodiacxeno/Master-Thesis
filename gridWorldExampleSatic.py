import numpy as np


class gridWorld:
    def __init__(self, grid):
        self.grid = grid
        self.actions = np.array(['up', 'down', 'left', 'right'])
        self.states= np.array(range(0,self.grid.shape[0] * self.grid.shape[1]))
        self.rewards = np.zeros(grid.shape)
        self.rewards[grid == '#'] = -10
        self.rewards[grid == '*'] =0
        self.rewards[grid == 'o'] = -1
        self.Trans = np.ndarray([self.grid.size, self.actions.size, 2], dtype='f2,u2,i2,u2')

    def transitionFunc(self):
        action_possibilities = [None] * (self.grid.shape[0] * self.grid.shape[1])

        for i in range(self.grid.shape[0]):

            for j in range(self.grid.shape[1]):

                if i > 0 and i != self.grid.shape[0] - 1:
                    action_possibilities[4 * i + j] = (['up', 'down'])

                elif i > 0 and i == self.grid.shape[0] - 1:
                    action_possibilities[4 * i + j] = (['up',None])

                else:
                    action_possibilities[4 * i + j] = ([None,'down'])

                if j > 0 and j != self.grid.shape[1] - 1:
                    action_possibilities[4 * i + j].append('left')
                    action_possibilities[4 * i + j].append('right')

                elif j > 0 and j == self.grid.shape[1] - 1:
                    action_possibilities[4 * i + j].append('left')
                    action_possibilities[4 * i + j].append(None)
                else:
                    action_possibilities[4 * i + j].append(None)
                    action_possibilities[4 * i + j].append('right')

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
                        next_state = s - 4
                    elif action == 'down':
                        next_state = s + 4

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

def valueIterationAlgo(env,discount_factor=0.7,theta=0.0001):
    V=np.zeros(env.states.size)
    P=env.transitionFunc()
    policy=np.zeros([env.states.size,env.actions.size])
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
        print V.reshape(grid.shape)
        if delta<theta:
           return policy,V
           break


grid = np.array([['o', 'o', 'o', '*'],
                 ['#', '#', 'o', '#'],
                 ['o', 'o', 'o', 'o'],
                 ['o', 'o', 'o', 'o']])

env = gridWorld(grid)



policy, V=valueIterationAlgo(env)

print policy
# grid_policy=np.zeros(grid.size,dtype=env.actions.dtype)
for i in range(grid.size):
   print env.actions[policy[i]==1]
#
# print ("{0},\n{1}").format(policy,V)







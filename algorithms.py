import numpy as np

def valueIterationAlgo(env, discount_factor=0.7, theta=0.0001):
    V=np.zeros(env.states.size)
    P=env.transitionFunc()
    policy=np.zeros([env.states.size,env.actions.size])
    V_trend=[]
    while True:
        delta = 0
        for s in env.states:
            action_values = np.zeros(env.actions.size)
            for a, actions in enumerate(env.actions):
                for prob,next_state, reward, done in P[s,a]:
                    action_values[a] += prob * (reward+discount_factor*V[next_state])

            V_new = max(action_values)
            policy[s] = np.eye(env.actions.size)[np.argmax(action_values)]
            delta = max(delta, np.abs(V_new-V[s]))
            V[s] = V_new
        V_trend.append(sum(V))
        print V.reshape(env.grid.shape)
        if delta < theta:
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
        episode = []
        print ("Episode: {0}".format(i))
        print "--------------------------------------Start of Episode ----------------------------------------------"
        for t in range(100):
            print "--------------------------------------Step {0} ----------------------------------------------" .format(t)
            action = np.random.choice([k for k in range(env.actions.size)], p=policy[state])
            if i>num_episodes-50:
                print env.grid, env.actions[action]
            # print state, env.actions[action]
            next_state, reward, done = env.executeAction(state, action)
            # print next_state
            episode.append((state, action, reward))

            if done:
                break
            state = next_state



        # Version 1: Only first state,action pair in episode is considered for updating the Q-value table
        # states_in_episode = set([episode[0][0]])
        # G = sum(x[2] * discount_factor ** i for i, x in enumerate(episode[0:]))
        # returns_value[episode[0][0]][episode[0][1]] += G
        # returns_count[episode[0][0]][episode[0][1]] += 1
        # Q[episode[0][0]][episode[0][1]] = returns_value[episode[0][0]][episode[0][1]] / returns_count[episode[0][0]][episode[0][1]]



        # Version 2: Subsequent episodes are considered but with first-visit states
        states_in_episode = set([(x[0]) for x in episode])
        for state in states_in_episode:
            first_occurrence_id = next(i for i, x in enumerate(episode) if x[0] == state)
            G = sum(x[2]*discount_factor**i for i, x in enumerate(episode[first_occurrence_id:]))
            returns_value[state][episode[first_occurrence_id][1]] += G
            returns_count[state][episode[first_occurrence_id][1]] += 1
            Q[state][episode[first_occurrence_id][1]] = returns_value[state][episode[first_occurrence_id][1]]/returns_count[state][episode[first_occurrence_id][1]]


        for state in states_in_episode:
            epsilon_greedy_prob = (np.eye(env.actions.size)[np.argmax(Q[state])]*(1-epsilon))+(epsilon/env.actions.size)
            policy[state] = epsilon_greedy_prob

        for state, Q_values in Q.items():
            V[state]=sum([Q_values[val]*policy[state][val] for val in range(len(Q_values))])

        V_trend.append(sum(V.values()))


        print "--------------------------------------- End of Episode -------------------------------------------------"


    return policy,V_trend, sum(V.values())


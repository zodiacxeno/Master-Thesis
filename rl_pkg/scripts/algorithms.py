import numpy as np
from collections import namedtuple
import math

State=namedtuple("State", ["static_location", "dynamic_location"])
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
    V_episodic_trend = []
    for key in policy:
        Q[key] = [0] * env.actions.size
        returns_value[key] = [0] * env.actions.size
        returns_count[key] = [0] * env.actions.size

    for i in range(0, num_episodes):
        track_episode = iter(np.linspace(0, num_episodes, 10))
        #env.reset()
        state = env.initializeState()
        episode = []
        print ("Episode: {0}".format(i))
        print "--------------------------------------Start of Episode ----------------------------------------------"
        for t in range(100):
            print "--------------------------------------Step {0} ----------------------------------------------" .format(t)
            action = np.random.choice([k for k in range(env.actions.size)], p=policy[state])
            if i>num_episodes-50:
                print env.actions[action]
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
        if i == next(track_episode):
            V_episodic_trend.append(sum(V.values()))

        print "--------------------------------------- End of Episode -------------------------------------------------"


    return policy,V_trend, V_episodic_trend


def sarsaTD(env, policy, epsilon, num_episodes, step_rate=0.1, discount_factor=1.0):
    Q = dict(policy)
    V = {}
    V_trend = []
    V_episodic_trend = []
    sample_state = 12

    for key in policy:
        Q[key] = [0] * env.actions.size

    for i in range(0, num_episodes):

        env.reset()
        state = env.initializeState()

        print ("Episode: {0}".format(i))
        if i > num_episodes - 50:
            print "--------------------------------------Start of Episode ----------------------------------------------"
        current_state_action = np.random.choice([k for k in range(env.actions.size)], p=policy[state])
        for t in range(100):
            # if i > num_episodes - 50:
            print "--------------------------------------Step {0} ----------------------------------------------" .format(t)
            print env.grid, env.actions[current_state_action]

            next_state, reward, done = env.executeAction(state, current_state_action)
            epsilon_greedy_prob = (np.eye(env.actions.size)[np.argmax(Q[next_state])] * (1 - epsilon)) + (epsilon / env.actions.size)
            next_state_action = np.random.choice([k for k in range(env.actions.size)], p=epsilon_greedy_prob)
            # next_state_action = np.argmax(Q[next_state])
            Q[state][current_state_action] += step_rate*(reward+(discount_factor*Q[next_state][next_state_action]) - Q[state][current_state_action])
            if done:
                break
            state = next_state
            current_state_action = next_state_action

        for state, Q_values in Q.items():
            policy[state] = (np.eye(env.actions.size)[np.argmax(Q_values)] * (1 - epsilon)) + (epsilon / env.actions.size)
            V[state] = sum([Q_values[val] * policy[state][val] for val in range(len(Q_values))])

        V_trend.append(sum(V.values()))

        if i == 0:
            V_episodic_trend.append(sum(V.values()[sample_state * 16:(sample_state * 16) + 16]))

        elif (i+1) % (num_episodes/10) == 0:
            V_episodic_trend.append(sum(V.values()[sample_state*16:(sample_state*16)+16]))

        if i > num_episodes - 50:
            print "--------------------------------------- End of Episode -------------------------------------------------"

    return policy, V_trend, V_episodic_trend


def qLearning(env, policy, epsilon, num_episodes, step_rate=0.1, discount_factor=0.9):
    Q = dict(policy)
    V = {}
    V_trend = []
    V_episodic_trend = []
    sample_state=12
    for key in policy:
        Q[key] = [0] * env.actions.size

    for i in range(0, num_episodes):
        env.reset()
        state = env.initializeState()

        print ("Episode: {0}".format(i))
        # if i > num_episodes - 50:
        #     print "--------------------------------------Start of Episode ----------------------------------------------"

        for t in range(100):
            current_state_action = np.random.choice([k for k in range(env.actions.size)], p=policy[state])
            # if i > num_episodes - 50:
            #     print "--------------------------------------Step {0} ----------------------------------------------" .format(t)
            #     print env.grid, env.actions[current_state_action]

            next_state, reward, done = env.executeAction(state, current_state_action)
            next_state_action = np.argmax(Q[next_state])
            Q[state][current_state_action] += step_rate * (reward + (discount_factor * Q[next_state][next_state_action]) - Q[state][current_state_action])
            if done:
                break
            state = next_state


        for state, Q_values in Q.items():
            policy[state] = (np.eye(env.actions.size)[np.argmax(Q_values)] * (1 - epsilon)) + (epsilon / env.actions.size)
            V[state] = sum([Q_values[val] * policy[state][val] for val in range(len(Q_values))])

        V_trend.append(sum(V.values()))

        if i == 0:
            V_episodic_trend.append(sum(V.values()[sample_state * 16:(sample_state * 16) + 16]))

        elif (i+1) % (num_episodes/10) == 0:
            V_episodic_trend.append(sum(V.values()[sample_state*16:(sample_state*16)+16]))
        # if i > num_episodes - 50:
        #     print "--------------------------------------- End of Episode -------------------------------------------------"

    return policy, V_trend, V_episodic_trend


def nStepSarsa(env, policy, epsilon, num_episodes, step_count, step_rate=0.1, discount_factor=1.0):

    Q = dict(policy)
    V = {}
    V_trend = []
    V_episodic_trend = []
    sample_state = 12
    for key in policy:
        Q[key] = [0] * env.actions.size
        V[key] = 0

    for i in range(0, num_episodes):
        #env.reset()
        state = env.initializeState()
        T = 100
        episode = []
        print ("Episode: {0}".format(i))
        if i > num_episodes - 50:
             print "--------------------------------------Start of Episode ----------------------------------------------"

        epsilon_greedy_prob = (np.eye(env.actions.size)[np.argmax(Q[state])] * (1. - epsilon)) + (
        epsilon / env.actions.size)
        current_state_action = np.random.choice([k for k in range(env.actions.size)], p=epsilon_greedy_prob)
        episode.append((state, current_state_action, None))
        for t in range(0, T):
            print "--------------------------------------Step {0} ----------------------------------------------".format(
                t)
            print state, env.actions[current_state_action]

            if t < T:
                next_state, reward, done = env.executeAction(state, current_state_action)
                if done:
                    T = t+1
                else:
                    epsilon_greedy_prob = (np.eye(env.actions.size)[np.argmax(Q[next_state])] * (1 - epsilon)) + (epsilon / env.actions.size)
                    next_state_action = np.random.choice([k for k in range(env.actions.size)], p=epsilon_greedy_prob)
                    episode.append((next_state, next_state_action, reward))
            state = next_state
            current_state_action = next_state_action
            update_index = t-step_count+1

            if update_index == T:
                break
            elif update_index >= 0:
                backup_range = [update_index+1, min([update_index+step_count, T])+1]
                G = sum(x[2]*discount_factor**i for i, x in enumerate(episode[backup_range[0]:backup_range[1]]))
                if update_index + step_count < T:
                    G += (discount_factor**step_count)*Q[next_state][next_state_action]

                Q_index = [episode[update_index][0], episode[update_index][1]]
                Q[Q_index[0]][Q_index[1]] += step_rate*(G - Q[Q_index[0]][Q_index[1]])


        # if (i+1) % (num_episodes/10) == 0:
        #     state_return = 0
        #     # for j in range(16):
        #     env.reset()
        #     test_state = env.initializeState(State(63,0))
        #
        #     for test_count in range(0,350):
        #         epsilon_greedy_prob = np.eye(env.actions.size)[np.argmax(Q[test_state])]
        #         current_test_action = np.random.choice([k for k in range(env.actions.size)], p=epsilon_greedy_prob)
        #         next_test_state, reward, test_done = env.executeAction(test_state, current_test_action)
        #         state_return += reward
        #         if test_done:
        #             break
        #         test_state = next_test_state
        #     V_episodic_trend.append(state_return)

        for state, Q_values in Q.items():
            V[state] = sum([Q_values[val] * policy[state][val] for val in range(len(Q_values))])


        V_trend.append(sum(V.values()))



        if i > num_episodes - 50:
            print "--------------------------------------- End of Episode -------------------------------------------------"

    for state, Q_values in Q.items():
        policy[state] = (np.eye(env.actions.size)[np.argmax(Q_values)] * (1 - epsilon)) + (epsilon / env.actions.size)

    return policy, V
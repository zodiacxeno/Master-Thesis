import matplotlib.pyplot as plt
import multiprocessing
import matplotlib.lines as mlines
import numpy as np
import algorithms
from collections import namedtuple
import gridWorldExampleDynamic

# Function to compare policy results
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


# Function to plot convergence
def plotTrend(algorithm,subplot):
    policy, V_trend, V = algorithm
    print policy
    plt.figure(1)
    plt.subplot(subplot)
    plt.plot(V_trend)
    plt.ylabel('V values')
    plt.xlabel('Iteration Number')


# Function to test and plot performance metrics of the desired algorithm
def testAlgoParam(env, policy, num_episodes):

    num_iters = np.linspace(0, num_episodes, 11)
    epsilon = np.linspace(0.0, 1.0, 5)


    ## Plotting performance with respect to just number of iterations
    parameters = []
    V_iter = []

    print "Running: Performance based on No. of Episodes"

    plt.figure(1)
    _, _, V_test = algorithms.sarsaTD(env, policy, 0.1, num_episodes)

    print "Performance based on No. of Episodes ---- Complete"
    plt.title("Iteration Performance based on number of epsiodes (e = 0.2)")
    plt.xlabel('Number of Iterations')
    plt.ylabel('V Value')
    plt.plot(num_iters, V_test, marker='8')
    for i, j in zip(num_iters,V_test):
        plt.annotate(str(j), xy=(i, j))


    ## Plotting performance with respect to both number of iterations and epsilon
    parameters = []
    lines = []
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i, eps in enumerate(epsilon):
        parameters.append((env, policy, eps, num_episodes))
        lines.append(mlines.Line2D([], [], color=color[i], marker='8', label="eps: {0}".format(eps)))

    print "Running : Performance based on No. of Episodes and Epsilon" .format(eps)
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    p_iter = [p.apply_async(algorithms.sarsaTD, param) for param in parameters]
    p.close()
    p.join()

    print "Performance based on No. of Episodes and Epsilon ---- Complete"
    plt.figure(2)
    plt.title("Iteration Performance based on exploration and number of episodes")
    plt.xlabel('Number of Iterations')
    plt.ylabel('V Value')

    for i, value in enumerate(p_iter):

        _, _, V_test = value.get()

        plt.plot(num_iters, V_test, color[i], marker='8')

        for i, j in zip(num_iters, V_test):
            plt.annotate(str(j), xy=(i, j))

    plt.legend(handles=lines)

    plt.show()


grid = np.array([['o', 'o', 'o', '*'],
                 ['#', '#', 'o', '#'],
                 ['o', 'o', 'o', 'o'],
                 ['o', 'o', 'o', 'o']], dtype= 'S4')

State=namedtuple("State", ["static_location", "dynamic_location"])
policy = {}
for i in range(grid.size):
    for j in range(grid.size):
        policy[State(i, j)] = [0.2, 0.2, 0.2, 0.2, 0.2]

env = gridWorldExampleDynamic.gridWorld(grid)

# algorithms.qLearning(env, policy, 0.2, 10000)
testAlgoParam(env, policy, 100000)
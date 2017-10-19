import matplotlib.pyplot as plt
import multiprocessing
import matplotlib.lines as mlines
import numpy as np
import algorithms
from collections import namedtuple
import gridWorldExampleDynamic
from scipy import stats

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
    np.random.seed(23)
    num_iters = np.linspace(10, num_episodes, 10)
    epsilon = np.linspace(0.0, 1.0, 5)

    print "Running: Performance based on No. of Episodes"

    # plt.figure(1)
    # _, _, V_test = algorithms.nStepSarsa(env, policy, 0.1, num_episodes, 8)
    #
    # print "Performance based on No. of Episodes ---- Complete"
    # plt.title("Iteration Performance based on number of epsiodes (e = 0.2)")
    # plt.xlabel('Number of Iterations')
    # plt.ylabel('V Value')
    # plt.plot(num_iters, V_test, marker='8')
    # for i, j in zip(num_iters, V_test):
    #     plt.annotate(str(j), xy=(i, j))


    ## Plotting performance with respect to just number of iterations
    parameters = []
    lines = []
    print "Running: Performance based on No. of Episodes and Step Count"
    color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

    for i in range(0,9):
        parameters.append((env, policy, 0.2, num_episodes, i+1))
        lines.append(mlines.Line2D([], [], color=color[i], marker='8', label="Step Count: {0}".format(i+1)))

    p = multiprocessing.Pool(multiprocessing.cpu_count())
    p_iter = [p.apply_async(algorithms.nStepSarsa, param) for param in parameters]
    p.close()
    p.join()
    print "Performance based on No. of Episodes ---- Complete"

    plt.figure(algorithms.nStepSarsa.__name__)
    plt.title("Iteration Performance based on number of epsiodes (e = 0.1)")
    plt.xlabel('Number of Iterations')
    plt.ylabel('V Value')

    for i, value in enumerate(p_iter):
        _, _, V_test = value.get()
        print np.mean(V_test), stats.mode(V_test)
        plt.plot(num_iters, V_test, color[i], marker='8')

        for i, j in zip(num_iters, V_test):
            plt.annotate(str(j), xy=(i, j))

    plt.legend(handles=lines)





    ## Plotting performance with respect to both number of iterations and epsilon
    # parameters = []
    # lines = []
    # color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    # for i, eps in enumerate(epsilon):
    #     parameters.append((env, policy, eps, num_episodes))
    #     lines.append(mlines.Line2D([], [], color=color[i], marker='8', label="eps: {0}".format(eps)))
    #
    # print "Running : Performance based on No. of Episodes and Epsilon" .format(eps)
    # p = multiprocessing.Pool(multiprocessing.cpu_count())
    # p_iter = [p.apply_async(algorithms.sarsaTD, param) for param in parameters]
    # p.close()
    # p.join()
    #
    # print "Performance based on No. of Episodes and Epsilon ---- Complete"
    # plt.figure(2)
    # plt.title("Iteration Performance based on exploration and number of episodes")
    # plt.xlabel('Number of Iterations')
    # plt.ylabel('V Value')
    #
    # for i, value in enumerate(p_iter):
    #
    #     _, _, V_test = value.get()
    #
    #     plt.plot(num_iters, V_test, color[i], marker='8')
    #
    #     for i, j in zip(num_iters, V_test):
    #         plt.annotate(str(j), xy=(i, j))
    #
    # plt.legend(handles=lines)

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
testAlgoParam(env, policy, 2000)
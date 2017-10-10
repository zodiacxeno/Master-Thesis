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
def testAlgoParam():

    num_iters = np.linspace(10000, 100000, 10)
    epsilon = np.linspace(0.0, 1.0, 5)


    color=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    lines = []
    for i, eps in enumerate(epsilon):
        parameters = []
        V_iter = []
        for num in num_iters:
            parameters.append((env, policy, eps, int(num)))

        p = multiprocessing.Pool(multiprocessing.cpu_count())
        p_iter = [p.apply_async(algorithms.monteCarloPredictor, param) for param in parameters]
        p.close()
        p.join()
        for value in p_iter:
            _, _, V_test = value.get()
            V_iter.append(V_test)

        plt.figure(1)
        plt.subplot(121).set_title("Iteration Performance (e = 0.1)")
        plt.xlabel('Number of Iterations')
        plt.ylabel('V Value')
        lines.append(mlines.Line2D([], [], color=color[i], marker='8', label="eps: {0}" .format(eps)))

        plt.plot(num_iters, V_iter, color[i])

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

testAlgoParam()
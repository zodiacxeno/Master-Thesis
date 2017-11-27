#!/usr/bin/env python
import rospy

from move_base_msgs.msg import MoveBaseActionGoal
from std_srvs.srv import Empty
import tf
import algorithms
import math
import gridEnv
from collections import namedtuple
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
# def goalPublisher(x,y,z):
#     pub = rospy.Publisher('/robot_1/move_base/goal', MoveBaseActionGoal, queue_size=10)
#     rospy.init_node('goalPublisher',anonymous=True)
#
#     q = tf.transformations.quaternion_from_euler(0,0,3.14)
#     goal = MoveBaseActionGoal()
#     goal.goal.target_pose.header.frame_id = "map"
#     goal.goal.target_pose.pose.position.x = x
#     goal.goal.target_pose.pose.position.y = y
#     goal.goal.target_pose.pose.position.z = z
#
#     goal.goal.target_pose.pose.position.orientation.x = q[0]
#     goal.goal.target_pose.pose.position.orientation.y = q[1]
#     goal.goal.target_pose.pose.position.orientation.z = q[2]
#     goal.goal.target_pose.pose.position.orientation.w = q[3]
#     rospy.loginfo(goal)
#     pub.publish(goal)

State = namedtuple("State", ["static_location", "dynamic_location"])
# grid = np.array([['o', 'o', 'o', 'o', '*'],
#                  ['o', 'o', 'o', 'o', '*'],
#                  ['#', '#', '#', 'o', '#'],
#                  ['#', '#', '#', 'o', '#'],
#                  ['o', 'o', 'o', 'o', 'o'],
#                  ['o', 'o', 'o', 'o', 'o']], dtype='S4')






def planner(env,policy):

    listener = tf.TransformListener()
    rate = rospy.Rate(10.0)
    listener.waitForTransform("map", "robot_1/base_link", rospy.Time(), rospy.Duration(4.0))

    while not rospy.is_shutdown():

        algorithms.monteCarloPredictor(env, policy, 0.1, 1000)
        rate.sleep()
        rospy.loginfo(location)


if __name__ == '__main__':
    rospy.init_node('rl_planner', anonymous=True)
    policy = {}
    grid = np.empty(shape=(8,13))
    env = gridEnv.gridWorld(grid)
    num_episodes = 1000
    intervals = 10
    num_iters = np.linspace(0, num_episodes-1, intervals)
    lines = []
    color = ['C0', 'C1']




    for i in range(grid.size):
        for j in range(grid.size):
            policy[State(i, j)] = [0.0, 1.0]

    V_baseline = []
    for i in range(intervals):

        test_state = env.initializeState(State(79, 23))
        state_return = 0
        for test_count in range(0, 100):

            current_test_action = np.argmax(policy[test_state])
            next_test_state, reward, test_done = env.executeAction(test_state, current_test_action)
            state_return += reward
            if test_done:
                break
            test_state = next_test_state
        V_baseline.append(state_return)

    lines.append(mlines.Line2D([], [], color='r', marker='8', label='Worst-case Baseline Performance'))
    lines.append(mlines.Line2D([], [], color='b', marker='8', label='Learning Performance'))
    plt.title("Iteration Performance based on number of epsiodes (e = 0.2)")
    plt.xlabel('Number of Iterations')
    plt.ylabel('V Value')

    plt.errorbar(num_iters, [np.mean(V_baseline)]*intervals, [np.std(V_baseline)]*intervals, linestyle='solid', color='r', label='Worst-case Baseline Performance')



    for i in range(grid.size):
        for j in range(grid.size):
            policy[State(i, j)] = [0.5, 0.5]

    p, v, v_trend=algorithms.nStepSarsa(env, policy, 0.1, num_episodes, 5)
    v_trend_mean = zip(*v_trend)[0]
    v_trend_std = zip(*v_trend)[1]

    plt.errorbar(num_iters, v_trend_mean, v_trend_std, linestyle='solid', color='b', marker='8', label='Learning Performance')
    for i, j in zip(num_iters, v_trend_mean):
        plt.annotate(str(j), xy=(i, j))
    plt.legend()
    plt.show()
    print p,v
    #
    # while True:
    #     robot_state = input('Enter start location')
    #     dynamic_object_state = 19
    #     state = State(robot_state, dynamic_object_state)
    #     env.initializeState(state)
    #
    #     while True:
    #         action = np.argmax(p[state])
    #         print state, action
    #         next_state, reward, done = env.executeAction(state, action)
    #         if done:
    #             break
    #         state = next_state
    # state = env.initializeState()
    #
    #
    # print state
    # while not rospy.is_shutdown():
    #     action = input('Provide an action')
    #     if action != 0 and action!=1:
    #         rospy.on_shutdown()
    #     state, reward, done = env.executeAction(state, action)
    #     print state, reward, done




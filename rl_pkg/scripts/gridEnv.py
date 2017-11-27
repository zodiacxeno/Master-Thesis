from collections import namedtuple
import copy
import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, PoseStamped
from std_srvs.srv import Empty
from actionlib_msgs.msg import GoalStatusArray
from stage_ros.srv import SetGlobalPose
import tf
import math
import time
State = namedtuple("State", ["static_location", "dynamic_location"])

class gridWorld:

    def __init__(self, grid):

        self.grid = grid
        # self.actions = np.array(['up', 'down', 'left', 'right', 'wait'])  # Normal Mode
        self.actions = np.array(['proceed_to_wait', 'proceed_to_goal'])     # Controller Mode
        self.states = np.array(range(0, self.grid.size))
        self.rewards = np.zeros(self.grid.shape)
        self.goal_location = [(1,11)]
        self.wait_locations = [(4,8)]
        self.obstacles = []
        self.initial_pose = rospy.ServiceProxy('/global_pose', SetGlobalPose)
        self.step = rospy.ServiceProxy('/step', Empty)
        self.clear_costmaps = rospy.ServiceProxy('/robot_1/move_base/clear_costmaps', Empty)
        self.map_info = rospy.Subscriber('/robot_1/map', OccupancyGrid, self.getObstacleInfo)

        rospy.wait_for_message('/robot_1/map',OccupancyGrid)

        self.listener = tf.TransformListener()
        self.previous_action = None
        self.previous_robot_pose = None
        self.pub = rospy.Publisher('/robot_1/move_base_simple/goal', PoseStamped, queue_size=10)


    def computeShortestPath(self, start, goals):
        nodes = []
        distances = dict()
        for x in range(0, np.shape(self.grid)[0]):
            for y in range(0, np.shape(self.grid)[1]):
                n = (x, y)
                nodes.append(n)
                distances[n] = dict()
                if (x > 0):
                    neighbor = (x - 1, y)
                    if neighbor not in self.obstacles:
                        distances[n][neighbor] = 1

                if (x < np.shape(self.grid)[0] - 1):
                    neighbor = (x + 1, y)
                    if neighbor not in self.obstacles:
                        distances[n][neighbor] = 1

                if (y > 0):
                    neighbor = (x, y - 1)
                    if neighbor not in self.obstacles:
                        distances[n][neighbor] = 1

                if (y < np.shape(self.grid)[1] - 1):
                    neighbor = (x, y + 1)
                    if neighbor not in self.obstacles:
                        distances[n][neighbor] = 1

        goal_distances = {}
        for goal in goals:
            unvisited = {node: None for node in nodes}  # using None as +inf
            visited = {}
            currentDistance = 0
            current = goal
            unvisited[current] = currentDistance
            while True:
                for neighbour, distance in distances[current].items():
                    if neighbour not in unvisited: continue
                    newDistance = currentDistance + distance
                    if unvisited[neighbour] is None or unvisited[neighbour] > newDistance:
                        unvisited[neighbour] = newDistance
                visited[current] = currentDistance
                del unvisited[current]
                if not unvisited: break
                candidates = [node for node in unvisited.items() if node[1]]
                if not candidates:
                    break
                current, currentDistance = sorted(candidates, key=lambda x: x[1])[0]
            goal_distances[goal] = visited[start]

        goal = min(goal_distances, key=goal_distances.get)
        return goal

    def location_to_state(self, robot_x, robot_y, dynamic_x, dynamic_y):
        robot_location = ((self.grid.shape[0] - 1) - math.floor(robot_y / (8. / self.grid.shape[0])),
                            math.floor(robot_x / (12.8 / self.grid.shape[1])))

        dynamic_object_location = ((self.grid.shape[0] - 1) - math.floor(dynamic_y / (8. / self.grid.shape[0])),
                          math.floor(dynamic_x / (12.8 / self.grid.shape[1])))

        robot_state = int(self.grid.shape[1] * robot_location[0] + robot_location[1])
        dynamic_object_state = int(self.grid.shape[1] * dynamic_object_location[0] + dynamic_object_location[1])

        print robot_location, dynamic_object_location
        return State(robot_state, dynamic_object_state), robot_location, dynamic_object_location

    def getObstacleInfo(self, data):

        obstacles = []
        count=0
        occupancy_info = data.data
        width = data.info.width
        height = data.info.height
        grid_resolution = (math.ceil(float(height) / self.grid.shape[0]), math.ceil(float(width) / self.grid.shape[1]))

        resolution = data.info.resolution


        for index, value in enumerate(occupancy_info):

            if value != 0:
                pixel_row = (index / width)
                row_index = self.grid.shape[0] - int(pixel_row / grid_resolution[0]) - 1
                pixel_column = index % width
                column_index = int(pixel_column / grid_resolution[1])

                grid_location = (row_index, column_index)
                obstacles.append(grid_location)

        self.obstacles = set(obstacles)
        print self.obstacles
        # if len(obstacles):
        #     self.map_info.unregister()



    def generateState(self):
        # if not len(self.obstacles):
        #     self.map_info = rospy.Subscriber('/robot_1/map', OccupancyGrid, self.getObstacleInfo)
        obstacles = []
        goal = []
        free_locations = []
        for obstacles_x, obstacles_y in self.obstacles:
            obstacles.append(self.grid.shape[1] * obstacles_x + obstacles_y)
        for goal_x, goal_y in self.goal_location:
            goal.append(self.grid.shape[1] * goal_x + goal_y)

        for i in range(self.grid.size):
            if i not in obstacles and i not in goal:
                free_locations.append(i)

        while True:
            robot_location = np.random.choice(free_locations, p=[1. / len(free_locations)] * len(free_locations))
            dynamic_object_location = 19
            state = State(robot_location, dynamic_object_location)
            if state.static_location != state.dynamic_location or state.static_location not in goal or state.dynamic_location not in goal:
                break
        return state


    def initializeState(self, state=None):


        self.rewards[:] = -1

        self.rewards[zip(*self.goal_location)] = 10

        state_to_location = np.reshape(np.arange(self.grid.size), self.grid.shape)
        if state == None:
            state = self.generateState()

        # for i in range(self.grid.size):
        #     state= State(i)

        for i,j in zip(*np.where(state_to_location == state.static_location)):
            initialRobotState = (i,j)




        robot_x_initial = (12.8 / self.grid.shape[1]) * (initialRobotState[1] + 0.5)
        robot_y_initial = (8. / self.grid.shape[0]) * (self.grid.shape[0] - initialRobotState[0] - 0.5)


        robot_initial_pose_msg = Pose()
        robot_id = "robot_1"

        robot_initial_pose_msg.position.x = robot_x_initial
        robot_initial_pose_msg.position.y = robot_y_initial
        robot_initial_pose_msg.position.z = 0.02
        robot_initial_pose_msg.orientation.x = 0
        robot_initial_pose_msg.orientation.y = 0
        robot_initial_pose_msg.orientation.z = 0
        robot_initial_pose_msg.orientation.w = 1

        self.initial_pose(robot_id, robot_initial_pose_msg)
        self.step()

        for i,j in zip(*np.where(state_to_location == state.dynamic_location)):
            initialDynamicState = (i,j)

        dynamic_x_initial = (12.8 / self.grid.shape[1]) * (initialDynamicState[1] + 0.5)
        dynamic_y_initial = (8. / self.grid.shape[0]) * (self.grid.shape[0] - initialDynamicState[0] - 0.5)

        dynamic_initial_pose_msg = Pose()
        dynamic_object_id = "pedestrian"

        dynamic_initial_pose_msg.position.x = dynamic_x_initial
        dynamic_initial_pose_msg.position.y = dynamic_y_initial
        dynamic_initial_pose_msg.position.z = 0.02
        dynamic_initial_pose_msg.orientation.x = 0
        dynamic_initial_pose_msg.orientation.y = 0
        dynamic_initial_pose_msg.orientation.z = 0
        dynamic_initial_pose_msg.orientation.w = 1

        self.initial_pose(dynamic_object_id, dynamic_initial_pose_msg)

        # initial_state = PoseWithCovarianceStamped()
        # initial_state.header.frame_id = "map"
        # initial_state.header.stamp = rospy.Time.now()
        # initial_state.pose.pose.position.x = x_initial
        # initial_state.pose.pose.position.y = y_initial
        # initial_state.pose.pose.position.z = 0.02
        #
        # initial_state.pose.pose.orientation.x = 0
        # initial_state.pose.pose.orientation.y = 0
        # initial_state.pose.pose.orientation.z = 0
        # initial_state.pose.pose.orientation.w = 0.99
        # initial_state.pose.covariance[0] = 0.25
        # initial_state.pose.covariance[7] = 0.25
        # initial_state.pose.covariance[35] = 0.06



        # rospy.loginfo(initial_state)
        # # step()
        # #time.sleep(1)
        # print "Send message"
        # pose_pub.publish(initial_state)
        # print "Done"
        #time.sleep(1)

        self.step()


        state, robot_location, dynamic_object_location = self.location_to_state(robot_initial_pose_msg.position.x, robot_initial_pose_msg.position.y, dynamic_initial_pose_msg.position.x, dynamic_initial_pose_msg.position.y)
        print robot_location, dynamic_object_location
        return state

    def executeAction(self, state, action):


        state_to_location = np.reshape(np.arange(self.grid.size), self.grid.shape)
        for i,j in zip(*np.where(state_to_location == state.static_location)):
            current_location = (i,j)
        self.listener.waitForTransform("map", "pedestrian/base_link", rospy.Time(), rospy.Duration(4.0))
        (dyn_trans, dyn_rot) = self.listener.lookupTransform("map", "pedestrian/base_link", rospy.Time(0))
        if action == 0:
            if self.previous_action != action:
                nearest_wait_location = self.wait_locations
                wait_x = (12.8 / self.grid.shape[1]) * (nearest_wait_location[0][1] + 0.5)
                wait_y = (8. / self.grid.shape[0]) * (self.grid.shape[0] - nearest_wait_location[0][0] - 0.5)


                q = tf.transformations.quaternion_from_euler(0, 0, 0)
                goal = PoseStamped()
                goal.header.frame_id = "map"
                goal.pose.position.x = wait_x
                goal.pose.position.y = wait_y
                goal.pose.position.z = 0.02

                goal.pose.orientation.x = q[0]
                goal.pose.orientation.y = q[1]
                goal.pose.orientation.z = q[2]
                goal.pose.orientation.w = q[3]

                self.pub.publish(goal)



            if current_location not in self.wait_locations:
                self.step()
                (robot_trans, robot_rot) = self.listener.lookupTransform("map", "robot_1/base_link", rospy.Time(0))
                next_state, next_robot_location, next_dynamic_location = self.location_to_state(robot_trans[0], robot_trans[1], dyn_trans[0], dyn_trans[1])

            else:
                self.step()
                (robot_trans, robot_rot) = self.listener.lookupTransform("map", "robot_1/base_link", rospy.Time(0))
                next_state, next_robot_location, next_dynamic_location = self.location_to_state(robot_trans[0], robot_trans[1], dyn_trans[0], dyn_trans[1])


        elif action == 1:
            if self.previous_action != action:

                q = tf.transformations.quaternion_from_euler(0, 0, 0)
                goal = PoseStamped()
                goal.header.frame_id = "map"

                goal.pose.position.x = 11.0
                goal.pose.position.y = 6.66
                goal.pose.position.z = 0.02

                goal.pose.orientation.x = q[0]
                goal.pose.orientation.y = q[1]
                goal.pose.orientation.z = q[2]
                goal.pose.orientation.w = q[3]

                self.pub.publish(goal)

            if current_location not in self.goal_location:
                self.step()
                (robot_trans, robot_rot) = self.listener.lookupTransform("map", "robot_1/base_link", rospy.Time(0))
                next_state, next_robot_location, next_dynamic_location = self.location_to_state(robot_trans[0], robot_trans[1], dyn_trans[0], dyn_trans[1])

            else:
                self.step()
                (robot_trans, robot_rot) = self.listener.lookupTransform("map", "robot_1/base_link", rospy.Time(0))
                next_state, next_robot_location, next_dynamic_location = self.location_to_state(robot_trans[0], robot_trans[1], dyn_trans[0], dyn_trans[1])


        self.previous_action = action

        if next_robot_location in self.goal_location and next_robot_location not in self.wait_locations: #self.previous_robot_pose == (robot_trans, robot_rot)
            done = True
        else:
            done = False
        connected_states = [next_state.static_location + 1, next_state.static_location - 1, next_state.static_location - self.grid.shape[1], next_state.static_location + self.grid.shape[1]]
        if next_state.static_location == next_state.dynamic_location or next_state.dynamic_location in connected_states:

            reward = -80
        else:
            rewards = self.rewards.reshape(self.grid.size)
            reward = rewards[next_state.static_location]

        self.previous_robot_pose = (robot_trans, robot_rot)

        return next_state, reward, done
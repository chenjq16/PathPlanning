"""
RRT_2D
@author: huiming zhou
"""

import os
import sys
import math
import numpy as np

import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Sampling_based_Planning/")

from rrt_2D import env, plotting, utils


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class DmpRrt:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, iter_max, ref_path):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.iter_max = iter_max
        self.vertex = [self.s_start]

        self.env = env.Env()
        self.plotting = plotting.Plotting(s_start, s_goal)
        self.utils = utils.Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        self.ref_path = ref_path
        ref_len = len(ref_path)
        print(f"Reference path length: {ref_len}")
        
    def planning(self):
        start_time = time.time()

        p = 1

        for i in range(self.iter_max):
            if p < len(self.ref_path):
                node_rand = Node(self.ref_path[p])
                if not self.utils.is_inside_obs(node_rand):
                    node_near = self.nearest_neighbor(self.vertex, node_rand)
                    if not self.utils.is_collision(node_near, node_rand):
                        node_new = self.new_dmp_state(node_near, node_rand)
                        p += 1
                        print("Not inside obstacle")
                    else:
                        node_rand = self.generate_dmp_random_node(self.goal_sample_rate, Node(self.ref_path[p]))
                        node_near = self.nearest_neighbor(self.vertex, node_rand)
                        node_new = self.new_state(node_near, node_rand)
                    
                        dmp_dist, _ = self.get_distance_and_angle(node_new, Node(self.ref_path[p]))
                        if dmp_dist <= self.step_len and not self.utils.is_collision(node_new, Node(self.ref_path[p])) and not self.utils.is_collision(node_near, node_new):
                            p += 1
                            self.vertex.append(node_new)
                            node_near = node_rand
                            node_new = self.new_dmp_state(node_rand, Node(self.ref_path[p]))
                            print("Not inside obstacle, but collision, reached goal")
                        else:
                            print("Not inside obstacle, but collision")
                        
                else:
                    p += 1
                    print("Inside obstacle")
                    continue
            else:
                node_rand = self.generate_random_node(self.goal_sample_rate)
                node_near = self.nearest_neighbor(self.vertex, node_rand)
                node_new = self.new_state(node_near, node_rand)

            if node_new and not self.utils.is_collision(node_near, node_new):
                print(p)
                self.vertex.append(node_new)
                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)

                if dist <= self.step_len and not self.utils.is_collision(node_new, self.s_goal):
                    self.new_state(node_new, self.s_goal)
                    print("Reached goal: ", i)
                    end_time = time.time()
                    print(f"Planning time: {end_time - start_time} seconds")
                    return self.extract_path(node_new)
        
        end_time = time.time()
        print(f"Planning time: {end_time - start_time} seconds")

        return None

    def new_dmp_state(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)

        # dist = min(self.step_len, dist)
        # node_new = Node((node_start.x + dist * math.cos(theta),
        #                  node_start.y + dist * math.sin(theta)))
        node_new = node_goal

        node_new.parent = node_start

        return node_new
    
    def generate_dmp_random_node(self, goal_sample_rate, dmp_goal):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta)*0.3 + dmp_goal.x*0.7,
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)*0.3 + dmp_goal.y*0.7))

        return dmp_goal

    def generate_random_node(self, goal_sample_rate):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start

        return node_new

    def extract_path(self, node_end):
        path = [(self.s_goal.x, self.s_goal.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)


def main():
    x_start = (10, 8)  # Starting node
    x_goal = (37, 18)  # Goal node

    # Calculate the increments
    num_parts = 1000
    dx = (x_goal[0] - x_start[0]) / num_parts
    dy = (x_goal[1] - x_start[1]) / num_parts

    # Generate the points
    ref_path = [(x_start[0] + i * dx, x_start[1] + i * dy) for i in range(num_parts + 1)]

    rrt = DmpRrt(x_start, x_goal, 0.5, 0.05, 10000, np.array(ref_path))
    path = rrt.planning()

    if path:
        rrt.plotting.animation(rrt.vertex, path, "dmp_RRT", True)
        rrt.plotting.animation_ref(rrt.vertex, path, "dmp_RRT", ref_path=np.array(ref_path))
    else:
        print("No Path Found!")


if __name__ == '__main__':
    main()

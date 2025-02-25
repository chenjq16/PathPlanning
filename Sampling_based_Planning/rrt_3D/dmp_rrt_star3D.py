"""
This is rrt star code for 3D
@author: yue qi
"""
import numpy as np
from numpy.matlib import repmat
from collections import defaultdict
import time
import matplotlib.pyplot as plt

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../Sampling_based_Planning/")
from rrt_3D.env3D import env
from rrt_3D.utils3D import getDist, sampleFree, nearest, steer, isCollide, near, visualization, cost, path, isinside, dmp_visualization


class dmprrtstar():
    def __init__(self):
        self.env = env()

        self.Parent = {}
        self.V = []
        # self.E = edgeset()
        self.COST = {}

        self.i = 0
        self.maxiter = 2500 # at least 2000 in this env
        self.stepsize = 2
        self.gamma = 7
        self.eta = self.stepsize
        self.Path = []
        self.done = False
        self.x0 = tuple(self.env.start)
        self.xt = tuple(self.env.goal)

        self.V.append(self.x0)
        self.ind = 0

        self.ref_path = []

    def wireup(self,x,y):
        # self.E.add_edge([s,y]) # add edge
        self.Parent[x] = y

    def removewire(self,xnear):
        xparent = self.Parent[xnear]
        a = [xnear,xparent]
        # self.E.remove_edge(a) # remove and replace old the connection

    def reached(self):
        self.done = True
        goal = self.xt
        xn = near(self,self.env.goal)
        c = [cost(self,tuple(x)) for x in xn]
        xncmin = xn[np.argmin(c)]
        self.wireup(goal , tuple(xncmin))
        self.V.append(goal)
        self.Path,self.D = path(self)

    def run(self):
        xnew = self.x0
        print('start dmp_rrt*... ')
        self.fig = plt.figure(figsize = (10,8))

        p = 1

        while self.ind < self.maxiter:
            if p < len(self.ref_path):
                x = self.ref_path[p]
                if not isinside(self, x):
                    xrand = x
                    xnearest = nearest(self,xrand)
                    xnew, dist  = steer(self,xnearest,xrand)
                    collide, _ = isCollide(self,xnearest,xnew,dist=dist)                    
                    if not collide:
                        p += 1
                        print("Not inside obstacle")
                    else:
                        xrand = sampleFree(self)
                else:
                    p += 1
                    print("Inside obstacle")
                    continue

            else:
                xrand = sampleFree(self)
            xnearest = nearest(self,xrand)
            xnew, dist  = steer(self,xnearest,xrand)
            collide, _ = isCollide(self,xnearest,xnew,dist=dist)
            if not collide:
                Xnear = near(self,xnew)
                self.V.append(xnew) # add point
                visualization(self)
                plt.title('dmp_rrt*')
                # minimal path and minimal cost
                xmin, cmin = xnearest, cost(self, xnearest) + getDist(xnearest, xnew)
                # connecting along minimal cost path
                Collide = []
                for xnear in Xnear:
                    xnear = tuple(xnear)
                    c1 = cost(self, xnear) + getDist(xnew, xnear)
                    collide, _ = isCollide(self, xnew, xnear)
                    Collide.append(collide)
                    if not collide and c1 < cmin:
                        xmin, cmin = xnear, c1
                self.wireup(xnew, xmin)
                # rewire
                for i in range(len(Xnear)):
                    collide = Collide[i]
                    xnear = tuple(Xnear[i])
                    c2 = cost(self, xnew) + getDist(xnew, xnear)
                    if not collide and c2 < cost(self, xnear):
                        # self.removewire(xnear)
                        self.wireup(xnear, xnew)
                self.i += 1
            self.ind += 1
        # max sample reached
        self.reached()
        print('time used = ' + str(time.time()-starttime))
        print('Total distance = '+str(self.D))
        dmp_visualization(self)
        plt.show()

    def generate_ref(self):
        # Calculate the increments
        num_parts = 1000
        dx = (self.env.goal[0] - self.env.start[0]) / num_parts
        dy = (self.env.goal[1] - self.env.start[1]) / num_parts
        dz = (self.env.goal[2] - self.env.start[2]) / num_parts

        # Generate the points
        self.ref_path = np.array([[self.env.start[0] + i * dx, self.env.start[1] + i * dy, self.env.start[2] + i * dz] for i in range(num_parts + 1)])

        

if __name__ == '__main__':
    p = dmprrtstar()
    starttime = time.time()
    p.generate_ref()
    p.run()
    

from typing import Tuple
import numpy as np
from numpy import array
import search

import time

# you can want to use the class registration_iasd
# from your solution.py (from previous assignment)
from solution import registration_iasd

# Choose what you think it is the best data structure
# for representing actions.
Action = float

# Choose what you think it is the best data structure
# for representing states.
class myState():
    def __init__(self, actualState: float, prevAction: float, transformedScan: array((...,3)) ) -> None:
        self.actualState = actualState
        self.prevAction = prevAction
        self.transformedScan = transformedScan

State = myState

class align_3d_search_problem(search.Problem):

    def __init__(self, scan1: array((...,3)), scan2: array((...,3))) -> None:
        """Module that instantiate your class.
        You CAN change the content of this __init__ if you want.

        :param scan1: input point cloud from scan 1
        :type scan1: np.array
        :param scan2: input point cloud from scan 2
        :type scan2: np.array
        """

        # Creates an initial state.
        # You may want to change this to something representing
        # your initial state.
                      
        # registration = registration_iasd(scan1, scan2)
        # correspondencies = registration.find_closest_points()
        # dists = []
        # for element in correspondencies.values():
        #     dists.append(element['dist2'])

        # avg_dist = np.average(np.array(dists))
        
        # print(avg_dist)
        
        # self.initial["scan1"] = scan1
        # self.initial["scan2"] = scan2
        # self.initial["dist"] = np.average( np.array(dists) )

        self.initial = myState(0, 0, scan1)
        self.scan1 = scan1
        self.goal = scan2
        self.center = np.average(scan1, axis=0)
        
        return


    def actions(
            self,
            state: State
            ) -> Tuple:
        """Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment

        :param state: Abstract representation of your state
        :type state: State
        :return: Tuple with all possible actions
        :rtype: Tuple
        """
        
        action = []
        if state.actualState == 0:
            action.append( ["x",-np.radians( 1 )] )
            #action.append( ["x",-np.radians( 1 ) ] )

            action.append( ["y",-np.radians( 1  )] )
            #action.append( ["y",-np.radians( 1 )] )

            action.append( ["z", -np.radians( 1  )] )
            #action.append( ["z",-np.radians( 1 )] )
        else:
            action.append( ["x" ,state.actualState - np.radians( 1  )] )
            #action.append( ["x", state.actualState - np.radians( 1 )] )

            action.append( ["y" ,state.actualState - np.radians( 1  )] )
            #action.append( ["y",state.actualState - np.radians( 1 )] )
            
            action.append( ["z" ,state.actualState - np.radians( 1  )] )
            #action.append( ["z",state.actualState - np.radians( 1 )] )

        # action = [state.actualState + np.radians( 0.01 ), state.actualState + np.radians( 0.02 )]

        return tuple(action)

    def result(
            self,
            state: State,
            action: Action
            ) -> State:
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: Abstract representation of your state
        :type state: [type]
        :param action: An action
        :type action: [type]
        :return: A new state
        :rtype: State
        """
        #print(action)
        # R = angleToMatrix(state.actualState, state.actualState, state.actualState)
        print(action)
        if action[0] == "x":
            R = eulerAnglesToRotationMatrix([action[1], 0, 0])
        elif action[0] == "y":
            R = eulerAnglesToRotationMatrix([0, action[1], 0])
        elif action[0] == "z":
            R = eulerAnglesToRotationMatrix([0, 0, action[1]]) 
        

        T = np.vstack( (np.column_stack( ( R, 
            [self.center[0]-R[0][0]*self.center[0] - R[0][1]*self.center[1] - R[0][2]*self.center[2],
            self.center[1]-R[1][0]*self.center[0] - R[1][1]*self.center[1] - R[1][2]*self.center[2],
            self.center[2]-R[2][0]*self.center[0] - R[2][1]*self.center[1] - R[2][2]*self.center[2]]) ), 
            [0,0,0,1] ) )
        
        #here we apply the rotation and translation to the actual points
        r = T[0:3,0:3]
        t = T[0:3, 3]
        #print("R:")
        #print(r)
        #print("T:")
        #print(t)

        num_points, _ = self.scan1.shape
        transformedScan = (
                        np.dot(r, self.scan1.T) +
                        np.dot(t.reshape((3,1)),np.ones((1,num_points)))
                        ).T
        
        result = myState(action[1], state.actualState, transformedScan)
        # result = state.actualState + action
        # resultState = myAction(np.dot(action.getMatrix(), state.getMatrix()))

        return result


    def goal_test(
            self,
            state: State
            ) -> bool:
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough.

        :param state: gets as input the state
        :type state: State
        :return: returns true or false, whether it represents a node state or not
        :rtype: bool
        """
        
        # print(f"Im testing if this is goal: rotation of {np.degrees(state.actualState)} degrees")
        
        
                
        registration = registration_iasd(state.transformedScan, self.goal)
        correspondencies = registration.find_closest_points()
        # dists = []
        # for element in correspondencies.values():
        #     dists.append(element['dist2'])

        # avg_dist = np.average( np.array(dists) )
        
        # print(avg_dist)

        error = sum(correspondence['dist2']**2 for correspondence in correspondencies.values())
        # if error <0.001:
        #print(error)
        #time.sleep(1)
        
        return (error < 0.0001)#1E-16)


    def path_cost(
            self,
            c,
            state1: State,
            action: Action,
            state2: State
            ) -> float:
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path.

        :param c: cost to get to the state1
        :type c: [type]
        :param state1: parent node
        :type state1: State
        :param action: action that changes the state from state1 to state2
        :type action: Action
        :param state2: state2
        :type state2: State
        :return: [description]
        :rtype: float
        """

        pass

import math
# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

#Converts the euler angles to a rotation matrix to be applied to the clouds
def angleToMatrix(theta, phi, psi):
    # Rz_phi = np.array([[1,0,0],
    #                 [0, np.cos(phi), -np.sin(phi)],
    #                 [0, np.sin(phi), np.cos(phi)]])

    # Ry = np.array([[np.cos(theta),0,np.sin(theta)],
    #                 [0, 1, 0],
    #                 [-np.sin(theta), 0, np.cos(theta)]])

    # Rz_psi = np.array([[1,0,0],
    #                 [0, np.cos(psi), -np.sin(psi)],
    #                 [0, np.sin(psi), np.cos(psi)]])

    # print(np.linalg.det(Rz_phi),np.linalg.det(Ry),np.linalg.det(Rz_psi))

    # R_mult = np.matmul(np.matmul(Rz_phi, Ry), Rz_psi)
    # print(R_mult)
    # print(np.linalg.det(R_mult))
    
    R = np.array([[np.cos(phi), np.sin(theta)*np.sin(psi), np.sin(theta)*np.cos(psi)],
                [np.sin(theta)*np.sin(phi), np.cos(psi)*np.cos(phi)-np.cos(theta)*np.sin(psi)*np.sin(phi), 
                        -np.cos(phi)*np.sin(psi)-np.cos(theta)*np.cos(psi)*np.sin(phi)],
                [-np.sin(theta)*np.cos(phi), np.cos(psi)*np.sin(phi)+np.cos(theta)*np.cos(phi)*np.sin(psi),
                        -np.sin(psi)*np.sin(phi)+np.cos(theta)*np.cos(psi)*np.cos(phi)]])
    return R

def compute_alignment(
        scan1: array((...,3)),
        scan2: array((...,3)),
        ) -> Tuple[bool, array, array, int]:
    """Function that will return the solution.
    You can use any UN-INFORMED SEARCH strategy we study in the
    theoretical classes.

    :param scan1: first scan of size (..., 3)
    :type scan1: array
    :param scan2: second scan of size (..., 3)
    :type scan2: array
    :return: outputs a tuple with: 1) true or false depending on
        whether the method is able to get a solution; 2) rotation parameters
        (numpy array with dimension (3,3)); 3) translation parameters
        (numpy array with dimension (3,)); and 4) the depth of the obtained
        solution in the proposes search tree.
    :rtype: Tuple[bool, array, array, int]
    """
    
    # #find the center of each cloud
    cloud1_center = np.average(scan1, axis=0)
    cloud2_center = np.average(scan2, axis=0)

    print(f"cloud1 center = {cloud1_center}\ncloud2 center = {cloud2_center}")

    dist = cloud2_center - cloud1_center

    #Move center of scan1 to center of scan2
    scan1 = scan1 + dist

    # cloud1_center = np.average(scan1, axis=0)
    # cloud2_center = np.average(scan2, axis=0)

    # print(f"cloud1 center = {cloud1_center}\ncloud2 center = {cloud2_center}")

    align_problem = align_3d_search_problem(scan1, scan2)
    
    # ret = align_problem.goal_test(scan1)
    print("search")
    ret = search.breadth_first_tree_search(align_problem)
    print(ret.state)

    # R = angleToMatrix(-0.1,0,0)
    # R = np.eye(3)

    # R = angleToMatrix(0,0,np.pi)
    # R = eulerAnglesToRotationMatrix([0,0,-np.pi/50])

    # # R = matrix[0:3,0:3] #The rotation matrix
    # T = np.array([0,0,0]) #translation
    # num_points, _ = scan1.shape
    # scan1 = (
    #                 np.dot(R, scan1.T) +
    #                 np.dot(T.reshape((3,1)),np.ones((1,num_points)))
    #                 ).T

    # cloud1_center = np.average(scan1, axis=0)
    # cloud2_center = np.average(scan2, axis=0)

    # print(f"cloud1 center = {cloud1_center}\ncloud2 center = {cloud2_center}")
    
    # R = eulerAnglesToRotationMatrix([np.pi/2, np.pi/2, np.pi/2])

    # T = np.vstack( (np.column_stack( ( R, 
    #     [cloud2_center[0]-R[0][0]*cloud2_center[0] - R[0][1]*cloud2_center[1] - R[0][2]*cloud2_center[2],
    #     cloud2_center[1]-R[1][0]*cloud2_center[0] - R[1][1]*cloud2_center[1] - R[1][2]*cloud2_center[2],
    #     cloud2_center[2]-R[2][0]*cloud2_center[0] - R[2][1]*cloud2_center[1] - R[2][2]*cloud2_center[2]]) ), 
    #     [0,0,0,1] ) )

    # #here we apply the rotation and translation to the actual points
    # r = T[0:3,0:3]
    # t = T[0:3, 3]
    # R = np.array([[ 0.99998245, -0.0041712,   0.00420625],
    # [ 0.00418874,  0.99998253, -0.0041712 ],
    # [-0.00418878 , 0.00418874 , 0.99998245]])
    # T = [-0.00032739,  0.00046003, -0.00013457]
    
    R=np.eye(3)
    T=0
    return (True, R, T, 1)

if __name__ == "__main__":
    R = angleToMatrix(0,0,np.pi)
    R2 = eulerAnglesToRotationMatrix([0,0,np.pi])
    print(np.linalg.det(R))
    print(np.linalg.det(R2))
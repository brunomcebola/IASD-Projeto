from typing import Tuple, final
import numpy as np
from numpy import array
import search

import time

# you can want to use the class registration_iasd
# from your solution.py (from previous assignment)
from solution import registration_iasd

# Choose what you think it is the best data structure
# for representing actions.
Action = tuple

# Choose what you think it is the best data structure
# for representing states.
class myState():
    def __init__(self, actualState: float, prevAction: float) -> None:
        self.actualState = actualState
        self.prevAction = prevAction

State = tuple

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
        # #find the center of each cloud
        cloud1_center = np.average(scan1, axis=0)
        cloud2_center = np.average(scan2, axis=0)

        print(f"cloud1 center = {cloud1_center}\ncloud2 center = {cloud2_center}")
        print(f"scan1: {scan1.shape}, scan2 = {scan2.shape}")

        dist = cloud2_center - cloud1_center

        #Move center of scan1 to center of scan2
        scan1 = scan1 + dist

        # cloud1_center = np.average(scan1, axis=0)
        # cloud2_center = np.average(scan2, axis=0)

        # print(f"cloud1 center = {cloud1_center}\ncloud2 center = {cloud2_center}")

        # exit()

        self.initial = (np.radians(180),np.radians(180),np.radians(180))
        self.scan1 = scan1
        self.goal = scan2
        self.center = cloud2_center
        
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
        
        # action = []
        # if state.actualState == 0 and state.prevAction == 0:
        #     action.append( np.radians(45) )
        #     action.append( -np.radians(45) )
        # else:
        #     action.append( state.prevAction/10 )
        #     action.append( -state.prevAction/10 )

        actions = [(2,1,1),
                    (1,2,1),
                    (1,1,2),
                    (-2,1,1),
                    (1,-2,1),
                    (1,1,-2)]
                    # (0.5,1,1),
                    # (1,0.5,1),
                    # (1,1,0.5),
                    # (-0.5,1,1),
                    # (1,-0.5,1),
                    # (1,1,-0.5)]

        return actions

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
        
        result = (state[0] / action[0], state[1] / action[1], state[2] / action[2])
       
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
        
        # print(f"Im testing if this is goal: rotation of {tuple(map(np.degrees, state))} degrees")
        
        # R = angleToMatrix(state.actualState, state.actualState, state.actualState)
        R = eulerAnglesToRotationMatrix(list(state))

        T = np.vstack( (np.column_stack( ( R, 
            [self.center[0]-R[0][0]*self.center[0] - R[0][1]*self.center[1] - R[0][2]*self.center[2],
            self.center[1]-R[1][0]*self.center[0] - R[1][1]*self.center[1] - R[1][2]*self.center[2],
            self.center[2]-R[2][0]*self.center[0] - R[2][1]*self.center[1] - R[2][2]*self.center[2]]) ), 
            [0,0,0,1] ) )
        
        #here we apply the rotation and translation to the actual points
        r = T[0:3,0:3]
        t = T[0:3, 3]
        # print("R:")
        # print(r)
        # print("T:")
        # print(t)

        num_points, _ = self.scan1.shape
        transformedScan = (
                        np.dot(r, self.scan1.T) +
                        np.dot(t.reshape((3,1)),np.ones((1,num_points)))
                        ).T
                
        registration = registration_iasd(transformedScan, self.goal)
        correspondencies = registration.find_closest_points()

        errors = [correspondence['dist2']**2 for correspondence in correspondencies.values()]
        
        error = np.average(errors)
        # if error <1E-7:
        # print(f"Current error is {error}\n")
        
        # time.sleep(1)
        
        return (error < 0.0001)


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
    align_problem = align_3d_search_problem(scan1, scan2)
    
    # ret = align_problem.goal_test(scan1)
    print("search")
    ret = search.breadth_first_graph_search(align_problem)
    print("After search") 
    
    cloud1_center = np.average(scan1, axis=0)
    cloud2_center = np.average(scan2, axis=0)

    # print(f"cloud1 center = {cloud1_center}\ncloud2 center = {cloud2_center}")
    # print(f"scan1: {scan1.shape}, scan2 = {scan2.shape}")

    dist = cloud2_center - cloud1_center

    # print(f"rotacao: {tuple(map(np.degrees, ret.state))}")

    R = eulerAnglesToRotationMatrix(list(ret.state))

    T = np.vstack( (np.column_stack( ( R, 
        [cloud1_center[0]-R[0][0]*cloud1_center[0] - R[0][1]*cloud1_center[1] - R[0][2]*cloud1_center[2],
        cloud1_center[1]-R[1][0]*cloud1_center[0] - R[1][1]*cloud1_center[1] - R[1][2]*cloud1_center[2],
        cloud1_center[2]-R[2][0]*cloud1_center[0] - R[2][1]*cloud1_center[1] - R[2][2]*cloud1_center[2]]) ), 
        [0,0,0,1] ) )

    print(f"T from search \n{T}", end="\n---------------\n")
    
    r = T[0:3,0:3]
    t = T[0:3, 3]
    # t=np.zeros(3)

    #Transform the scan1 with the r and t matrix found in the search
    num_points, _ = scan1.shape
    transformedScan1 = (
                    np.dot(r, scan1.T) +
                    np.dot(t.reshape((3,1)),np.ones((1,num_points)))
                    ).T
    
    
    #Use the 1st submission algorithm
    reg = registration_iasd(transformedScan1, scan2)

    r1, t1 = reg.get_compute()
    
    R = np.dot(r, r1)
    T = t + t1

    # sub1_T = np.vstack( (np.column_stack( ( r, t) ), [0,0,0,1] ) )

    # print(f"T from sub1 \n{sub1_T}", end="\n---------------\n")
    
    # finalT = np.dot(sub1_T, T)

    # r = finalT[0:3,0:3]
    # t = finalT[0:3, 3]

    num_points, _ = transformedScan1.shape
    transformedScan1 = (
                    np.dot(R, transformedScan1.T) +
                    np.dot(T.reshape((3,1)),np.ones((1,num_points)))
                    ).T

    # print(f"returned T: \n{finalT}", end="\n---------------\n")

    cloud1_center = np.average(transformedScan1, axis=0)
    
    dist = cloud2_center - cloud1_center

    T = T + dist
    

    # print(f"returned T \n{finalT}", end="\n---------------\n")
    return (True, R, T, ret.depth)

if __name__ == "__main__":
    print("cringe")
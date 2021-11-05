from typing import Tuple
import numpy as np
from numpy import array
import search
from math import sqrt, sin, cos
import time

# you can want to use the class registration_iasd
# from your solution.py (from previous assignment)
from solution import registration_iasd


# Choose what you think it is the best data structure
# for representing actions.
Action = tuple

# Choose what you think it is the best data structure
# for representing states.
State = tuple


class align_3d_search_problem(search.Problem):
    def __init__(
        self,
        scan1: array((..., 3)),
        scan2: array((..., 3)),
        maxError: float,
    ) -> None:
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

        # First we try to filter some points so it's not needed
        # to do all the math for huge matrixes.

        # #find the center of each cloud
        cloud1_center = np.average(scan1, axis=0)
        cloud2_center = np.average(scan2, axis=0)
        print(cloud1_center)

        norm1 = ((scan1 - cloud1_center) ** 2).sum(axis=1)
        norm2 = ((scan2 - cloud2_center) ** 2).sum(axis=1)

        self.scan1 = []
        self.goal = []

        # number of points to fetch from each scan
        k = scan1.shape[0] // 10

        # get the k points closest to the center of each scan
        sorted_norm1 = np.sort(norm1)
        sorted_norm2 = np.sort(norm2)

        for id in range(k):
            # scan1
            real_min_idx1 = np.where(norm1 == sorted_norm1[id])
            self.scan1.append(scan1[real_min_idx1[0][0]])
            # scan2
            real_min_idx2 = np.where(norm2 == sorted_norm2[id])
            self.goal.append(scan2[real_min_idx2[0][0]])

        # convert everything to numpy to ensure compatibility
        self.scan1 = np.array(self.scan1)
        self.goal = np.array(self.goal)

        # Our states will be the rotation matrix and the depth
        # of the search
        self.initial = [eulerAnglesToRotationMatrix(np.array([np.pi,np.pi,np.pi])), 0]

        self.center = np.average(self.goal, axis=0)
        self.maxError = maxError

        self.i = 0

        return

    def eval_error(self, testScan) -> float:
        #Same as find_closests_points from previous submission
        #but the return is de average of the distances instead of the
        #correspondencies

        testScan_center = np.average(testScan, axis=0)
        t = self.center - testScan_center
        testScan = testScan + t

        dist_matrix = (self.goal - testScan) ** 2

        dist_matrix = (dist_matrix.sum(axis=1)) ** 0.5

        return np.average(dist_matrix)

    def actions(self, state: State) -> Tuple:
        """Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment

        :param state: Abstract representation of your state
        :type state: State
        :return: Tuple with all possible actions
        :rtype: Tuple
        """
        # Our actions will be rotations in the positive
        # direction and negative direction of the x
        # and y axis

        # Our groupd decided that the depth should be
        # higher than 7 and wanted to try something
        # behond this point

        if state[-1] <= 7:
            actions = [
                #positive rotations
                rotateX(np.pi / pow(2,  1 + state[-1])),
                rotateY(np.pi / pow(2,  1 + state[-1])),
                rotateZ(np.pi / pow(2,  1 + state[-1])),

                #negative rotations
                rotateX(-np.pi / pow(2, 1 + state[-1])),
                rotateY(-np.pi / pow(2, 1 + state[-1])),
                rotateZ(-np.pi / pow(2, 1 + state[-1])),
            ]
        else:
            actions = []

        return tuple(actions)

    def result(self, state: State, action: Action) -> State:
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
        # the result is the multiplication of the old state matrix
        # and the new rotation matrix, this way we search for more
        # angles in the search
        result = [np.dot(state[0], action.T).T, state[-1] + 1]

        return tuple(result)

    def goal_test(self, state: State) -> bool:
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough.

        :param state: gets as input the state
        :type state: State
        :return: returns true or false, whether it represents a node state or not
        :rtype: bool
        """

        # here we apply the rotation and translation to the "filtered points"
        transformedScan = np.dot(state[0], self.scan1.T).T

        return self.eval_error(transformedScan) < self.maxError

    def path_cost(self, c, state1: State, action: Action, state2: State) -> float:
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

        # we didnt define a path cost

        pass

# Calculates the Rotation Matrix for the x axis 
def rotateX(theta):
    return np.array(
        [
            [1, 0, 0],
            [0, cos(theta), -sin(theta)],
            [0, sin(theta), cos(theta)],
        ]
    )

# Calculates the Rotation Matrix for the y axis
def rotateY(theta):
    return np.array(
        [
            [cos(theta), 0, sin(theta)],
            [0, 1, 0],
            [-sin(theta), 0, cos(theta)],
        ]
    )

def rotateZ(theta):
    return np.array(
        [
          [cos(theta), -sin(theta), 0],
             [sin(theta), cos(theta), 0],
             [0, 0, 1],
         ]
    )


# Calculates Rotation Matrix given euler angles.
# Not used in this version of the project
def eulerAnglesToRotationMatrix(theta):

    R_x = np.array(
        [
            [1, 0, 0],
            [0, cos(theta[0]), -sin(theta[0])],
            [0, sin(theta[0]), cos(theta[0])],
        ]
    )

    R_y = np.array(
        [
            [cos(theta[1]), 0, sin(theta[1])],
            [0, 1, 0],
            [-sin(theta[1]), 0, cos(theta[1])],
        ]
    )

    R_z = np.array(
        [
          [cos(theta[2]), -sin(theta[2]), 0],
             [sin(theta[2]), cos(theta[2]), 0],
             [0, 0, 1],
         ]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def compute_alignment(
    scan1: array((..., 3)),
    scan2: array((..., 3)),
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
    # use our search algorithm
    align_problem = align_3d_search_problem(scan1, scan2, 2.5e-2)

    ret = search.breadth_first_tree_search(align_problem)
    if ret == None:
        return (False, np.eye(3), np.zeros(3), 0)

    R = ret.state[0]

    # Transform the scan1 with the r and t matrix found in the search
    transformedScan1 = (np.dot(R, scan1.T)).T

    #dist = np.average(scan2, axis=0) - np.average(transformedScan1, axis=0)

    # Use the 1st submission algorithm
    reg = registration_iasd(transformedScan1, scan2)
    r_2, t_2 = reg.get_compute()

    R = np.dot(r_2, R)

    T = t_2

    return (True, R, T, ret.depth)

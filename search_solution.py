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
        initialAngle: int,
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

        # #find the center of each cloud
        cloud1_center = np.average(scan1, axis=0)
        cloud2_center = np.average(scan2, axis=0)

        dist = cloud2_center - cloud1_center

        # Move center of scan1 to center of scan2
        scan1 = scan1 + dist

        self.initial = (
            np.radians(initialAngle),
            np.radians(initialAngle),
            np.radians(initialAngle),
        )
        self.scan1 = scan1
        self.goal = scan2
        self.center = cloud2_center
        self.maxError = maxError

        self.i = 0

        return

    def find_dist(self, testScan) -> dict:
        """Computes the closest points in the two scans.
        There are many strategies. We are taking all the points in the first scan
        and search for the closes in the second. This means that we can have > than 1 points in scan
        1 corresponding to the same point in scan 2. All points in scan 1 will have correspondence.
        Points in scan 2 do not have necessarily a correspondence.

        :param search_alg: choose the searching option
        :type search_alg: str, optional
        :return: a dictionary with the correspondences. Keys are numbers identifying the id of the correspondence.
                Values are a dictionaries with 'point_in_pc_1', 'point_in_pc_2' identifying the pair of points in the correspondence.
        :rtype: dict
        """
        p_center = np.average(testScan, axis=0)
        t = self.center - p_center
        testScan = testScan + t

        min_dists = []

        for line in testScan:
            dist_matrix = (self.goal - line) ** 2
            dist_matrix = dist_matrix.sum(axis=1)

            min_idx = np.argmin(dist_matrix)

            min_dists.append(sqrt(dist_matrix[min_idx]))

        return np.average(np.array(min_dists))

    def actions(self, state: State) -> Tuple:
        """Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment

        :param state: Abstract representation of your state
        :type state: State
        :return: Tuple with all possible actions
        :rtype: Tuple
        """

        self.i = self.i + 1

      
        return [    
                        (np.radians(40), 0, 0),
                        (0, np.radians(40), 0),
                        (0, 0, np.radians(120)),

                        (-np.radians(40), 0, 0),
                        (0, -np.radians(20), 0),
                        (0, 0, -np.radians(40)),
                        
                    
                    ]                 

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
        result = [state[0] + action[0], state[1] + action[1], state[2] + action[2]]

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
        r = eulerAnglesToRotationMatrix(list(state))
        # here we apply the rotation and translation to the actual points
        t = np.zeros(3)

        num_points, _ = self.scan1.shape
        transformedScan = (
            np.dot(r, self.scan1.T)
            + np.dot(t.reshape((3, 1)), np.ones((1, num_points)))
        ).T

        return self.find_dist(transformedScan) < self.maxError

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

        pass

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    g,b,a = theta
    R = np.array(
        [
            [ cos(a)*cos(b), cos(a)*sin(b)*sin(g) - sin(a)*cos(g), cos(a)*sin(b)*cos(g) + sin(a)*sin(g) ] ,
            [ sin(a)*cos(b), sin(a)*sin(b)*sin(g) + cos(a)*cos(g), sin(a)*sin(b)*cos(g) - cos(a)*sin(g) ],
            [ -sin(b), cos(b)*sin(g), cos(b)*cos(g)]
        ]
    )


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

    # cloud1_center = np.average(scan1, axis=0)
    # cloud2_center = np.average(scan2, axis=0)
    # dist = cloud2_center-cloud1_center

    # transformedScan1 = scan1 + dist

    # # Use the 1st submission algorithm
    # reg = registration_iasd(transformedScan1, scan2)
    # r_1, t_1 = reg.get_compute()

    # # Transform the scan1 with the r and t matrix found in the search
    # num_points, _ = scan1.shape
    # transformedScan1 = (
    #                 np.dot(r_1, scan1.T) +
    #                 np.dot(t_1.reshape((3,1)),np.ones((1,num_points)))
    #                 ).T

    # # check the error
    # registration = registration_iasd(transformedScan1, scan2)
    # correspondencies = registration.find_closest_points()

    # errors = [
    #     correspondence["dist2"] ** 2 for correspondence in correspondencies.values()
    # ]

    # error = np.average(errors)

    # if error < 10E-12:
    #     return (True, r_1, t_1, 0)

    # use our search algorithm
    align_problem = align_3d_search_problem(scan1, scan2, 0, 1e-2)

    ret = search.breadth_first_graph_search(align_problem)
    if ret == None:
        return (False, np.eye(3), np.zeros(3), 0)

    R = eulerAnglesToRotationMatrix(list(ret.state))

    # Transform the scan1 with the r and t matrix found in the search
    num_points, _ = scan1.shape
    transformedScan1 = (np.dot(R, scan1.T)).T

    # Use the 1st submission algorithm
    reg = registration_iasd(transformedScan1, scan2)
    r_2, t_2 = reg.get_compute()

    R = np.dot(r_2, R)

    T = t_2

    return (True, R, T, ret.depth)

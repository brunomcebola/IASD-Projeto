from typing import Tuple
import numpy as np
from numpy import array
from numpy.core.fromnumeric import shape
import search
from math import sqrt, sin, cos, inf
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


def get_plot(scan1, scan2):
    from visualization_vtk import point_clouds_visualization

    figure = point_clouds_visualization()

    avg_x, avg_y, avg_z = tuple(np.average(scan2, axis=0))

    figure.set_camera_settings(
        camera_position=(avg_x, avg_y + 1, avg_z + 0.35),
        camera_focal_point=(-avg_x, -avg_y, -avg_z),
    )

    figure.set_windows_name_size(name="IASD21/22: Point Cloud Registration")

    figure.make_point_cloud(scan1, point_weight=5, point_cloud_color=(0.9, 0.1, 0.1))
    figure.make_point_cloud(scan2, point_weight=5, point_cloud_color=(0.1, 0.1, 0.9))

    figure.render(block=True)


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

        # set max erro
        self.maxError = maxError

        #  set intial state
        self.initial = ((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi))

        

        # convert everything to numpy to ensure compatibility
        #get_plot(self.initial_scan, self.goal_scan)
        self.initial_scan = np.array(scan1)
        self.goal_scan = np.array(scan2)

        return

    def eval_error(self, test_scan) -> float:
        # Same as find_closests_points from previous submission
        # but the return is de average of the distances instead of the
        # correspondencies
        dist_matrix = np.zeros(test_scan.shape[0])
        for (index, element) in enumerate(test_scan):

            # compute the distance of each point of scan2 to to the current element of scan1
            norma = (self.goal_scan - element) ** 2
            norma = norma.sum(axis=1)

            # find where the min distance is
            min_idx = np.argmin(norma)

            norma = sqrt(norma[min_idx])

            # and save the actual distance
            dist_matrix[index] = norma

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
        # and y axis and z axis

        actions = (1, 2, 3, 4, 5, 6)

        return actions

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

        if action == 1:
            x_avg = (state[0][0] + state[0][1]) / 2
            result = ((state[0][0], x_avg), state[1], state[2])

        elif action == 2:
            x_avg = (state[0][0] + state[0][1]) / 2
            result = ((x_avg, state[0][1]), state[1], state[2])

        elif action == 3:
            y_avg = (state[1][0] + state[1][1]) / 2
            result = (state[0], (y_avg, state[1][1]), state[2])

        elif action == 4:
            y_avg = (state[1][0] + state[1][1]) / 2
            result = (state[0], (state[1][0], y_avg), state[2])

        elif action == 5:
            z_avg = (state[2][0] + state[2][1]) / 2
            result = (state[0], state[1], (z_avg, state[2][1]))

        elif action == 6:
            z_avg = (state[2][0] + state[2][1]) / 2
            result = (state[0], state[1], (state[2][0], z_avg))

        return result

    def goal_test(self, state: State) -> bool:
        """Return True if the state is a goal. The default method compares the
        state to self.goalScan or checks for state in self.goalScan if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goalScan is not enough.

        :param state: gets as input the state
        :type state: State
        :return: returns true or false, whether it represents a node state or not
        :rtype: bool
        """

        # here we apply the rotation and translation to the "filtered points"
        rotation_matrix = eulerAnglesToRotationMatrix([np.average(list(state_el)) for state_el in state])

        transformedScan = np.dot(
            rotation_matrix,
            self.initial_scan.T,
        ).T

        if self.eval_error(transformedScan) < self.maxError:
            reg = registration_iasd(transformedScan, self.goal_scan)
            R,T = reg.get_compute()
            transformedScan = np.dot(
                R,
                transformedScan.T,
            ).T

            return self.eval_error(transformedScan) < self.maxError*0.5


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
        #path cost is basically the depth
        return c+1

    def h(self, node):
        """Returns the heuristic at a specific node.
        note: use node.state to access the state

        :param node: node to include the heuristic
        :return: heuristic value
        :rtype: float
        """
        #Our heuristic is basically the depth, making it similar
        #to a breadth-first-search
        return node.depth

        #Note when we use this heuristic somehow the program is slower
        #so we basically got confused in why
        if node.action == None:
            return 1    
        state = self.result(node.state,node.action)
        rotation_matrix = eulerAnglesToRotationMatrix([np.average(list(state_el)) for state_el in state])
        transformedScan = np.dot(
            rotation_matrix,
            self.initial_scan.T,
        ).T

        return self.eval_error(transformedScan)

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

    # calculate the centroid of the scans
    scan1_center = np.average(scan1, axis=0)

    scan2_center = np.average(scan2, axis=0)

    # putting the centroids of the scans on (0,0,0)
    initial_scan = scan1 - scan1_center
    goal_scan = scan2 - scan2_center

    # normalzing the problem
    initial_scan_max = np.max(initial_scan)
    initial_scan_min = np.min(initial_scan)

    goal_scan_max = np.max(initial_scan)
    goal_scan_min = np.min(initial_scan)

    scans_max_value = max(initial_scan_max, goal_scan_max)
    scans_min_value = min(initial_scan_min, goal_scan_min)

    scans_max_abs_value = max(abs(scans_max_value), abs(scans_min_value))

    initial_scan = initial_scan / abs(scans_max_abs_value)
    goal_scan = goal_scan / abs(scans_max_abs_value)

    # get only a % of the points
    initial_scan_center = np.average(initial_scan, axis=0)
    goal_scan_center = np.average(goal_scan, axis=0)

    initial_scan_distances_to_center = ((initial_scan_center - initial_scan) ** 2).sum(axis=1)
    goal_scan_distances_to_center = ((goal_scan_center - goal_scan) ** 2).sum(axis=1)

    initial_scan_distances_to_center_sorted = np.sort(initial_scan_distances_to_center)[::-1]
    goal_scan_distances_to_center_sorted = np.sort(goal_scan_distances_to_center)[::-1]

    # set the amount of points to be fetched
    if goal_scan.shape[0] > 800:
        k = int(goal_scan.shape[0] // 4) 
    else:
        k = int(goal_scan.shape[0] // 5) 

    initial_scan_aux = []
    goal_scan_aux = []

    for id in range(k):
        correspondences_aux = abs(initial_scan_distances_to_center_sorted - goal_scan_distances_to_center_sorted[id])

        initial_scan_corresponding_id_sorted = np.where(correspondences_aux == np.min(correspondences_aux))
        initial_scan_corresponding_id = np.where(
            initial_scan_distances_to_center
            == initial_scan_distances_to_center_sorted[initial_scan_corresponding_id_sorted[0][0]]
        )
        #initial scan
        initial_scan_aux.append(initial_scan[initial_scan_corresponding_id[0][0]])
        initial_scan_distances_to_center_sorted[initial_scan_corresponding_id_sorted] = inf

        # goal scan
        goal_scan_real_id = np.where(goal_scan_distances_to_center == goal_scan_distances_to_center_sorted[id])
        goal_scan_aux.append(goal_scan[goal_scan_real_id[0][0]])

    initial_scan = initial_scan_aux + scan1_center
    goal_scan = goal_scan_aux + scan2_center

    # use our search algorithm
    align_problem = align_3d_search_problem(initial_scan, goal_scan, 20.5e-2)

    ret = search.astar_search(align_problem)

    # if the search doesn't work
    if ret == None:
        return (False, np.eye(3), np.zeros(3), 0)

    
    R_3 = eulerAnglesToRotationMatrix([np.average(list(state_el)) for state_el in ret.state])

    # Transform the scan1 with the r and t matrix found in the search
    transformedScan1 = (np.dot(R_3, initial_scan.T)).T


    # Use the 1st submission algorithm
    reg = registration_iasd(transformedScan1, goal_scan)
    r_2, t_2 = reg.get_compute()

    R = r_2 @ R_3

    T = t_2

    return (
        True,
        R,
        T,
        ret.depth,
    )

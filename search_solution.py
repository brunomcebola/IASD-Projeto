from typing import Tuple
import numpy as np
from numpy import array
from numpy.core.fromnumeric import shape
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


def get_plot(scan1, scan2):
    from visualization_vtk import point_clouds_visualization

    figure = point_clouds_visualization()

    avg_x, avg_y, avg_z = tuple(np.average(scan2, axis=0))

    figure.set_camera_settings(
        camera_position=(avg_x, avg_y + 1, avg_z + 0.35),
        camera_focal_point=(-avg_x, -avg_y, -avg_z),
    )

    figure.set_windows_name_size(name="IASD21/22: Point Cloud Registration")

    figure.make_point_cloud(scan1, point_weight=5, point_cloud_color=(0.1, 0.9, 0.1))
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

        # Creates an initial state.
        # You may want to change this to something representing
        # your initial state.

        # First we try to filter some points so it's not needed
        # to do all the math for huge matrixes.

        # get_plot(scan1, scan2)

        cloud1_center = np.average(scan1, axis=0)
        cloud2_center = np.average(scan2, axis=0)

        distCenter = ((cloud2_center - cloud1_center) ** 2).sum()

        # normalize scans

        max_value = abs(scan1.max() if scan1.max() > scan2.max() else scan2.max())
        min_value = abs(scan1.min() if scan1.min() < scan2.min() else scan2.min())

        max_abs_value = max_value if max_value > min_value else min_value

        # scan1_normalized = scan1 / abs(max_abs_value)
        # scan2_normalized = scan2 / abs(max_abs_value)

        scan1_normalized = scan1 / (max_abs_value)
        scan2_normalized = scan2 / (max_abs_value)

        # get_plot(scan1_normalized, scan2_normalized)

        cloud1_center_normalized = np.average(scan1_normalized, axis=0)
        cloud2_center_normalized = np.average(scan2_normalized, axis=0)

        distCenter_normalized = (
            (cloud2_center_normalized - cloud1_center_normalized) ** 2
        ).sum()

        max_value_normalized = abs(
            scan1_normalized.max()
            if scan1_normalized.max() > scan2_normalized.max()
            else scan2_normalized.max()
        )
        min_value_normalized = abs(
            scan1_normalized.min()
            if scan1_normalized.min() < scan2_normalized.min()
            else scan2_normalized.min()
        )

        max_abs_value_normalized = (
            max_value_normalized
            if max_value_normalized > min_value_normalized
            else min_value_normalized
        )

        print(distCenter_normalized < distCenter)

        print("inicial")
        print(distCenter)
        print(max_abs_value)
        print("normalizado")
        print(distCenter_normalized)
        print(max_abs_value_normalized)

        if distCenter_normalized < distCenter / 2:
            scan1 = scan1_normalized
            scan2 = scan2_normalized

        # #find the center of each cloud
        cloud1_center = np.average(scan1, axis=0)
        cloud2_center = np.average(scan2, axis=0)

        # TODO: check how to set error threshold
        n1 = sqrt((cloud1_center ** 2).sum())
        n2 = sqrt((cloud2_center ** 2).sum())

        self.maxError = (n1 + n2) / 2
        if 300 < scan1.shape[0] < 1000 or scan1.shape[0] < 100 or scan1.shape[0] > 2000:
            self.maxError = self.maxError * 0.1
        elif 100 < scan1.shape[0] < 300 or scan1.shape[0] > 1500:
            self.maxError = self.maxError * 0.075
        else:
            self.maxError = self.maxError * 0.02

        self.maxError = maxError

        # cloud2dists = ((scan2 - cloud2_center) ** 2).sum(axis=1)
        # self.maxError = cloud2dists.max() / 200

        # print(self.maxError)

        # get the k points closest to the center of each scan
        norm1 = ((scan1 - cloud1_center) ** 2).sum(axis=1)
        norm2 = ((scan2 - cloud2_center) ** 2).sum(axis=1)

        self.scan1 = []
        self.goal = []

        # number of points to fetch from each scan
        k = scan1.shape[0] // 10

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

        self.initial = ((-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi))
        self.center = np.average(self.goal, axis=0)

        return

    def eval_error(self, testScan) -> float:
        # Same as find_closests_points from previous submission
        # but the return is de average of the distances instead of the
        # correspondencies

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
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough.

        :param state: gets as input the state
        :type state: State
        :return: returns true or false, whether it represents a node state or not
        :rtype: bool
        """

        # here we apply the rotation and translation to the "filtered points"
        rotation_matrix = eulerAnglesToRotationMatrix(
            [np.average(list(state_el)) for state_el in state]
        )

        transformedScan = np.dot(
            rotation_matrix,
            self.scan1.T,
        ).T

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
    align_problem = align_3d_search_problem(scan1, scan2, 1.2e-2)

    ret = search.breadth_first_graph_search(align_problem)

    if ret == None:
        return (False, np.eye(3), np.zeros(3), 0)

    R_3 = eulerAnglesToRotationMatrix(
        [np.average(list(state_el)) for state_el in ret.state]
    )

    # Transform the scan1 with the r and t matrix found in the search
    transformedScan1 = (np.dot(R_3, scan1.T)).T

    trasformedTranslation = transformedScan1 + (
        np.average(scan2, axis=0) - np.average(transformedScan1, axis=0)
    )

    # get_plot(
    #     trasformedTranslation,
    #     scan2,
    # )

    reg = registration_iasd(trasformedTranslation, scan2)
    correspondencies = reg.find_closest_points()

    if all(
        correspondencie["dist2"] < 1e-13
        for correspondencie in correspondencies.values()
    ):
        R = R_3
        T = np.average(scan2, axis=0) - np.average(transformedScan1, axis=0)
    else:
        # Use the 1st submission algorithm
        reg = registration_iasd(transformedScan1, scan2)
        r_2, t_2 = reg.get_compute()

        R = r_2 @ R_3

        T = t_2

    return (
        True,
        R,
        T,
        ret.depth,
    )

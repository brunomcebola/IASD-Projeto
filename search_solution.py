from typing import Tuple
import numpy as np
from numpy import array
import search
import math
import time

# you can want to use the class registration_iasd
# from your solution.py (from previous assignment)
from solution import registration_iasd

#
#
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

#
#

# Choose what you think it is the best data structure
# for representing actions.
Action = float

# Choose what you think it is the best data structure
# for representing states.
class myState:
    def __init__(
        self,
        transformedScan: array((..., 3)),
    ) -> None:
        self.transformedScan = transformedScan


State = myState


class align_3d_search_problem(search.Problem):
    def __init__(self, scan1: array((..., 3)), scan2: array((..., 3))) -> None:
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

        if np.array(scan2).shape[0] > np.array(scan1).shape[0]:
            self.goal = scan2
            self.original = scan1
        else:
            self.goal = scan1
            self.original = scan2

        self.initial = myState(self.original)
        self.center = np.average(self.original, axis=0)

        self.i = 1

        return

    def actions(self, state: State) -> Tuple:
        """Return the actions that can be executed in the given state.
        The result would be a list, since there are only four possible actions
        in any given state of the environment

        :param state: Abstract representation of your state
        :type state: State
        :return: Tuple with all possible actions
        :rtype: Tuple
        """

        action = []

        if self.i % 3:
            action.append(("z", -np.radians(90)))
        elif self.i % 2:
            action.append(("y", -np.radians(90)))
        else:
            action.append(("x", -np.radians(90)))

        return tuple(action)

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

        if action[0] == "x":
            R = eulerAnglesToRotationMatrix([action[1], 0, 0])
        elif action[0] == "y":
            R = eulerAnglesToRotationMatrix([0, action[1], 0])
        elif action[0] == "z":
            R = eulerAnglesToRotationMatrix([0, 0, action[1]])

        T = np.vstack(
            (
                np.column_stack(
                    (
                        R,
                        [
                            self.center[0]
                            - R[0][0] * self.center[0]
                            - R[0][1] * self.center[1]
                            - R[0][2] * self.center[2],
                            self.center[1]
                            - R[1][0] * self.center[0]
                            - R[1][1] * self.center[1]
                            - R[1][2] * self.center[2],
                            self.center[2]
                            - R[2][0] * self.center[0]
                            - R[2][1] * self.center[1]
                            - R[2][2] * self.center[2],
                        ],
                    )
                ),
                [0, 0, 0, 1],
            )
        )

        # here we apply the rotation and translation to the actual points
        r = T[0:3, 0:3]
        t = T[0:3, 3]

        num_points = np.array(state.transformedScan).shape[0]
        state.transformedScan = np.array(
            (
                np.dot(r, np.array(state.transformedScan).T)
                + np.dot(np.array(t).reshape((3, 1)), np.ones((1, num_points)))
            )
        ).T

        return state

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

        registration = registration_iasd(state.transformedScan, self.goal)
        correspondencies = registration.find_closest_points()

        error = sum(
            correspondence["dist2"] ** 2 for correspondence in correspondencies.values()
        )

        self.i = self.i + 1

        #
        #
        plt.ion()
        plt.show()

        ax = plt.axes(projection="3d")

        # Data for three-dimensional scattered points
        xdata = [row[0] for row in state.transformedScan]
        ydata = [row[1] for row in state.transformedScan]
        zdata = [row[2] for row in state.transformedScan]
        ax.scatter3D(
            xdata,
            ydata,
            zdata,
            c=zdata,
            cmap="Greens",
        )

        xdata = [row[0] for row in self.goal]
        ydata = [row[1] for row in self.goal]
        zdata = [row[2] for row in self.goal]
        ax.scatter3D(
            xdata,
            ydata,
            zdata,
            c=zdata,
            cmap="Blues",
        )

        plt.draw()
        plt.pause(0.000001)
        #
        #

        return error < 0.0001  # 1E-16

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

    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )

    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )

    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


# Converts the euler angles to a rotation matrix to be applied to the clouds
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

    R = np.array(
        [
            [np.cos(phi), np.sin(theta) * np.sin(psi), np.sin(theta) * np.cos(psi)],
            [
                np.sin(theta) * np.sin(phi),
                np.cos(psi) * np.cos(phi) - np.cos(theta) * np.sin(psi) * np.sin(phi),
                -np.cos(phi) * np.sin(psi) - np.cos(theta) * np.cos(psi) * np.sin(phi),
            ],
            [
                -np.sin(theta) * np.cos(phi),
                np.cos(psi) * np.sin(phi) + np.cos(theta) * np.cos(phi) * np.sin(psi),
                -np.sin(psi) * np.sin(phi) + np.cos(theta) * np.cos(psi) * np.cos(phi),
            ],
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

    # #find the center of each cloud
    cloud1_center = np.average(scan1, axis=0)
    cloud2_center = np.average(scan2, axis=0)

    print(f"cloud1 center = {cloud1_center}\ncloud2 center = {cloud2_center}")

    dist = cloud2_center - cloud1_center

    # Move center of scan1 to center of scan2
    scan1 = scan1 + dist

    align_problem = align_3d_search_problem(scan1, scan2)

    node = search.breadth_first_tree_search(align_problem)

    R = np.eye(3)
    T = 0
    return (True, R, T, node.depth)


if __name__ == "__main__":
    R = angleToMatrix(0, 0, np.pi)
    R2 = eulerAnglesToRotationMatrix([0, 0, np.pi])

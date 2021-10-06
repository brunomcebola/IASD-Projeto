from registration import registration
from get_pointcloud import point_cloud_data
from math import sqrt, inf

import numpy as np
from typing import Tuple


class registration_iasd(registration):
    def __init__(self, scan_1: np.array((..., 3)), scan_2: np.array((..., 3))) -> None:

        # inherit all the methods and properties from registration
        super().__init__(scan_1, scan_2)

        return

    def compute_pose(self, correspondences: dict) -> Tuple[np.array, np.array]:
        """compute the transformation that aligns two
        scans given a set of correspondences

        :param correspondences: set of correspondences
        :type correspondences: dict
        :return: rotation and translation that align the correspondences
        :rtype: Tuple[np.array, np.array]
        """

        s1_center = np.zeros(3)
        s2_center = np.zeros(3)

        s1_center = np.mean(self.scan_1, axis=0)
        s2_center = np.mean(self.scan_2, axis=0)

        s1_stack = []
        s2_stack = []

        for key in correspondences:
            s1_stack.append(correspondences[key]["point_in_pc_1"] - s1_center)
            s2_stack.append(correspondences[key]["point_in_pc_2"] - s2_center)

        A = np.matmul(np.transpose(s2_stack), s1_stack)

        U, S, V = np.linalg.svd(A)

        # check below for possible optimizations
        # (we cant change the algorithm)

        det = np.linalg.det(np.matmul(U, V))

        diag = np.array([[1, 0, 0], [0, 1, 0], [0, 0, det]])

        R_out = np.matmul(
            np.matmul(U, diag),
            V,
        )

        T_out = s2_center - np.matmul(R_out, s1_center)

        return (R_out, T_out)

    def find_closest_points(self) -> dict:
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
        correspondencies = {}

        for s1_p_index in range(self.scan_1.shape[0]):
            correspondencies[s1_p_index] = {}
            correspondencies[s1_p_index]["point_in_pc_1"] = self.scan_1[s1_p_index]
            
            aux_matrix = (self.scan_2 - self.scan_1[s1_p_index])**2
            square_dists = aux_matrix.sum(axis=1)
            min_index = np.argmin(square_dists)
            correspondencies[s1_p_index]["point_in_pc_2"] = self.scan_2[min_index]
            
            # not using sqrt in order to meet the stopping condition
            correspondencies[s1_p_index]["dist2"] = square_dists[min_index]

        return correspondencies


class point_cloud_data_iasd(point_cloud_data):
    def __init__(
        self,
        fileName: str,
    ) -> None:

        super().__init__(fileName)

        return

    def load_point_cloud(self, file: str) -> bool:
        """Loads a point cloud from a ply file

        :param file: source file
        :type file: str
        :return: returns true or false, depending on whether the method
        the ply was OK or not
        :rtype: bool
        """

        with open(file, "r") as fp:
            point_order = [None, None, None]
            counter = 0
            while True:
                line = fp.readline().split()

                if line[0] == "element":
                    if line[1] == "vertex":
                        n_vertex = int(line[2])
                elif line[0] == "property":
                    if line[2] == "x":
                        point_order[0] = counter
                    elif line[2] == "y":
                        point_order[1] = counter
                    elif line[2] == "z":
                        point_order[2] = counter
                    counter += 1
                elif line[0] == "end_header":
                    break

            if None in point_order:
                return False

            for i in range(n_vertex):
                line = fp.readline().split()
                #if there's less properties found than the ones mentioned
                if len(line) != counter:
                    return False
                point_list = [
                    float(line[point_order[0]]),
                    float(line[point_order[1]]),
                    float(line[point_order[2]]),
                ]
                self.data[str(i)] = np.array(point_list)
        return True

# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:32:38 2020

@author: abrsh
"""
import numpy as np
import cv2
import bisect
import matplotlib.pyplot as plt
import copy
import unionfind
import sys
from scipy.ndimage import gaussian_filter

sys.setrecursionlimit(5000)
DATA_DIR = 'data/mser/'


def find_neighbors(id, shape):
    """
    Finds the coordinates of neighbors of a given pixel.
    """
    return [id + 1, id - 1, id - shape[1], id + shape[1]]


def coords2id(coords, shape):
    return coords[0] * shape[1] + coords[1]


def id2coords(pixid, shape):
    return [pixid // shape[0], pixid % shape[0]]


class MSER:
    def __init__(self, num_thresh=5, max_area=20000, min_area=40, max_var=0.25,
                 delta=2):
        self.num_thresh = num_thresh
        self.max_area = max_area
        self.min_area = min_area
        self.max_var = max_var
        self.delta = 2

        if self.num_thresh <= 2 * self.delta:
            raise ValueError(f'The number of thresholds must at least \
                             be 2 * delta + 1')

    @staticmethod
    def order_rotationally(points):
        """
        Orders points starting from a point and
        @param points: A 2D numpy.ndarray with each row representing a point
        @return:
        """
        centroid = points.mean(axis=0, keepdims=True)
        coords = points - centroid
        angles = np.arctan2(coords[:,1], coords[:,0])
        sort_idxs = np.argsort(angles)
        return points[sort_idxs, :]

    def detect(self, image):
        """
        Detect maximally stable extremal regions
        @type image: np.ndarray
        @param image: The image from which MSER are to be extracted from
        @rtype: list
        @return: A list of contours representing detected MSERs
        """
        if image.ndim > 2:
            raise TypeError(f'A grayscale image expected but image had \
                            {image.ndim} channels')

        image_shape = image.shape
        image = image.flatten()
        sorting_indices = np.argsort(image)
        row_indices, column_indices = np.unravel_index(sorting_indices,
                                                       image_shape)
        sorted_coordinates = np.array(list(zip(row_indices, column_indices)))

        threshold_list = np.linspace(0, 255, self.num_thresh, endpoint=True)

        output_images = np.zeros((self.num_thresh, image_shape[0], image_shape[1]), dtype=np.uint8)

        sorted_intensity = image[sorting_indices]

        # connected component history
        UF = unionfind.UnionFind(image.size)
        for i, threshold in enumerate(threshold_list):
            # Find the index at which intensity > thershold
            end_index = bisect.bisect(sorted_intensity, threshold)
            if end_index >= image.size:
                continue

            output_images[i, row_indices[end_index:], column_indices[end_index:]] = 255
            ids = sorting_indices[end_index:]
            neighbors = np.apply_along_axis(find_neighbors, 0, ids, image_shape).T

            for j, pix_neighbors in enumerate(neighbors):
                for neighbor in pix_neighbors:
                    if neighbor < 0 or neighbor > image.size:
                        continue
                    # If a neighbor is a white pixel it is part of connected component else it is border pixel
                    pix = ids[j]
                    if neighbor in ids:
                        UF.union(neighbor, pix)
                    else:
                        UF.add_neighbor(pix, neighbor)

        all_history = UF.get_top_level_history()
        history = filter(self.is_possibly_MSER, all_history)
        msers = []
        for parent_comp in history:
            ph = [parent_comp.size, parent_comp.size]
            pq = [float('inf'), float('inf')]
            MSER.find_msers(parent_comp, msers, ph, pq, self.max_area, self.min_area, self.max_var)

        contours = []
        for mser in msers:
            neighbors = mser.neighbors.difference(mser.members)
            border_pixels = np.array(list(neighbors))
            contour = np.apply_along_axis(id2coords, 0, border_pixels, image_shape).T
            # Make contours OpenCV compatible (switch x and y)
            contour = np.flip(contour, axis=1)
            contour = MSER.order_rotationally(contour)
            contours.append(contour)
        return contours

    @staticmethod
    def find_msers(parent_comp, msers, parent_sizes, parent_q, max_area=2000, min_area=2, max_var=0.25):
        if parent_comp is None:
            return [0] * len(parent_sizes)

        q = (parent_sizes[-1] - parent_comp.children_sizes[0]) / parent_comp.size
        ph = copy.copy(parent_sizes)
        pq = copy.copy(parent_q)
        ph = [parent_comp.size] + ph[:-1]
        pq = [q] + pq[:-1]
        # The Union-Find algorithm always puts the larger componenet on the left
        child_q = MSER.find_msers(parent_comp.left, msers, ph, pq, max_var)
        MSER.find_msers(parent_comp.right, msers, ph, pq, max_var)

        all_q = child_q + [q] + parent_q
        var = (parent_comp.size - parent_comp.children_sizes[-1])/(parent_comp.size)
        if min(all_q) == q and var < max_var and (min_area < parent_comp.size < max_area):
            msers.append(parent_comp)
        return child_q[1:] + [q]

    def is_possibly_MSER(self, component):
        return self.min_area < component.size < self.max_area


def generate_test_image(size, type):
    image = np.zeros((size, size))
    # One hollow square
    r, c, w, h = size // 10, size // 7, size // 4, size // 4
    if type == 'hollow':
        image[r, c:c+w] = 255
        image[r+h, c:c+w] = 255
        image[r:r+h, c] = 255
        image[r:r+h+1, c+w] = 255
    elif type == 'full':
        image[r:r+h, c:c+w] = 255
        r2, c2 = r + size// 2, c + size // 2
        image[r2:r2+h, c2:c2+w] = 255
        image = gaussian_filter(image, sigma=2)
    return image


if __name__ == '__main__':
    im = generate_test_image(100, 'full')
    mser_detector = MSER()
    contours = mser_detector.detect(im)

    print("{} MSERS found".format(len(contours)))
    for i, contour in enumerate(contours):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(im, [box], 0, (255, 0, 0))
        #cv2.drawContours(im, contours, i, (255, 0, 0), 0)
    plt.imshow(im)
    plt.show()
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 20:32:38 2020

@author: abrih
"""
import numpy as np
import cv2
import bisect
import os
import matplotlib.pyplot as plt
import copy
import unionfind
import logging
import sys
sys.setrecursionlimit(5000)
DATA_DIR = 'data/mser/'


def is_N4_adjacent(coord1, coord2):
    """

    """
    #print(coord1, coord2)
    return sum([abs(coord1[i] - coord2[i]) for i in range(len(coord1))]) == 1


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
    def order_rotationally(points, order='CCW'):
        """

        @param points: A 2D numpy.ndarray with each row representing a point
        @param order: Sort the points in clockwise or counter-clockwise order
        @return:
        """
        centroid = points.mean(axis=0, keepdims=True)
        coords = points - centroid
        angles = np.arctan2(coords[:,1], coords[:,0])
        sort_idxs = np.argsort(angles)
        return points[sort_idxs, :]

    def detect(self, image):
        if image.ndim > 2:
            raise TypeError(f'A grayscale image expected but image had \
                            {image.ndim} channels')

        image_shape = image.shape
        sorting_indices = np.argsort(image, axis=None)
        row_indices, column_indices = np.unravel_index(sorting_indices,
                                                       image_shape)
        sorted_coordinates = np.array(list(zip(row_indices, column_indices)))

        threshold_list = [127]#np.linspace(0, 255, self.num_thresh, endpoint=True)

        output_images = np.zeros((self.num_thresh, image_shape[0], image_shape[1]), dtype=np.uint8)

        sorted_intensity = image[row_indices, column_indices]

        # connected component history
        UF = unionfind.UnionFind(image.size)
        for i, threshold in enumerate(threshold_list):
            # Find the index at which intensity > thershold
            end_index = bisect.bisect(sorted_intensity, threshold)
            if end_index >= image.size:
                continue
            # print(start_index, end_index)
            # output_images[i:, row_indices[start_index:end_index], column_indices[start_index:end_index]] = 255
            output_images[i, row_indices[end_index:], column_indices[end_index:]] = 255
            ids = np.apply_along_axis(coords2id, 1, sorted_coordinates[end_index:, :], image_shape)
            neighbors = np.apply_along_axis(find_neighbors, 0, ids, image_shape).T
            # print(len(ids))
            print(neighbors)
            for j, pix_neighbors in enumerate(neighbors):
                #print(pix_neighbors)
                for neighbor in pix_neighbors:
                    if neighbor < 0 or neighbor > image.size:
                        continue

                    pix = ids[j]
                    if neighbor in ids:
                        # print(f"Merging {neighbor} and {j}")
                        UF.union(neighbor, pix)
                    else:
                        UF.add_neighbor(pix, neighbor)

            # im = output_images[i]
            # plt.imshow(im, cmap='gray')
            # plt.figure()
            # im = im.flatten()
            # toplevel = UF.get_top_level_history()
            # # print(len(toplevel), UF.count())
            # for comp_history in toplevel:
            #     if comp_history.size == 1:
            #         continue
            #     members = np.array(list(comp_history.members))
            #     im[members] = np.random.randint(0, 256)
            # im = im.reshape(image_shape)
            # plt.imshow(im, cmap='gray', vmin=0, vmax=255)
            # plt.title(str(i + 1))
            # plt.figure()

        all_history = UF.get_top_level_history()
        # print(all_history)
        history = filter(self.is_possibly_MSER, all_history)
        msers = []
        for parent_comp in history:
            ph = [parent_comp.size, parent_comp.size]
            pq = [float('inf'), float('inf')]
            MSER.find_msers(parent_comp, msers, ph, pq, self.max_area, self.min_area, self.max_var)
        #start_index = end_index
        print(msers)
        contours = []
        for mser in msers:
            neighbors = mser.neighbors.difference(mser.members)
            border_pixels = np.array(list(neighbors))
            contour = np.apply_along_axis(id2coords, 0, border_pixels, image_shape).T
            #contour = np.flip(contour, axis=1)
            print(len(contour))
            contour = MSER.order_rotationally(contour)
            # print(contour.shape)
            contours.append(contour)
        # print(contours)
        return output_images, msers, contours

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
        print(var)
        if min(all_q) == q and var < max_var and (min_area < parent_comp.size < max_area):
            msers.append(parent_comp)
        return child_q[1:] + [q]

    def is_possibly_MSER(self, component):
        return self.min_area < component.size < self.max_area

    def detect2(self, image):
        if image.ndim > 2:
            raise TypeError(f'A grayscale image expected but image had \
                            {image.ndim} channels')

        image_shape = image.shape
        sorting_indices = np.argsort(image, axis=None)
        row_indices, column_indices = np.unravel_index(sorting_indices,
                                                       image_shape)
        sorting_coords = np.array(list(zip(row_indices, column_indices)))
        # print(_coords.shape, pixel_coords.dtype, pixel_coords[:5])

        output_images = np.zeros((self.num_thresh, image_shape[0],
                                  image_shape[1]), dtype=np.uint8)
        threshold_list = np.linspace(0, 255, self.num_thresh, endpoint=True)
        sorted_intensity = image[row_indices, column_indices]

        parent_list = -1 * np.ones_like(sorting_indices)
        start_index = 0

        for i, threshold in enumerate(threshold_list):
            end_index = bisect.bisect(sorted_intensity, threshold)

            pixel_positions = sorting_indices[start_index:end_index]
            unmerged_pixels = copy.deepcopy(pixel_positions)
            pixel_coords = sorting_coords[start_index:end_index]

            for position, pixel in zip(pixel_positions, pixel_coords):
                neighbors = np.apply_along_axis(is_N4_adjacent, 1,
                                                sorting_coords, pixel)

                added_to_existing = np.logical_and(neighbors,
                                                   parent_list != -1)

                # print(f"Added to existing {np.sum(added_to_existing)}")
                if np.sum(added_to_existing):
                    neighbor_parents = parent_list[added_to_existing]
                    #print(neighbor_parents)
                    parent_list[added_to_existing] = position
                    for neighbor in neighbor_parents:
                        parent_list[parent_list == neighbor] = position
                        print(np.sum(parent_list == neighbor))
                else:
                    parent_list[position] = position
                # print(neighbors) if np.sum(neighbors) > 0 else np.__version__
                #print(np.unique(parent_list))
                #print(f"We now have {len(np.unique(parent_list)) - 1} \
                #componenets")
            
            components = np.unique(parent_list)
            ncomponents = len(components)
            color = {c:int(255.*i/ncomponents) for i, c in enumerate(components)}
            for val, c in color.items():
                parent_indices = np.nonzero(parent_list == val)
                image_indices = sorting_indices[parent_indices]
                
                r_idx, col_idx = np.unravel_index(image_indices,
                                                  output_images[i].shape)
                #print(r_idx, col_idx)
                output_images[i, r_idx, col_idx]  = c
                #print(np.unique(output_images[i]))
        return output_images


if __name__ == '__main__':
    # from scipy.ndimage import gaussian_filter
    image = np.zeros((50, 50))
    # length = 5
    # weight = np.linspace(0.1, 1, length).tolist() + np.linspace(1, 0.1, length).tolist()
    # weight = np.repeat(np.array(weight), 2 * length).reshape((2*length, 2*length))
    # print(weight.shape)
    # One hollow square
    # r, c, w, h = 10, 15, 5, 5
    # image[r, c:c+w] = 255
    # image[r+h, c:c+w] = 255
    # image[r:r+h, c] = 255
    # image[r:r+h+1, c+w] = 255
    # cv2.imwrite(DATA_DIR + 'rect.png', image)
    # Type two filled squares
    image[10:20, 10:20] = 255
    image[30:40, 30:40] = 255
    # image = gaussian_filter(image, sigma=2)
    #
    # cv2.imwrite(DATA_DIR + 'test.png', image)
    im = cv2.imread(os.path.join(DATA_DIR, 'test.png'))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    mser_detector = MSER()
    output, history, contours = mser_detector.detect(im)
    # print(len(contours))
    # print(history[0].members.intersection(history[0].neighbors))
    for i in range(len(contours)):
        cv2.drawContours(im, contours, i, (255, 0, 0), 0)
    # print(sorted(history, key=lambda g:g.size, reverse=True))
    # cv2.imshow("win", im)
    # cv2.waitKey()
    plt.imshow(im)
    plt.figure()
    # for i, im in enumerate(output):
    #     plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    #     plt.title(str(i + 1))
    #     plt.figure()

    plt.show()
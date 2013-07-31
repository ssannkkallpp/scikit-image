#cython: cdivision=True
#cython: boundscheck=True
#cython: nonecheck=True
#cython: wraparound=True

cimport numpy as cnp
import numpy as np


def _censure_dob_loop(double[:, ::1] image, Py_ssize_t n,
	                  double[:, ::1] integral_img,
	                  double[:, ::1] filtered_image,
	                  double inner_weight, double outer_weight):

    cdef Py_ssize_t i, j
    cdef double inner, outer

    for i in range(2 * n, image.shape[0] - 2 * n):
        for j in range(2 * n, image.shape[1] - 2 * n):
            inner = integral_img[i + n, j + n] + integral_img[i - n - 1, j - n - 1] - integral_img[i + n, j - n - 1] - integral_img[i - n - 1, j + n]
            outer = integral_img[i + 2 * n, j + 2 * n] + integral_img[i - 2 * n - 1, j - 2 * n - 1] - integral_img[i + 2 * n, j - 2 * n - 1] - integral_img[i - 2 * n - 1, j + 2 * n]
            filtered_image[i, j] = outer_weight * outer - (inner_weight + outer_weight) * inner


def _slanted_integral_image(double[:, :] image,
                            double[:, :] integral_img):

    cdef Py_ssize_t i, j
    cdef double[:] left_sum = np.zeros(image.shape[0], dtype=np.float)

    flipped_lr = np.asarray(image[:, ::-1])
    for i in range(image.shape[1] - image.shape[0], image.shape[1]):
        left_sum[image.shape[1] - 1 - i] = np.sum(flipped_lr.diagonal(i))
    left_sum_np = np.asarray(left_sum)

    # Initializing the leftmost column of the slanted integral image
    left_sum_np = left_sum_np.cumsum(0)

    # Initializing the rightmost column of the slanted integral image
    right_sum_np = np.sum(image, 1).cumsum(0)

    for i in range(image.shape[0]):
        image[i, 0] = left_sum_np[i]
        image[i, -1] = right_sum_np[i]

    for i in range(1, integral_img.shape[0]):
        for j in range(integral_img.shape[1]):
            integral_img[i, j] = image[i - 1, j]

    for i in range(1, integral_img.shape[0]):
        for j in range(1, integral_img.shape[1] - 1):
            integral_img[i, j] += integral_img[i, j - 1] + integral_img[i - 1, j + 1] - integral_img[i - 1, j]


def _censure_octagon_loop(double[:, :] image, double[:, :] integral_img,
                          double[:, :] integral_img1,
                          double[:, :] integral_img2,
                          double[:, :] integral_img3,
                          double[:, :] integral_img4,
                          double[:, :] filtered_image,
                          double outer_weight, double inner_weight,
                          int mo, int no, int mi, int ni):
                    
    cdef Py_ssize_t i, j, o_m, i_m, o_set, i_set

    """
    For a (5, 2) octagon, i.e. mo = 5 and no = 2,

                 |---o_set---|
    [0, 0, 1, 1, 1, 1, 1, 0, 0]
    [0, 1, 1, 1, 1, 1, 1, 1, 0]
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
    [0, 1, 1, 1, 1, 1, 1, 1, 0]
    [0, 0, 1, 1, 1, 1, 1, 0, 0]
                 |-o_m-|
    """
    o_m = (mo - 1) / 2
    i_m = (mi - 1) / 2

    # o_set and i_set are the distances of the center of the octagon
    # from the horizontal or vertical sides of the octagon,
    # for outer and inner octagon respectively
    o_set = o_m + no
    i_set = i_m + ni

    for i in range(o_set + 1, image.shape[0] - o_set - 1):
        for j in range(o_set + 1, image.shape[1] - o_set - 1):
            # Calculating the sum of pixels in the outer octagon
            outer = integral_img1[i + o_set, j + o_m] - integral_img1[i + o_m - 1, j + o_set + 1] - integral_img[i + o_set, j - o_m] + integral_img[i + o_m - 1, j - o_m]
            outer += integral_img[i + o_m - 1, j + o_m - 1] - integral_img[i - o_m, j + o_m - 1] - integral_img[i + o_m - 1, j - o_m] + integral_img[i - o_m, j - o_m]
            outer += integral_img4[i + o_m, j - o_set] - integral_img4[i + o_set + 1, j - o_m + 1] - integral_img[i - o_m, j - o_m] + integral_img[i - o_m, j - o_set - 1]
            outer += integral_img2[i - o_set, j - o_m] - integral_img2[i - o_m + 1, j - o_set - 1] - integral_img[i - o_m, -1] - integral_img[i - o_set - 1, j + o_m - 1] + integral_img[i - o_m, j + o_m - 1] + integral_img[i - o_set - 1, -1]
            outer += integral_img3[i - o_m, j + o_set] - integral_img3[i - o_set - 1, j + o_m - 1] - integral_img[-1, j + o_set] - integral_img[i + o_m - 1, j + o_m - 1] + integral_img[-1, j + o_m - 1] + integral_img[i + o_m - 1, j + o_set]

            # Calculating the sum of pixels in the inner octagon
            inner = integral_img1[i + i_set, j + i_m] - integral_img1[i + i_m - 1, j + i_set + 1] - integral_img[i + i_set, j - i_m] + integral_img[i + i_m - 1, j - i_m]
            inner += integral_img[i + i_m - 1, j + i_m - 1] - integral_img[i - i_m, j + i_m - 1] - integral_img[i + i_m - 1, j - i_m] + integral_img[i - i_m, j - i_m]
            inner += integral_img4[i + i_m, j - i_set] - integral_img4[i + i_set + 1, j - i_m + 1] - integral_img[i - i_m, j - i_m] + integral_img[i - i_m, j - i_set - 1]
            inner += integral_img2[i - i_set, j - i_m] - integral_img2[i - i_m + 1, j - i_set - 1] - integral_img[i - i_m, -1] - integral_img[i - i_set - 1, j + i_m - 1] + integral_img[i - i_m, j + i_m - 1] + integral_img[i - i_set - 1, -1]
            inner += integral_img3[i - i_m, j + i_set] - integral_img3[i - i_set - 1, j + i_m - 1] - integral_img[-1, j + i_set] - integral_img[i + i_m - 1, j + i_m - 1] + integral_img[-1, j + i_m - 1] + integral_img[i + i_m - 1, j + i_set]

            filtered_image[i, j] = outer_weight * outer - (outer_weight + inner_weight) * inner

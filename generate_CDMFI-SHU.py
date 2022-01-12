import random

import numpy as np
import os
from numpy.random import randint as RADint
from numpy.random import uniform as RAD
import cv2 as cv


def guidedFilter_oneChannel(srcImg, guidedImg, rad=4, eps=0.01):
    srcImg = srcImg.astype(np.uint8) / 255.0
    guidedImg = guidedImg.astype(np.uint8) / 255.0
    # img_shape = np.shape(srcImg)
    P_mean = cv.boxFilter(srcImg, -1, (rad, rad), normalize=True)
    I_mean = cv.boxFilter(guidedImg, -1, (rad, rad), normalize=True)
    I_square_mean = cv.boxFilter(np.multiply(guidedImg, guidedImg), -1, (rad, rad), normalize=True)
    I_mul_P_mean = cv.boxFilter(np.multiply(srcImg, guidedImg), -1, (rad, rad), normalize=True)
    var_I = I_square_mean - np.multiply(I_mean, I_mean)
    cov_I_P = I_mul_P_mean - np.multiply(I_mean, P_mean)
    a = cov_I_P / (var_I + eps)
    b = P_mean - np.multiply(a, I_mean)
    a_mean = cv.boxFilter(a, -1, (rad, rad), normalize=True)
    b_mean = cv.boxFilter(b, -1, (rad, rad), normalize=True)
    dstImg = np.multiply(a_mean, guidedImg) + b_mean
    return (dstImg * 255).astype(np.uint8)


# def guidedFilter_threeChannel(srcImg, guidedImg, rad=3, eps=0.01):
#     img_shape = np.shape(srcImg)
#     dstImg = np.zeros(img_shape, dtype=float)
#     for ind in range(0, img_shape[2]):
#         dstImg[:, :, ind] = guidedFilter_oneChannel(srcImg[:, :, ind],
#                                                     guidedImg[:, :, ind], rad, eps)
#     dstImg = dstImg.astype(np.uint8)
#     return dstImg


def ero_dil_dil_ero(img, k_size):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (k_size, k_size))
    img_eroded = cv.erode(img, kernel)  # 腐蚀图像
    img_dilated = cv.dilate(img_eroded, kernel)
    img_dilated = cv.dilate(img_dilated, kernel)
    img_eroded = cv.erode(img_dilated, kernel)
    return img_eroded


def get_hist_matrix(img):
    img = img.ravel()
    hist_matrix = np.zeros([1, 256])
    for i in range(len(img)):
        hist_matrix[0, img[i]] += 1
    return hist_matrix


def get_threshold_value(img, num_seg):
    W, C = img.shape
    hist_matrix = get_hist_matrix(img)
    pix_per = int(W * C / num_seg)
    seg_thres = np.zeros([num_seg - 1, 1, 2])
    for i in range(hist_matrix.shape[1]):
        for j in range(3):
            if seg_thres[j, 0, 0] <= pix_per * (j + 1):
                seg_thres[j, 0, 0] += hist_matrix[0, i]
                seg_thres[j, 0, 1] = i
    return seg_thres


def get_segment_interval(img, seg_thres):
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    pix_min = np.min(img)
    pix_max = np.max(img)
    pix_per = int((pix_max - pix_min) * 0.2)
    lower = pix_min + pix_per if pix_min + pix_per < seg_thres[0, 0, 1] else pix_min
    upper = pix_max - pix_per if pix_max - pix_per > seg_thres[2, 0, 1] else pix_max
    segment_interval = [RADint(int(lower), seg_thres[0, 0, 1])]
    for i in range(seg_thres.shape[0] - 1):
        segment_interval.append(RADint(seg_thres[0, 0, 1], seg_thres[i + 1, 0, 1]))
    segment_interval.append(RADint(seg_thres[2, 0, 1], int(upper)))
    return segment_interval


def get_binary_img(img, threshold, guide_img, k_size):
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    _, img_bin = cv.threshold(img, threshold, 255, cv.THRESH_BINARY_INV)
    img_ero_dil = ero_dil_dil_ero(img_bin, 3)
    img_guided = guidedFilter_oneChannel(img_ero_dil, guide_img, k_size)
    return img_guided


def get_focus_img(img, img_bin, c_size):
    W, H = img_bin.shape
    img_gau = cv.GaussianBlur(img, (9, 9), c_size)
    img_A = np.ones_like(img)
    img_B = np.ones_like(img)
    for w in range(W):
        for h in range(H):
            img_A[w, h, :] =\
                (255 - img_bin[w, h]) / 255 * img[w, h, :] + img_bin[w, h] / 255 * img_gau[w, h, :]
            img_B[w, h, :] = \
                (255 - img_bin[w, h]) / 255 * img_gau[w, h, :] + img_bin[w, h] / 255 * img[w, h, :]

    return img_A, img_B


def generate_database(img_D, img_RGB, seg_thres, index, k_size):
    for i in range(0, 2):
        for j in range(4):
            gau_core_size = [
                RAD(1.0, 1.5),
                RAD(2.5, 3.0),
                RAD(4.0, 4.5),
                RAD(5.5, 6.0),
            ]
            segment_interval = get_segment_interval(img_D, seg_thres)
            guided_bin = get_binary_img(img_D, segment_interval[i], img_D, k_size)
            img_A, img_B = get_focus_img(img_RGB, guided_bin, gau_core_size[j])
            cv.imwrite('../database/nyu/multifocus/background/train/' + str(index) + '.png', img_A)
            cv.imwrite('../database/nyu/multifocus/foreground/train/' + str(index) + '.png', img_B)
            cv.imwrite('../database/nyu/multifocus/bin/train/' + str(index) + '.png', guided_bin)
            cv.imwrite('../database/nyu/multifocus/truth/train/' + str(index) + '.png', img_RGB)
            index += 1
    return index


path_D = '../database/nyu/cropped/D/'
path_RGB = '../database/nyu/cropped/RGB/'
splits_D = os.listdir(path_D)
src_index = 0
num_total = 1
num_segment = 4
print('******开始处理******')
for sp in range(len(splits_D)):
    src_D = path_D + str(sp) + '.png'
    src_RGB = path_RGB + str(sp) + '.jpg'
    image_D = cv.resize(cv.imread(src_D, 0), (512, 512))
    image_RGB = cv.resize(cv.imread(src_RGB), (512, 512))
    segment_threshold = get_threshold_value(image_D, num_segment)
    guide_core = random.randint(2, 8)
    num_total = generate_database(image_D, image_RGB, segment_threshold, num_total, guide_core)
    src_index += 1
    print('>>>第%d张图像处理完毕，目前共有图像%d张。' % (src_index, num_total - 1))
print('******处理完毕******')

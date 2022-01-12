import numpy as np
import h5py
import os
from PIL import Image


f = h5py.File("../database/original_images/nyu_depth_v2_labeled.mat")
images = f["images"]
depths = f["depths"]
images = np.array(images)
depths = np.array(depths)
max_value = np.max(depths)
depths = depths / max_value * 255
depths = depths.transpose((0, 2, 1))

path_converted = '../database/original_images'
if not os.path.isdir(path_converted):
    os.makedirs(path_converted)

images_number = []
for i in range(len(images)):
    images_number.append(images[i])
    a = np.array(images_number[i])
    r = Image.fromarray(a[0]).convert('L')
    g = Image.fromarray(a[1]).convert('L')
    b = Image.fromarray(a[2]).convert('L')
    img = Image.merge("RGB", (r, g, b))
    img = img.transpose(Image.ROTATE_270)
    rgbpath = path_converted + '/RGB/' + str(i) + '.jpg'
    img.save(rgbpath, optimize=True)
    depths_img = Image.fromarray(np.uint8(depths[i]))
    depths_img = depths_img.transpose(Image.FLIP_LEFT_RIGHT)
    depthspath = path_converted + '/D/' + str(i) + '.png'
    depths_img.save(depthspath, 'PNG', optimize=True)

# coding: utf-8

import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import measure
from planefit import planeFit


def read_pitch_angle(metadata_list, index):
    metadata_path = metadata_list[index]
    alllines = open(metadata_path, 'r', encoding='UTF-8').readlines()
    pitch_angle = float(alllines[7].split(":")[1].split("°")[0])
    return pitch_angle



IMG_DIR = r"application/bendingleaves"

show_angle_on_img = False
show_proj_img = False

jpg_files = glob.glob(os.path.join(IMG_DIR, "*.jpg"))
uv_files = list(map(lambda x: os.path.join(os.path.dirname(x), os.path.splitext(os.path.basename(x))[0]+"_uv.txt"), jpg_files))
edge_files = list(map(lambda x: os.path.join(os.path.dirname(x), os.path.splitext(os.path.basename(x))[0] + "-edge.png"), jpg_files))
meta_files = list(map(lambda x: os.path.join(os.path.dirname(x), (os.path.splitext(os.path.basename(x))[0]).rsplit("_", 1)[0] + ".txt"), jpg_files))


pc = np.loadtxt(os.path.join(IMG_DIR, "pc.txt"))

NUM_POINTS = pc.shape[0]

pointVisited = np.zeros((NUM_POINTS, 1))

NUM_CAMERAS = len(jpg_files)

extractedPointCloud = []
rowcols = [] # row and col for each leaf point in pixel coordinate

startedLabelIndex = 0
NUM_CAMERAS = 1
for i in range(NUM_CAMERAS):
    img = Image.open(edge_files[i])
    edge = np.array(img)
    edge = edge > 0
    label_image = measure.label(edge)
    max_val = label_image.max()
    label_image[label_image > 0] += startedLabelIndex
    startedLabelIndex += max_val

    [img_rows, img_cols] = edge.shape

    uv = np.loadtxt(uv_files[i])
    if len(uv) > 0:
        for j in range(NUM_POINTS):
            row = int(np.floor(uv[j][1]))
            col = int(np.floor(uv[j][0]))

            if (pointVisited[j] == 0) and (0 <= row <= img_rows-1) and (0 <= col <= img_cols-1):
                if label_image[row][col] > 0:
                    extractedPointCloud.append(list(pc[j, :])+[label_image[row][col]])
                    rowcols.append([row, col])
                    pointVisited[j] = 1

extractedPointCloud = np.array(extractedPointCloud)
rowcols = np.array(rowcols)

leafIDs = set(extractedPointCloud[:, 3])

# fit plane
angle = read_pitch_angle(meta_files, 0)/180.0*np.pi
print(" - Pitch Angle: ", angle/np.pi*180)
v = np.array([0, np.cos(angle), np.sin(-angle)])
leaf_angles = []
for ID in leafIDs:
    leaf_points = extractedPointCloud[extractedPointCloud[:, 3] == ID][:, 0:3]
    if leaf_points.shape[0] >= 10:
        leaf_points = np.transpose(leaf_points)
        cps, normal = planeFit(leaf_points)
        normal = np.array(normal)
        dotproduct = np.dot(v, normal)
        leaf_angle = np.arccos(dotproduct)/np.pi*180
        if dotproduct < 0:
            leaf_angle = 180 - leaf_angle
        leaf_angles.append(leaf_angle)
        print(leaf_angle)


if show_angle_on_img:
    img = Image.open(jpg_files[0])
    img = np.array(img)
    plt.imshow(img)
    index = 0
    for ID in leafIDs:
        leaf_rowcols = rowcols[extractedPointCloud[:, 3] == ID]
        if leaf_rowcols.shape[0] >= 10:
            rowcol = leaf_rowcols.mean(0)
            row = rowcol[0]
            col = rowcol[1]
            plt.text(col-5, row, "%.2f" % leaf_angles[index])
            plt.text(col - 5, row+55, "ID: %d" % index)
            index += 1

    plt.show()

if show_proj_img:
    img = Image.open(jpg_files[0])
    img = np.array(img)
    plt.imshow(img)
    index = 0
    for ID in leafIDs:
        leaf_rowcols = rowcols[extractedPointCloud[:, 3] == ID]
        if leaf_rowcols.shape[0] >= 10:
            plt.plot(leaf_rowcols[:, 1], leaf_rowcols[:, 0], '*')
            index += 1
    plt.show()

f = open(os.path.join(IMG_DIR, "extracted_pointcloud.txt"), 'w')
for point in extractedPointCloud:
    f.write(' '.join(list(map(lambda x: str(x), point))))
    f.write("\n")
f.close()
import matplotlib.pyplot as plt
import numpy as np 
import argparse


#  TO SO : IMPORT ARGPARSE TO ALLOW FOR DIFFERENT DATASETS
parser = argparse.ArgumentParser()
parser.add_argument('--predicted_path', type=str, default = '../results/default/predicted_path.txt')
parser.add_argument('--ground_truth_path', type=str, default = '../results/default/trajectory.txt')
args = parser.parse_args()

groundTruth = args.ground_truth_path
predictedPath = args.predicted_path

groundTruthFile = open(groundTruth)
predictedPathFile = open(predictedPath)

if ("rock" in groundTruth):
    xsGt = []
    ysGt = []
    zsGt = []
    for line in groundTruthFile:
        l = line.split(",")
        xsGt.append(float(l[1]))
        ysGt.append(float(l[2]))
        zsGt.append(float(l[3]))

else:
    xsGt = []
    ysGt = []
    zsGt = []
    for line in groundTruthFile:
        l = line.split(" ")
        xsGt.append(float(l[2]))
        ysGt.append(float(l[3]))
        zsGt.append(float(l[4]))

xsPred = []
ysPred = []
zsPred = []
for line in predictedPathFile:
    l = line.split(" ")
    xsPred.append(float(l[1]))
    ysPred.append(float(l[2]))
    zsPred.append(float(l[3]))

print(f'# of pred samples : {len(xsPred)}, # of gt samples : {len(xsGt)}')

xsPred.reverse()

ax = plt.axes(projection='3d')
ax.plot3D(xsPred, ysPred, zsPred, 'green', label='Predicted Path')
ax.plot3D(xsGt, ysGt, zsGt,'red', label= 'Ground Truth Path')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()

plt.show()
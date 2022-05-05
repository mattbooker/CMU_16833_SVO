import matplotlib.pyplot as plt
import numpy as np 
import argparse

#  TO SO : IMPORT ARGPARSE TO ALLOW FOR DIFFERENT DATASETS
parser = argparse.ArgumentParser()
parser.add_argument('--predicted_path', type=str, default = '../results/default/predicted_path.txt')
parser.add_argument('--ground_truth_path', type=str, default 
= '../results/default/trajectory.txt')
parser.add_argument('--svo_og_path', type=str, default = '../results/default/from_svo_og.txt')
args = parser.parse_args()

groundTruth = args.ground_truth_path
predictedPath = args.predicted_path
svoOgPath = args.svo_og_path

groundTruthFile = open(groundTruth)
predictedPathFile = open(predictedPath)
svoPathFile = open(svoOgPath)

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

xsGt = np.array(xsGt, dtype=np.float64)
ysGt = np.array(ysGt, dtype=np.float64)
zsGt = np.array(zsGt, dtype=np.float64)

xsGt = xsGt - xsGt.mean()
xsGt = xsGt / xsGt.max()

ysGt = ysGt - ysGt.mean()
ysGt = ysGt / ysGt.max()

# zsGt = zsGt - zsGt.mean()
# zsGt = zsGt / zsGt.max()

zsGt = 2.0

xsPred = []
ysPred = []
zsPred = []

for line in predictedPathFile:
    l = line.split(" ")
    xsPred.append(float(l[1]))
    ysPred.append(float(l[2]))
    zsPred.append(float(l[3]))

xsPred = np.array(xsPred, dtype=np.float64)
ysPred = np.array(ysPred, dtype=np.float64)
zsPred = np.array(zsPred, dtype=np.float64)

xsPred = xsPred - xsPred.mean()
xsPred = xsPred / xsPred.max()

ysPred = ysPred - ysPred.mean()
ysPred = ysPred / ysPred.max()

zsPred = zsPred - zsPred.mean()
zsPred = zsPred / zsPred.max()

zsPred = 2.0

xsSVO = []
ysSVO = []
zsSVO = []

for line in svoPathFile:
    l = line.split(" ")
    xsSVO.append(float(l[0]))
    ysSVO.append(float(l[1]))
    zsSVO.append(float(l[2]))

xsSVO = np.array(xsSVO, dtype=np.float64)
ysSVO = np.array(ysSVO, dtype=np.float64)
zsSVO = np.array(zsSVO, dtype=np.float64)

xsSVO = xsSVO - xsSVO.mean()
xsSVO = xsSVO / xsSVO.max()

ysSVO = ysSVO - ysSVO.mean()
ysSVO = ysSVO / ysSVO.max()

zsSVO = zsSVO - zsSVO.mean()
zsSVO = zsSVO / zsSVO.max()

zsSVO = 2.0

print(f'# of SVO samples : {len(xsSVO)}, # of pred samples : {len(xsPred)}, # of gt samples : {len(xsGt)}')

# xsPred.reverse()
# xsSVO.reverse()

ax = plt.axes(projection='3d')
ax.plot3D(xsPred[::-1], ysPred, zsPred, 'green', label='Predicted Path')
ax.plot3D(xsGt, ysGt, zsGt,'red', label= 'Ground Truth Path')
ax.plot3D(xsSVO[::-1], ysSVO, zsSVO,'blue', label= 'SVO Path')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()

plt.show()
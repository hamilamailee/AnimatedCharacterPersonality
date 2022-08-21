import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import numpy as np
import os


plt.style.use('_mpl-gallery')

file_path = r'CollectedData_Byron_pos.csv'
file_path = r'CollectedData_Byron - nolost_pos.csv'

all_frames_x = []
all_frames_y = []

df = pd.read_csv(file_path)
all_frames_label = set(df.T[df.T.columns[0]].to_list()[1:])

for row in range(2, len(df)):
    X = []
    Y = []
    for x in range(1, len(df.columns), 3):
        X.append(float(df[df.columns[x]][row]))
    for y in range(2, len(df.columns), 3):
        Y.append(float(df[df.columns[y]][row]))
    all_frames_x.append(X)
    all_frames_y.append(Y)

print("creating images...")

minx = np.min(all_frames_x)
miny = np.min(all_frames_y)
maxx = np.max(all_frames_x)
maxy = np.max(all_frames_y)
xlim = maxx - minx
ylim = maxy - miny

for i in range(len(all_frames_y)):
    x = all_frames_x[i]
    y = all_frames_y[i]
    fig, ax = plt.subplots()
    ax.set(xlim=(minx - xlim/10, maxx + xlim/10), xticks=np.arange(minx, maxx, 50),
           ylim=(miny - ylim/10, maxy + ylim/10), yticks=np.arange(miny, maxy, 50))
    ax.scatter(np.asarray(x), np.asarray(y))
    # for j, txt in enumerate(all_frames_label):
    #     ax.annotate(txt, (x[j], y[j]))
    plt.savefig(fname=str(i)+".png", dpi='figure', format='png')
    # plt.show()

print("creating video...")
frameSize = (200, 200)
out = cv2.VideoWriter('output_video.avi',
                      cv2.VideoWriter_fourcc(*'DIVX'), 20, frameSize)

for i in range(len(all_frames_y)):
    img = cv2.imread(str(i)+".png")
    out.write(img)
    os.remove(str(i) + ".png")

out.release()

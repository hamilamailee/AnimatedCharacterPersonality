import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import numpy as np
import os


plt.style.use('_mpl-gallery')

file_path = r'C:\Users\HP\Documents\Computer Graphics\CSV to BVH animals\AnimatedCharacterPersonality\CollectedData_Byron.csv'
all_frames_x = []
all_frames_y = []

df = pd.read_csv(file_path)
all_frames_label = set(df.T[df.T.columns[0]].to_list()[1:])
for row in range(2, len(df)):
    X = []
    Y = []
    for x in range(1, len(df.columns), 3):
        X.append(-1 * float(df[df.columns[x]][row]))
    for y in range(2, len(df.columns), 3):
        Y.append(-1 * float(df[df.columns[y]][row]))
    all_frames_x.append(X)
    all_frames_y.append(Y)

print("creating images...")
for i in range(len(all_frames_y)):
    x = all_frames_x[i]
    y = all_frames_y[i]
    fig, ax = plt.subplots()
    ax.set(xlim=(-200, 0), xticks=np.arange(-100, 0, 50),
           ylim=(-200, 0), yticks=np.arange(-100, 0, 50))
    ax.scatter(np.asarray(x), np.asarray(y))
    # for j, txt in enumerate(all_frames_label):
    #     ax.annotate(txt, (x[j], y[j]))
    plt.savefig(fname=str(i)+".png", dpi='figure', format='png')
    # plt.show()

print("creating video...")
frameSize = (200, 200)
out = cv2.VideoWriter('output_video.avi',
                      cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)

for i in range(len(all_frames_y)):
    img = cv2.imread(str(i)+".png")
    out.write(img)
    os.remove(str(i) + ".png")

out.release()

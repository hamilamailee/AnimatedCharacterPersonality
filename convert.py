import pandas as pd
import numpy as np
all_joints = []


class Joint:
    def __init__(self, name) -> None:
        self.name = name
        self.connected_joints = []
        self.add_joint()

    def get_cords_from_df(self, df) -> None:
        model = df[self.name]
        model.columns = ['x', 'y', 'z']
        model['z'] = 0
        self.cords = model
        self.cords['frame'] = self.cords.index - 1

    def add_joint(self) -> None:
        all_joints.append(self)

    def connect_joints(self, joint) -> None:
        self.connected_joints.append(joint)

    def print_joint(self) -> None:
        print("name : ", self.name)
        print("connected joints : ")
        for i in range(len(self.connected_joints)):
            print(i.name)


file_path = "My Version\p1DLC_effnet_b0_Simba ModifiedAug27shuffle0_150000.csv"
df = pd.read_csv(file_path)
df.columns = df[df['scorer'] == 'bodyparts'].to_numpy().tolist()
df = df.drop([0, 1], axis=0)

for i in np.unique(list(df.columns)):
    if i != "bodyparts":
        Joint(i)

for j in all_joints:
    j.get_cords_from_df(df)

import yaml
import numpy as np
from asyncio.windows_events import NULL
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


all_joints = []


class Joint:
    def __init__(self, name) -> None:
        self.name = name
        self.connected_joints = []
        self.add_joint()

    def get_joint(name):
        for j in all_joints:
            if j.name == name:
                return j
        return None

    def connect_joints(bone) -> None:
        j1 = Joint.get_joint(bone[0])
        j2 = Joint.get_joint(bone[1])
        j1.connected_joints.append(j2)
        j2.connected_joints.append(j1)

    def get_cords_from_df(self, df) -> None:
        model = df[self.name]
        model.columns = ['x', 'y', 'z']
        model['z'] = 0
        self.cords = model
        self.cords['frame'] = self.cords.index - 1

    def add_joint(self) -> None:
        all_joints.append(self)

    def print_joint(self) -> None:
        print("name : ", self.name)
        print("connected joints : ")
        for i in range(len(self.connected_joints)):
            print(self.connected_joints[i].name)


# file_path = input("Enter the path of your csv file: ")
# conf_path = input("Enter the path of your config.yaml file: ")

file_path = "p1DLC_effnet_b0_Simba ModifiedAug27shuffle0_150000.csv"
conf_path = "config.yaml"

df = pd.read_csv(file_path)
df.columns = df[df['scorer'] == 'bodyparts'].to_numpy().tolist()
df = df.drop([0, 1], axis=0)

for i in np.unique(list(df.columns)):
    if i != "bodyparts":
        Joint(i)

for j in all_joints:
    j.get_cords_from_df(df)

with open(conf_path) as file:
    skeleton = yaml.load(file, Loader=yaml.FullLoader)['skeleton']
    for bone in skeleton:
        Joint.connect_joints(bone)

for i in all_joints:
    i.print_joint()

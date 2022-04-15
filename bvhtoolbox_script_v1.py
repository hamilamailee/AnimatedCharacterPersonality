from cmath import acos, pi, sqrt
from statistics import mode
from unicodedata import name
import yaml
import numpy as np
from numpy.linalg import norm
from asyncio.windows_events import NULL
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


all_joints = []
all_bones = []
fc = 0


def has_nan(array) -> bool:
    return np.isnan(np.sum(array))


def degy(vec12y) -> np.double:
    cos_theta = vec12y[0] / sqrt(vec12y[0] ** 2 + vec12y[2] ** 2)
    if vec12y[2] <= 0:
        return np.real(acos(cos_theta) * 180 / pi)
    else:
        return np.real(acos(cos_theta) * -180 / pi)


def degz(vec12z) -> np.double:
    cos_theta = sqrt(vec12z[0] ** 2 + vec12z[2] ** 2) / \
        sqrt(vec12z[0] ** 2 + vec12z[1] ** 2+vec12z[2] ** 2)
    if vec12z[1] > 0:
        return np.real(acos(cos_theta) * 180 / pi)
    elif vec12z[1] == 0:
        return 0.0
    else:
        return np.real(acos(cos_theta) * -180 / pi)


class Bone:
    def __init__(self, joint1, joint2) -> None:
        self.joint1 = joint1
        self.joint2 = joint2
        self.bone_length()

    def get_angles() -> np.array:
        all_rotations = np.empty((fc + 1, 0))
        for b in all_bones:
            vec1 = b.joint1.cords[:, :3]
            vec2 = b.joint2.cords[:, :3]
            vec = vec2 - vec1
            info = np.array(
                [[b.joint1.name+".y", b.joint1.name+".z", b.joint1.name+".x"]])
            col1 = np.apply_along_axis(degy, 1, vec)
            col2 = np.apply_along_axis(degz, 1, vec)
            col3 = np.zeros_like(col1)
            rotation_angles = np.column_stack((col1, col2, col3))
            rotation_angles = np.row_stack((info, rotation_angles))
            all_rotations = np.concatenate(
                (all_rotations, rotation_angles), axis=1)
        return all_rotations

    def bone_length(self) -> None:
        cords1 = self.joint1.cords[:, :3]
        cords2 = self.joint2.cords[:, :3]
        dist = []
        for i in range(len(cords1)):
            if np.isnan(norm(cords1[i] - cords2[i])):
                continue
            dist.append(norm(cords1[i] - cords2[i]))
        self.length = np.average(dist)

    def print_bone(self) -> None:
        string = "({bone1} , {bone2}) -> length: {length}".format(
            bone1=self.joint1.name,
            bone2=self.joint2.name,
            length=self.length)
        print(string)


class Joint:
    def __init__(self, name) -> None:
        self.name = name
        self.connected_joints = []
        all_joints.append(self)

    def get_joint(name):
        for j in all_joints:
            if j.name == name:
                return j
        return None

    def get_root() -> str:
        count = len(all_joints[0].connected_joints)
        name = None
        for j in all_joints:
            if len(j.connected_joints) > count:
                count = len(j.connected_joints)
                name = j.name
        return name

    def connect_joints(bone) -> None:
        j1 = Joint.get_joint(bone[0])
        j2 = Joint.get_joint(bone[1])
        j1.connected_joints.append(j2)
        j2.connected_joints.append(j1)
        Bone(j1, j2)

    def get_cords_from_df(self, df) -> None:
        model = df[self.name].astype(np.float64)
        model.columns = ['x', 'y', 'z']
        model.loc[model['z'] < 0.4, :] = np.NaN
        model.loc[model['z'] >= 0.4, 'z'] = 0
        model['frame'] = model.index - 1
        self.cords = model.to_numpy()

    def print_joint(self) -> None:
        print("name : ", self.name)
        print("connected joints : ")
        for i in range(len(self.connected_joints)):
            print(self.connected_joints[i].name, end=" - ")
        print()


# file_path = input("Enter the path of your csv file: ")
# conf_path = input("Enter the path of your config.yaml file: ")

file_path = "p1DLC_effnet_b0_Simba ModifiedAug27shuffle0_150000.csv"
conf_path = "config.yaml"

df = pd.read_csv(file_path)
df.columns = df[df['scorer'] == 'bodyparts'].to_numpy().tolist()
df = df.drop([0, 1], axis=0)
fc = len(df)

for i in np.unique(list(df.columns)):
    if i != "bodyparts":
        Joint(i)

for j in all_joints:
    j.get_cords_from_df(df)

with open(conf_path) as file:
    skeleton = yaml.load(file, Loader=yaml.FullLoader)['skeleton']
    for bone in skeleton:
        Joint.connect_joints(bone)

root = Joint.get_root()

# hierarchy


def traverse(hierarchy, parent):
    if parent.name == root:
        hierarchy = np.row_stack(
            (hierarchy, np.array([root, "", 0, 0, 0])))
    index = initial_frame_index()
    for c in parent.connected_joints:
        if c.name not in hierarchy:
            all_bones.append(Bone(parent, c))
            cord = (c.cords[index] - parent.cords[index])[:3]
            bone = np.array([c.name, parent.name])
            cord = np.concatenate((bone, cord))
            hierarchy = np.row_stack((hierarchy, cord))
            hierarchy = traverse(hierarchy, c)
    return hierarchy


def initial_frame_index() -> int:
    for f in range(fc):
        found = True
        for j in all_joints:
            if has_nan(j.cords[f][:3]):
                found = False
        if found:
            return f


hierarchy = np.empty((0, 5))
hierarchy = traverse(hierarchy, Joint.get_joint(root))
df = pd.DataFrame(
    hierarchy, columns=['joint', 'parent', 'offset.x', 'offset.y', 'offset.z'])
df.to_csv(file_path.split(
    ".")[0]+"_hierarchy.csv", index=False)


# positions

df = pd.DataFrame(range(fc), columns=['time'])
df['time'] = df['time'] * 1 / 60
for j in all_joints:
    df[j.name+".x"] = j.cords[:, 0]
    df[j.name+".y"] = j.cords[:, 1]
    df[j.name+".z"] = j.cords[:, 2]
df.to_csv(file_path.split(
    ".")[0]+"_pos.csv", index=False)

# rotations

angles = Bone.get_angles()
df = pd.DataFrame(angles[1:, :], columns=angles[0])
df['time'] = list(range(fc))
df['time'] = df['time'] * 1 / 60
df.to_csv(file_path.split(
    ".")[0]+"_rot.csv", index=False)

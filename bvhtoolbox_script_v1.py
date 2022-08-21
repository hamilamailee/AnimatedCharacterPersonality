from __future__ import annotations
from cmath import acos, pi
import yaml
import numpy as np
from bvhtoolbox.convert import csv2bvh_file
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


all_joints = dict()


def has_nan(array) -> bool:
    return np.isnan(np.sum(array))


def degy(vec12y) -> np.double:
    return 0


def degz(vec12z, offset) -> np.double:
    dot = np.dot(vec12z, offset)
    cross = np.cross(vec12z, offset)
    norm = np.linalg.norm(vec12z) * np.linalg.norm(offset)

    if norm == 0:
        return 0

    cos_theta = dot/norm
    sin_theta = cross[2]/norm

    return np.real(acos(cos_theta)) * 180 * (-1) * np.sign(sin_theta) / pi


def bfs(main_root: str, root: str):
    root_j = Joint.get_joint(root)
    if root == main_root:
        Bone(root_j, root_j)
    for j in root_j.connected_joints:
        if root_j in j.connected_joints:
            Bone(j, root_j)
            j.connected_joints.remove(root_j)
    for j in root_j.connected_joints:
        bfs(main_root, j.name)


def dfs(bone: Bone, ordered: list):
    if bone not in ordered:
        ordered.append(bone)
        for b in Bone.all_bones:
            if b.parent.name == bone.child.name:
                dfs(b, ordered)
    return ordered


def read_file(file_path) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = df[df['scorer'] == 'bodyparts'].to_numpy().tolist()
    return df.drop([0, 1], axis=0)


class BVH:
    def __init__(self, file_path: str, conf_path: str, root: str) -> None:
        self.df = read_file(file_path)
        self.fc = len(self.df)
        self.root = root
        with open(conf_path) as file:
            self.conf = yaml.load(file, Loader=yaml.FullLoader)

    def initial_frame_index(self) -> int:
        for f in range(self.fc):
            found = True
            for j, joint in all_joints.items():
                if joint.cords is not None and has_nan(joint.cords[f][:3]):
                    found = False
            if found:
                return f
        return None

    def get_angles(self) -> np.array:
        all_rotations = np.empty((self.fc + 1, 0))
        for b in Bone.all_bones[:]:
            if b.child.is_leaf():
                continue
            vec1 = b.parent.cords[:, :3]
            vec2 = b.child.cords[:, :3]
            vec = vec2 - vec1
            info = np.array(
                [[b.child.name+".y", b.child.name+".z", b.child.name+".x"]])
            col1 = np.apply_along_axis(degy, 1, vec)
            col3 = np.apply_along_axis(degz, 1, vec, b.offset)
            col2 = np.zeros_like(col1)
            rotation_angles = np.column_stack((col1, col2, col3))
            rotation_angles = np.row_stack((info, rotation_angles))
            all_rotations = np.concatenate(
                (all_rotations, rotation_angles), axis=1)
        return all_rotations

    def offset_and_cord(self, hierarchy: np.array):
        for j, joint in all_joints.items():
            joint.get_cords_from_df(self.df, self.conf['alphavalue'])
        index = self.initial_frame_index()
        for b in Bone.all_bones:
            b.offset = (b.child.cords[index] - b.parent.cords[index])[:3]
            if b.child.name == self.root:
                bone = np.array([b.child.name, ""])
            else:
                bone = np.array([b.child.name, b.parent.name])
            cord = np.concatenate((bone, b.offset))
            hierarchy = np.row_stack((hierarchy, cord))
        return hierarchy

    def write_hierarchy_file(self, file_path: str):
        bfs(self.root, self.root)
        Bone.clean_hierarchy()
        Bone.all_bones = dfs(Bone.all_bones[0], [])
        hierarchy = np.empty((0, 5))
        hierarchy = self.offset_and_cord(hierarchy)
        df = pd.DataFrame(hierarchy, columns=[
                          'joint', 'parent', 'offset.x', 'offset.y', 'offset.z'])
        df.to_csv(file_path.split(".")[0]+"_hierarchy.csv", index=False)
        return file_path.split(".")[0]+"_hierarchy.csv"

    def write_position_file(self, file_path: str):
        df = pd.DataFrame(range(self.fc), columns=['time'])
        df['time'] = df['time'] * 1 / 60
        for name, j in all_joints.items():
            if len(j.connected_joints) != 0:
                df[j.name+".x"] = j.cords[:, 0]
                df[j.name+".y"] = j.cords[:, 1]
                df[j.name+".z"] = j.cords[:, 2]
        df.to_csv(file_path.split(".")[0]+"_pos.csv", index=False)
        return file_path.split(".")[0]+"_pos.csv"

    def write_rotation_file(self, file_path: str):
        angles = self.get_angles()
        df = pd.DataFrame(angles[1:, :], columns=angles[0])
        df['time'] = list(range(self.fc))
        df['time'] = df['time'] * 1 / 60
        df.to_csv(file_path.split(".")[0]+"_rot.csv", index=False)
        return file_path.split(".")[0]+"_rot.csv"


class Bone:
    all_bones = []

    def __init__(self, child: Joint, parent: Joint) -> None:
        self.child = child
        self.parent = parent
        self.offset = None
        Bone.all_bones.append(self)

    def clean_hierarchy():
        for bi in Bone.all_bones:
            for bj in Bone.all_bones:
                if bi != bj and bi.child.name == bj.child.name:
                    Bone.all_bones.remove(bj)

    def print_bone(self) -> None:
        string = f"(parent : {self.parent.name} , child : {self.child.name}) -> length: {self.length}"
        print(string)


class Joint:
    def __init__(self, name):
        self.name = name
        self.connected_joints = []
        self.children = []
        self.cords = None
        all_joints[name] = self

    def get_joint(name: str) -> Joint:
        if name not in all_joints:
            return None
        return all_joints[name]

    def connect_joints(joint1: str, joint2: str) -> None:
        j1 = Joint.get_joint(joint1)
        j2 = Joint.get_joint(joint2)
        j1.connected_joints.append(j2)
        j2.connected_joints.append(j1)

    def is_leaf(joint: Joint):
        for b in Bone.all_bones:
            if b.parent == joint:
                return False
        return True

    def get_cords_from_df(self, df, alpha) -> None:
        if self.name in df.columns:
            model = df[self.name].astype(np.float64)
            model['z'] = 1
            model.columns = ['x', 'y', 'z']
            model.loc[model['z'] < alpha, :] = np.NaN
            model.loc[model['z'] >= alpha, 'z'] = 0
            model['frame'] = model.index - 1
            model['y'] = -model['y']
            self.cords = model.to_numpy()

    def print_joint(self) -> None:
        print("name : ", self.name)
        print("connected joints : ")
        for i in range(len(self.connected_joints)):
            print(self.connected_joints[i].name, end=" - ")
        print("\n___________________________________")


# file_path = input("Enter the path of your csv file: ")
# conf_path = input("Enter the path of your config.yaml file: ")
# root = input("Enter the name of your root: ")

# file_path = "test\p1DLC_effnet_b0_Simba ModifiedAug27shuffle0_150000.csv"
# conf_path = "config.yaml"
# root = "tailbase"

file_path = r"AnimatedCharacterPersonality\CollectedData_Byron - nolost.csv"
conf_path = r"AnimatedCharacterPersonality\config.yaml"
root = "Ischium"

gen_bvh = BVH(file_path, conf_path, root)
skeleton = gen_bvh.conf['skeleton']

for bone in skeleton:
    if bone[0] not in all_joints:
        Joint(bone[0])
    if bone[1] not in all_joints:
        Joint(bone[1])
    Joint.connect_joints(bone[0], bone[1])

hierarchy = gen_bvh.write_hierarchy_file(file_path)
position = gen_bvh.write_position_file(file_path)
rotation = gen_bvh.write_rotation_file(file_path)

csv2bvh_file(hierarchy, position, rotation, r"./nolost-colschange.bvh")
# csv2bvh_file(hierarchy, position, rotation, r"test\simba.bvh")

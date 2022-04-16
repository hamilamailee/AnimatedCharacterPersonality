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


def initial_frame_index() -> int:
    for f in range(fc):
        found = True
        for j in all_joints:
            if has_nan(j.cords[f][:3]):
                found = False
        if found:
            return f


class Bone:

    def __init__(self, joint1, joint2) -> None:
        """Create bone form two joints.

        Args:
            joint1 (Joint): First joint of a bone
            joint2 (Joint): Second joint of a bone
        """
        self.joint1 = joint1
        self.joint2 = joint2
        self.bone_length()

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
        """Create joint by getting its name.
        Joint has an array called children, which keeps the name of the connected joints.
        Parent is also string. 

        Args:
            name (str): name of joint
        """
        self.name = name
        self.parent = None
        self.children = []
        all_joints.append(self)

    def get_joint(name):
        """Get joint by name

        Args:
            name (str): name of wanted joint

        Returns:
            Joint: desired joint is returned
        """
        for j in all_joints:
            if j.name == name:
                return j
        return None

    def get_root() -> str:
        """Find the root of the skeleton

        Returns:
            str: name of root joint
        """
        count = len(all_joints[0].children)
        name = None
        for j in all_joints:
            if len(j.children) > count:
                count = len(j.children)
                name = j.name
        return Joint.get_joint(name)

    def connect_joints(bone) -> None:
        j1 = Joint.get_joint(bone[0])
        j2 = Joint.get_joint(bone[1])
        j1.children.append(j2.name)
        j2.children.append(j1.name)
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
        for i in range(len(self.children)):
            print(self.children[i], end=" - ")
        print()


class Hierarchy:
    def __init__(self, file_path) -> None:
        self.root = Joint.get_root()
        self.tree = self.skeleton_tree()
        self.file_path = file_path
        self.write_hierarchy_csv()

    def skeleton_tree(self):
        traversed = [self.root.name]
        for t in traversed:
            for j in all_joints:
                if j.name not in traversed and t in j.children:
                    j.children.remove(t)
                    j.parent = t
                    traversed.append(j.name)

    def write_hierarchy_csv(self):
        hierarchy = np.empty((0, 5))
        hierarchy = np.row_stack(
            (hierarchy, np.array([self.root.name, "", 0, 0, 0])))
        index = initial_frame_index()
        for j in all_joints:
            if j.name != self.root.name:
                print(j.name)
                parent = Joint.get_joint(j.parent)
                cords = (j.cords[index] - parent.cords[index])[:3]
                bone = np.array([j.name, j.parent])
                cords = np.concatenate((bone, cords))
                hierarchy = np.row_stack((hierarchy, cords))
        print(hierarchy)
        df = pd.DataFrame(
            hierarchy, columns=['joint', 'parent', 'offset.x', 'offset.y', 'offset.z'])
        df.to_csv(self.file_path.split(
            ".")[0]+"_hierarchy.csv", index=False)

# file_path = input("Enter the path of your csv file: ")
# conf_path = input("Enter the path of your config.yaml file: ")


file_path = "test/p1DLC_effnet_b0_Simba ModifiedAug27shuffle0_150000.csv"
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

# hierarchy
hierarchy = Hierarchy(file_path)

# positions
df = pd.DataFrame(range(fc), columns=['time'])
df['time'] = df['time'] * 1 / 60
for j in all_joints:
    df[j.name+".x"] = j.cords[:, 0]
    df[j.name+".y"] = j.cords[:, 1]
    df[j.name+".z"] = j.cords[:, 2]
df.to_csv(file_path.split(
    ".")[0]+"_pos.csv", index=False)

from bvhtoolbox import BvhNode, BvhTree
import pandas as pd
import numpy as np

# file_path = input("Enter the path of your csv file: ")
# conf_path = input("Enter the path of your config.yaml file: ")

file_path = "p1DLC_effnet_b0_Simba ModifiedAug27shuffle0_150000.csv"
conf_path = "config.yaml"

df = pd.read_csv(file_path)
df.columns = df[df['scorer'] == 'bodyparts'].to_numpy().tolist()
df = df.drop([0, 1], axis=0)
fc = len(df)

for i in np.unique()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
rcParams['figure.figsize'] = 100, 30
from constant import *


def create_train_data(data_path, event_path):
    data = pd.ExcelFile(data_path)
    event = pd.ExcelFile(event_path)
    sheet_names = data.sheet_names

    for sheet_name in sheet_names:
        char = sheet_name[-2:-1]
        position = positions[char]
        sheet_data = pd.read_excel(data, sheet_name, header=None).to_numpy()
        sheet_event = pd.read_excel(event, sheet_name, header=None).to_numpy()
        for i in range(sheet_event.shape[0]):
            start_idx = int(sheet_event[i][1])
            sample_points = sheet_data[start_idx: start_idx+125] # 125 (1/250 x 125 s) --> 0.5 s --> 500 ms
            if int(sheet_event[i][0]) in position:
                # positive data
                save_path = f'datas/train_data/positive/{data_path.split("/")[1]}_{sheet_name}_{i}.npz'
            else:
                save_path = f'datas/train_data/negative/{data_path.split("/")[1]}_{sheet_name}_{i}.npz'
                # negative data
            with open(save_path, "wb") as outfile:
                np.save(outfile, sample_points)

if __name__ == "__main__":
    for i in range(1, 6):
        data_path = f"datas/S{i}/S{i}_train_data.xlsx"
        event_path = f"datas/S{i}/S{i}_train_event.xlsx"
        print(f"process S{i}")
        create_train_data(data_path, event_path)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pylab import rcParams
rcParams['figure.figsize'] = 100, 30


i = 1
train_path = f"datas/S{i}/S{i}_train_data.xlsx"
event_path = f"datas/S{i}/S{i}_train_event.xlsx"

train_data = pd.ExcelFile(train_path)
event_data = pd.ExcelFile(event_path)

datas = pd.read_excel(train_data, 'char01(B)', header=None).to_numpy()
events = pd.read_excel(event_data, 'char01(B)', header=None).to_numpy()

position = [6, 9]

x = np.arange(0, datas.shape[0])

datas = (datas-datas.mean(axis=0)) / datas.std(axis=0)

plt.plot(x, np.mean(datas, axis=1))
for j in range(events.shape[0]):
    if int(events[j][0]) in position:
        plt.axvline(events[j][1], color="red")
        plt.axvline(events[j][1]+75, color="blue")
    #else:
    #    plt.axvline(events[j][1], color="blue")

plt.savefig(f"plot_{i}.png")
plt.clf()


print("datas.shape", datas.shape)
print(datas)

import torch
from model1 import Model
from model import Model
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from constant import *
from torch import nn

if __name__ == "__main__":
    positive_dataset = MyDataset(kind="positive")
    negative_dataset = MyDataset(kind="negative")
    batch_size = 20
    iterations = 500000

    positive_loader = DataLoader(positive_dataset, batch_size=batch_size,
            shuffle=True, num_workers=3)

    negative_loader = DataLoader(negative_dataset, batch_size=batch_size,
            shuffle=True, num_workers=3)

    positive_itr = iter(positive_loader)
    negative_itr = iter(negative_loader)
    model = Model()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for itr in range(iterations):
        optimizer.zero_grad()
        try:
            positive_data = next(positive_itr)['data'] # batch_size x 20 x 125
        except Exception:
            positive_itr = iter(positive_loader)
            positive_data = next(positive_itr)['data'] 

        try:
            negative_data = next(negative_itr)['data']
        except Exception:
            negative_itr = iter(negative_loader)
            negative_data = next(negative_itr)['data']

        data = torch.cat([positive_data, negative_data], dim=0).to(device).float()[:, :, :100] # batch_size x 20 x 125

        y = model(data)
        targets = torch.tensor([1]*batch_size + [0]*batch_size, device=device)
        loss = criterion(y, targets)
        loss.backward()
        optimizer.step()

        y_max = y.argmax(dim=1)
        accuracy = (y_max==targets).float().sum() / (2*batch_size)
        print(f"iteration:{itr}, loss:{loss}, accuracy:{accuracy}")

        if itr % 500==0:
            torch.save(model.state_dict(), f"models/model_{itr}.pt")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from util import *
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.autograd import Variable



filtered_data = pd.read_csv('import.csv')
filtered_data.head(2)


X = filtered_data[['year_code', 'pos_sentiment', 'neg_sentiment', 'winery', 'cat_country',
       'country_code', 'province_code', 'designation_code', 'variety',
       'num_description', 'point_cat']]
y = filtered_data['price']


x_train = np.array(X.values, dtype=np.float32)

y_train = np.array(y.values, dtype=np.float32)

x_train.shape,y_train.shape


x_train = torch.from_numpy(x_train)

y_train = torch.from_numpy(y_train)

# Linear Regression Model
class linearRegression(nn.Module):
    def __init__(self):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(11, 1)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out


model = linearRegression()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)


num_epochs = 100
for epoch in range(num_epochs):
    inputs = x_train
    target = y_train

    # forward
    out = model(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f'Epoch[{epoch+1}/{num_epochs}], loss: {loss.item():.6f}')

model.eval()
with torch.no_grad():
    predict = model(x_train)
predict = predict.data.numpy()
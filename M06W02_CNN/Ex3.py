import torch
import torch.nn as nn
import numpy as np

data = np.array([[[1, 6], [3, 4]]])
data = torch.tensor(data, dtype=torch.float32)

bnorm = nn.BatchNorm2d(1)
data = data.unsqueeze(0)
with torch.no_grad():
    output = bnorm(data)
print(output)


a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[1, 2], [3, 4]])

a = a.reshape(1, 2, 2)
b = b.reshape(1, 2, 2)

c = torch.cat((a, b))
print(c)


seed = 1
torch.manual_seed(seed)
input_tensor = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1)
conv_output = conv_layer(input_tensor)

with torch.no_grad():
    output = conv_output + input_tensor
print(output)
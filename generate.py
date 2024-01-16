from model import *
import numpy as np
sample_number = 10
model = VAE()
state_dict = torch.load('./model5_2.pth')
model.load_state_dict(state_dict)
for i in range(sample_number):
    sample, label = model.sample()
    sample = sample.detach().numpy()
    print(sample.shape)
    print(label.detach().numpy().shape)
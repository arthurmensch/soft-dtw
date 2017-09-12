import time
import torch
from torch.autograd import Variable

from sdtw.pytorch import DTW, Distance

m, n = 8000, 4000
gamma = .1

x = Variable(torch.randn((m, 1)))
y = Variable(torch.randn((n, 1)))

distance = Distance()
dtw = DTW(gamma=gamma)

D = distance(x, y)

D = Variable(D.data, requires_grad=True)
t0 = time.clock()
loss = dtw(D)
elapsed = time.clock() - t0
print('Loss', loss.data[0])
print('Forward time', elapsed, 's')

t0 = time.clock()
loss.backward()
alignment = D.grad.data.numpy()
elapsed = time.clock() - t0
print('Soft alignment')
print(alignment)
print('Backward time', elapsed, 's')

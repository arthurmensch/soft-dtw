import torch
from torch.autograd import Variable

from sdtw.pytorch import DTW, Distance

m, n = 4000, 4000
gamma = .001

x = Variable(torch.randn((m, 1)))
y = Variable(torch.randn((n, 1)))

distance = Distance()
dtw = DTW(gamma=gamma)

D = distance(x, y)

D = Variable(D.data, requires_grad=True)
loss = dtw(D)
loss.backward()
import torch
from torch.autograd import Variable
import numpy as np
from sdtw.soft_dtw_fast import _soft_dtw, _soft_dtw_grad


class DTW(torch.autograd.Function):
    def __init__(self, gamma):
        super(DTW, self).__init__()
        self.gamma = gamma

    def forward(self, D):
        m, n = D.size()

        D_ = D.numpy()
        R_ = np.zeros((m + 2, n + 2), dtype=np.float32)
        _soft_dtw(D_, R_, gamma=self.gamma)

        R = torch.from_numpy(R_)
        self.save_for_backward(D)
        self.intermediate_tensors = R,
        return R[m, n:n + 1]

    def backward(self, grad_output):
        D, = self.saved_tensors
        R, = self.intermediate_tensors
        m, n = D.size()

        D_ = np.zeros((m + 1, n + 1), dtype=np.float32)
        D_[:-1, :-1] = D.numpy()
        R_ = R.numpy()
        E_ = np.zeros((m + 2, n + 2), dtype=np.float32)
        _soft_dtw_grad(D_, R_, E_, gamma=self.gamma)

        return torch.from_numpy(E_[1:-1, 1:-1]) * grad_output[0]


class Distance(torch.nn.Module):
    def __init__(self):
        super(Distance, self).__init__()

    def forward(self, x, y):
        xTy = 2 * x.matmul(y.transpose(0, 1))
        x2 = torch.sum(x * x, 1).view((-1, 1))
        y2 = torch.sum(y * y, 1).view((1, -1))
        K = x2 + y2 - xTy
        return K
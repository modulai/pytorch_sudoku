import torch
import torch.nn as nn


class SudokuSolver(nn.Module):
    def __init__(self, constraint_mask, n=9, hidden1=100):
        super(SudokuSolver, self).__init__()
        self.constraint_mask = constraint_mask.view(1, n * n, 3, n * n, 1)
        self.n = n
        self.hidden1 = hidden1

        # Feature vector is the 3 constraints
        self.input_size = 3 * n

        self.l1 = nn.Linear(self.input_size,
                            self.hidden1, bias=False)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(self.hidden1,
                            n, bias=False)
        self.softmax = nn.Softmax(dim=2)

    # x is a (batch, n^2, n) tensor
    def forward(self, x):
        n = self.n
        bts = x.shape[0]
        c = self.constraint_mask
        min_empty = (x.sum(dim=2) == 0).sum(dim=1).max()
        x_fill = x.clone()
        x_pred = x.clone()
        for a in range(min_empty):
            # score empty numbers
            constraints = (x_fill.view(bts, 1, 1, n * n, n) * c).sum(dim=3)
            # empty cells
            empty_mask = (x_fill.sum(dim=2) == 0)
            empty_mask_e = empty_mask.view(bts, n * n, 1)\
                                     .expand(bts, n * n, n)
            f = constraints.reshape(bts, n * n, 3 * n).float()
            s = torch.zeros_like(x_fill)
            y = torch.zeros_like(x_fill)
            y_ = self.l2(self.a1(self.l1(f[empty_mask])))
            y.masked_scatter_(empty_mask_e, y_)
            s_ = self.softmax(y)[empty_mask]
            s.masked_scatter_(empty_mask_e, s_)
            x_pred[empty_mask] = s[empty_mask]
            # find most probable guess
            score, score_pos = s.max(dim=2)
            num, mmax = score.max(dim=1)
            # fill it in
            nz = empty_mask.sum(dim=1).nonzero().view(-1)
            mmax_ = mmax[nz]
            ones = torch.ones(nz.shape[0])
            x_fill.index_put_((nz, mmax_, score_pos[nz, mmax_]), ones)
        return x_pred, x_fill

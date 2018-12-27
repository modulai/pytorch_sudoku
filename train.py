import dataset as d
import model as m
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

batch_size = 100

train_set, test_set = d.load_dataset()

constraint_mask = d.create_constraint_mask()


dataloader_ = data.DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True)

dataloader_val_ = data.DataLoader(test_set,
                                  batch_size=batch_size,
                                  shuffle=True)

# Use sum as reducer as sequences are padded with zeros and we
# don't mask in the loss function. Think about if this can be improved

loss = nn.MSELoss()

sudoku_solver = SudokuSolver(constraint_mask)

optimizer = optim.Adam(sudoku_solver.parameters(),
                       lr=0.01,
                       weight_decay=0.000)


def loss_func(pred, sol):
    return -torch.log((pred * sol).sum())


epochs = 20
loss_train = []
loss_val = []
for e in range(epochs):

    for i_batch, ts_ in enumerate(dataloader_):
        sudoku_solver.train()
        optimizer.zero_grad()
        pred, mat = sudoku_solver(ts_[0])
        ls = loss(pred, ts_[1])
        ls.backward()
        optimizer.step()
        print("Epoch " + str(e) + " batch " + str(i_batch)
              + ": " + str(ls.item()))
        sudoku_solver.eval()
        with torch.no_grad():
            n = 100
            rows = torch.randperm(test_set.tensors[0].shape[0])[:n]
            test_pred, test_fill = sudoku_solver(test_set.tensors[0][rows])
            errors = test_fill.max(dim=2)[1] != test_set.tensors[1][rows].max(dim=2)[1]
            print("Cells in error: " + str(errors.sum().item()))

import torch.utils.data as data
import torch
import pandas as pd


def create_sudoku_tensors(df, train_split=0.5):
    s = df.shape[0]

    def one_hot_encode(s):
        zeros = torch.zeros((1, 81, 9), dtype=torch.float)
        for a in range(81):
            zeros[0, a, int(s[a]) - 1] = 1 if int(s[a]) > 0 else 0
        return zeros

    quizzes_t = df.quizzes.apply(one_hot_encode)
    solutions_t = df.solutions.apply(one_hot_encode)
    quizzes_t = torch.cat(quizzes_t.values.tolist())
    solutions_t = torch.cat(solutions_t.values.tolist())
    randperm = torch.randperm(s)
    train = randperm[:int(train_split * s)]
    test = randperm[int(train_split * s):]

    return data.TensorDataset(quizzes_t[train], solutions_t[train]),\
        data.TensorDataset(quizzes_t[test], solutions_t[test])


def create_constraint_mask():
    constraint_mask = torch.zeros((81, 3, 81), dtype=torch.float)
    # row constraints
    for a in range(81):
        r = 9 * (a // 9)
        for b in range(9):
            constraint_mask[a, 0, r + b] = 1

    # column constraints
    for a in range(81):
        c = a % 9
        for b in range(9):
            constraint_mask[a, 1, c + 9 * b] = 1

    # box constraints
    for a in range(81):
        r = a // 9
        c = a % 9
        br = 3 * 9 * (r // 3)
        bc = 3 * (c // 3)
        for b in range(9):
            r = b % 3
            c = 9 * (b // 3)
            constraint_mask[a, 2, br + bc + r + c] = 1

    return constraint_mask


def load_dataset(subsample=10000):
    dataset = pd.read_csv("sudoku.csv", sep=',')
    my_sample = dataset.sample(subsample)
    train_set, test_set = create_sudoku_tensors(my_sample)
    return train_set, test_set

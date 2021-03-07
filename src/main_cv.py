import torch
import torch.nn as nn
from prepareData import prepare_data
from gat_model import Net
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_curve
import numpy as np
import tensorly as tl
import warnings

dim = 128

warnings.filterwarnings("ignore")
tl.set_backend('pytorch')


class Config(object):
    def __init__(self):
        self.data_path = '../data'
        self.epoch = 300
        self.lr = 0.001
        self.save_path = '../output'


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, one_index, zero_index, target, input):
        loss = nn.MSELoss(reduction='none')
        loss_sum = loss(input, target)
        return loss_sum[one_index].sum()+loss_sum[zero_index].sum()


class Params(object):
    def __init__(self):
        self.m = 325
        self.d = 283
        self.fm = 128
        self.fd = 128
        self.hid = 128
        self.out = 128
        self.h = 3
        self.drop = 0.4


opt = Config()


def main():

    five_cv_y_true, five_cv_y_pred = [], []

    params = Params()

    for file_i in range(1, 10):
        dataset, know_one_index, know_one_tensor, know_zero_index, know_zero_tensor = prepare_data(15484, file_i)

        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        fold, i = 0, 0
        for train_index, test_index in kf.split(know_one_index):
            fold += 1
            model = Net(params).cuda()
            loss_func = Myloss().cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

            # store train data index, test data index of tensor   eg. tensor([113,  23,  34])
            one_index_train, one_index_test = know_one_tensor[train_index], know_one_tensor[test_index]
            zero_index_train, zero_index_test = know_zero_tensor[train_index], know_zero_tensor[test_index]

            edge_one_index_test = one_index_test.cuda().t().tolist()
            edge_zero_index_test = zero_index_test.cuda().t().tolist()

            y_true = dataset['mmd_true'][edge_one_index_test].numpy().tolist()
            y_true = y_true + dataset['mmd_true'][edge_zero_index_test].numpy().tolist()

            edge_one_index_train = [[], [], []]
            for idx in one_index_train:
                edge_one_index_train[0].append(idx[0])
                edge_one_index_train[1].append(idx[1])
                edge_one_index_train[2].append(idx[2])

            # set value 1 in test data to 0
            dataset['mmd_p'][edge_one_index_test] = 0

            '''
                zero index in train and test set
            '''
            edge_zero_index = [[], [], []]
            for idx in zero_index_train:
                edge_zero_index[0].append(idx[0])
                edge_zero_index[1].append(idx[1])
                edge_zero_index[2].append(idx[2])
            for idx in one_index_test:
                edge_zero_index[0].append(idx[0])
                edge_zero_index[1].append(idx[1])
                edge_zero_index[2].append(idx[2])
            for idx in zero_index_test:
                edge_zero_index[0].append(idx[0])
                edge_zero_index[1].append(idx[1])
                edge_zero_index[2].append(idx[2])

            # train phase
            for t in range(1, opt.epoch+1):
                p, b, d = model(dataset)
                X_ = tl.fold((p.mm((tl.tenalg.khatri_rao([b, d])).t())), 0, (325, 283, 325))
                loss = loss_func(edge_one_index_train, edge_zero_index, dataset['mmd_p'].cuda(), X_)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # train over

            y_pred = X_[edge_one_index_test].cpu().detach().numpy().tolist()
            y_pred = y_pred + X_[edge_zero_index_test].cpu().detach().numpy().tolist()

            fpr, tpr, threshold = roc_curve(y_true, y_pred)
            tprs.append(np.interp(mean_fpr, fpr, tpr))

            dataset['mmd_p'][edge_one_index_test] = 1

            five_cv_y_true.append(y_true)
            five_cv_y_pred.append(y_pred)

        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        print('mean_auc {}'.format(mean_auc))


if __name__ == "__main__":
    main()




























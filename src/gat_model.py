import torch
import tensorly as tl
import torch.nn as nn
from torch_geometric.nn import conv

dim = 128
headnum = 3


class Net(torch.nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()

        self.m = params.m
        self.d = params.d
        self.fm = params.fm
        self.fd = params.fd
        self.hid = params.hid
        self.out = params.out
        self.drop = params.drop
        self.h = params.h

        self.gat_p1 = conv.GATConv(self.fm, self.hid, dropout=self.drop, heads=self.h)
        self.gat_b1 = conv.GATConv(self.fd, self.hid, dropout=self.drop, heads=self.h)
        self.gat_d1 = conv.GATConv(self.fm, self.hid, dropout=self.drop, heads=self.h)

        self.m1_l1 = nn.Linear(self.hid*self.h, self.hid)
        self.m1_l2 = nn.Linear(self.hid, self.hid)
        self.m1_l3 = nn.Linear(self.hid, self.out)

        self.m2_l1 = nn.Linear(self.hid*self.h, self.hid)
        self.m2_l2 = nn.Linear(self.hid, self.hid)
        self.m2_l3 = nn.Linear(self.hid, self.out)

        self.d_l1 = nn.Linear(self.hid*self.h, self.hid)
        self.d_l2 = nn.Linear(self.hid, self.hid)
        self.d_l3 = nn.Linear(self.hid, self.out)

    def forward(self, dataset):
        torch.manual_seed(1)
        # P, B, D random initialized here
        P = tl.tensor(torch.randn(self.m, self.fm), requires_grad=True)
        B = tl.tensor(torch.randn(self.d, self.fd), requires_grad=True)
        D = tl.tensor(torch.randn(self.m, self.fm), requires_grad=True)

        mm_func_edge_index = dataset['mm_func']['edge_index']
        mm_seq_edge_index = dataset['mm_seq']['edge_index']
        dd_sem_edge_index = dataset['dd_sem']['edge_index']

        P_1 = torch.relu(self.gat_p1(P.cuda(), mm_func_edge_index.cuda()))
        B_1 = torch.relu(self.gat_b1(B.cuda(), dd_sem_edge_index.cuda()))
        D_1 = torch.relu(self.gat_d1(D.cuda(), mm_seq_edge_index.cuda()))

        P_l1 = torch.relu(self.m1_l1(P_1))
        P_l2 = torch.relu(self.m1_l2(P_l1))
        P_l3 = self.m1_l3(P_l2)

        B_l1 = torch.relu(self.m2_l1(B_1))
        B_l2 = torch.relu(self.m2_l2(B_l1))
        B_l3 = self.m2_l3(B_l2)

        D_l1 = torch.relu(self.d_l1(D_1))
        D_l2 = torch.relu(self.d_l2(D_l1))
        D_l3 = self.d_l3(D_l2)

        return P_l3, B_l3, D_l3










import numpy as np
import torch
from torch.nn.init import xavier_normal_

import tensornetwork as tn 
import numpy as np
import torch
import time

class TTStack(torch.nn.Module):
    '''
    We index into the core set using the first row of input_facs; that said, the core set is general
    enough to permit conversion to the tensor train format
    '''
    def __init__(self, stackSize, input_fac, ranks):
        super(TTStack, self).__init__()
        assert(len(ranks) == len(input_fac[0]) - 1)
        ranks.insert(0, 1)
        ranks.append(1)

        self.input_fac = input_fac 
        self.ranks = ranks
        
        self.factorization_dim = input_fac.shape[1]

        self.nRow = np.prod(self.input_fac[0])
        self.nCol = np.prod(self.input_fac[1])
        
        self.cores = torch.nn.ParameterList()
        self.coreLabels = []


        for i in range(self.factorization_dim):
            coreSize = list(input_fac[:, i])
            if i > 0:
                coreSize.insert(0, ranks[i])
            if i < self.factorization_dim - 1:
                coreSize.append(ranks[i+1])

            coreSize.insert(0, stackSize)

            self.cores.append(torch.nn.Parameter(torch.rand(coreSize) * 0.1, requires_grad=True))
            if i == 0:
                self.coreLabels.append(("-batch", "-left{}".format(i), "-right{}".format(i), "rank1"))
            elif i == self.factorization_dim - 1:
                self.coreLabels.append(("-batch", "rank{}".format(i), "-left{}".format(i), "-right{}".format(i)))
            else:
                self.coreLabels.append(("-batch", "rank{}".format(i), "-left{}".format(i), "-right{}".format(i), "rank{}".format(i+1)))

        self.output_order = ["-batch"] + (["-left{}".format(i) for i in range(self.factorization_dim)] + 
            ["-right{}".format(i) for i in range(self.factorization_dim)])
        

    def gather_n(self, indices):
        batch = [] 
        for i in range(self.factorization_dim):
            batch.append(self.cores[i][indices]) 

        return tn.ncon(batch, self.coreLabels, out_order=self.output_order, backend='pytorch').reshape(len(indices), self.nRow, self.nCol)


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()

        d1 = 256 
        self.entity_count= len(d.entities)

        self.E = torch.nn.Embedding(len(d.entities), d1)
        self.core = TTStack(len(d.entities), np.array([[4, 4, 4, 4], [4, 4, 4, 4]]), [8, 8, 8]) 

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        

    def init(self):
        xavier_normal_(self.E.weight.data)
        # xavier_normal_(self.R.weight.data)
        pass 

    def forward(self, e1_idx, r_idx):
        # Hopefully, should never have to materialize the matrix 
        # e_full = self.E.tt_matrix.full()
        # e1 = e_full[e1_idx]

        e_full = self.E.weight
        e1 = self.E(e1_idx)

        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        W_mat = self.core.gather_n(r_idx) 
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, e_full.transpose(1,0))
        pred = torch.sigmoid(x)[:, 0:self.entity_count]
        return pred

if __name__ == '__main__':
    test = TTStack(500, np.array([[4, 4, 4, 4], [4, 4, 4, 4]]), ranks=[8, 8, 8])
    print(test.cores)
    print(test.coreLabels)
    print(test.output_order)
    pytorch_total_params = sum(p.numel() for p in test.parameters() if p.requires_grad)
    print(pytorch_total_params)

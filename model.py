import numpy as np
import torch
from torch.nn.init import xavier_normal_

import tensornetwork as tn
import numpy as np
import torch
import time
import t3nsor as t3

def unmap(indices, factorization):
        all_factored = []
        for idx in indices:
            rem = idx
            factored = []
            for factor in factorization:
                factored.append(rem % factor)
                rem //= factor
            all_factored.append(factored)
            
        return np.int_(all_factored)

class Indexable_Core_Set(torch.nn.Module):
    '''
    We index into the core set using the first row of input_facs; that said, the core set is general
    enough to permit conversion to the tensor train format
    '''
    def __init__(self, input_facs, ranks):
        super(Indexable_Core_Set, self).__init__()
        assert(len(ranks) == len(input_facs[0]) - 1)
        ranks.insert(0, 1)
        ranks.append(1)

        self.input_facs = input_facs
        self.ranks = ranks
        
        self.factorization_dim = input_facs.shape[1]
        self.real_dim = input_facs.shape[0]
        
        self.cores = torch.nn.ParameterList()

        for i in range(self.factorization_dim):
            coreSize = list(input_facs[:, i])
            coreSize.insert(0, ranks[i])
            coreSize.append(ranks[i+1])

            self.cores.append(torch.nn.Parameter(torch.rand(coreSize) * 0.1, requires_grad=True))


    def createIndexBatch(self, indices):
        perm = list(range(self.real_dim + 2))
        perm[0] = 1
        perm[1] = 0
        core_batches = []
        exploded_indices = unmap(indices, self.input_facs[0, :])
        for i in range(self.factorization_dim):
            core_batches.append(self.cores[i][:, exploded_indices[:, i]].transpose(1, 0))

        return core_batches 

class RESCAL_Core(torch.nn.Module):
    def __init__(self, rel_fac, emb_fac):
        super(RESCAL_Core, self).__init__()
        self.core_set = Indexable_Core_Set(np.array([rel_fac, emb_fac, emb_fac]), [25, 25])
        self.emb_fac = emb_fac
        self.rel_fac = rel_fac

    def getStack(self, batch_indices):
        indexBatch = self.core_set.createIndexBatch(batch_indices)

        res = tn.ncon(indexBatch, [
            ('-batch', '-rank0', '-left1', '-right1', 'rank1'),
            ('-batch', 'rank1', '-left2', '-right2', 'rank2'),
            ('-batch', 'rank2', '-left3', '-right3', '-rank3')
            ], out_order=['-batch', '-rank0', '-left1', '-left2', '-left3', '-right1', '-right2', '-right3', '-rank3'],
            backend='pytorch').squeeze()
        return res.reshape(len(batch_indices), int(np.prod(self.emb_fac)), int(np.prod(self.emb_fac)))

class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()
        
        self.entity_count= len(d.entities)

        self.E = t3.TTEmbedding(
            voc_size=len(d.entities),
            emb_size=d1,
            auto_shapes=True,       # Should figure out what these shapes are...
            auto_shape_mode='mixed',
            tt_rank=128,
            d=4
        )
        # self.E = torch.nn.Embedding(len(d.entities), d1)
        self.R = torch.nn.Embedding(len(d.relations), d2)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)), 
                                   dtype=torch.float, requires_grad=True))

        # self.core = RESCAL_Core([5, 10, 10], [8, 5, 5])        

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        

    def init(self):
        # xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)
        pass 

    def forward(self, e1_idx, r_idx):
        # Hopefully, should never have to materialize the matrix 
        e_full = self.E.tt_matrix.full()
        e1 = e_full[e1_idx]

        # e_full = self.E.weight
        # e1 = self.E(e1_idx)

        x = self.bn0(e1)
        # x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        # W_mat = self.core.getStack(r_idx)

        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat) 
        x = x.view(-1, e1.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, e_full.transpose(1,0))
        pred = torch.sigmoid(x)[:, 0:self.entity_count]
        return pred


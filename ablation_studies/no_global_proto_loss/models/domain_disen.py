import torch
import torch.nn as nn
import torch.nn.functional as F

class Domain_Disen(nn.Module):
    def __init__(self,**params):
        super(Domain_Disen, self).__init__()
        self.orig_emb_size = params['orig_emb_size']
        self.disen_emb_size = params['disen_emb_size']
        self.device = params['device']

        self.domain_common_layer = nn.Linear(self.orig_emb_size, self.disen_emb_size).to(self.device)
        self.domain_common_norm = nn.BatchNorm1d(self.disen_emb_size).to(self.device)

        self.domain_specific_layer = nn.Linear(self.orig_emb_size, self.disen_emb_size).to(self.device)
        self.domain_specific_norm = nn.BatchNorm1d(self.disen_emb_size).to(self.device)

    def forward(self,orig_emb):
        disen_common_emb = self.domain_common_norm(F.relu(self.domain_common_layer(orig_emb)))
        # disen_common_emb = F.relu(self.domain_specific_layer(orig_emb))
        disen_specific_emb = self.domain_specific_norm(F.relu(self.domain_specific_layer(orig_emb)))
        # disen_specific_emb = F.relu(self.domain_specific_layer(orig_emb))
        return disen_common_emb,disen_specific_emb


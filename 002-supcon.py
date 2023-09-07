from network_dataset import Task3Data, Task3DataNetwork, Task3DataSupCon
import torch
import sys
import yaml
import numpy as np
import os
from sklearn import preprocessing
from torch.utils.data.dataloader import DataLoader
from modeling_pretrain import BNTF
from modeling_pretrain import  MLPHead
import random
import torch.nn as nn
import torch.nn.functional as F

shuffle_seed = 42
batch_size = 64
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_dataset = Task3DataSupCon(shuffle_seed,is_train=True,is_test=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          num_workers=0, drop_last=False, shuffle=True)


config = yaml.load(open("./configs/large_run3.yaml", "r"), Loader=yaml.FullLoader)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
feature_size = config['network']['feature_dim']
depth = config['network']['depth']
heads = config['network']['heads']
dim_feedforward = config['network']['dim_feedforward']

class SupCon(nn.Module):
    def __init__(self,feat_dim):
        super(SupCon, self).__init__()
        self.encoder = BNTF(feature_size,depth,heads,dim_feedforward)
        self.head = nn.Sequential(
                nn.Linear(feature_size, feature_size),
                nn.ReLU(inplace=True),
                nn.Linear(feature_size, feat_dim)
            )
    def forward(self,x):
        x = self.encoder(x)
        feat = F.normalize(self.head(x), dim=1)
        return feat

model = SupCon(1024)

load_params = torch.load(os.path.join(config['saving']['checkpoint_dir'],'best_model.pth'),
                         map_location='cpu')['online_network_state_dict']

model.encoder.load_state_dict(load_params)
print("Parameters successfully loaded.")

model = model.to(device)

def evaluate(x_data,y_data,model):
    model.eval()

    logits = model(x_data)
    predictions = torch.argmax(logits, dim=1)
    
    total = y_data.size(0)
    correct = (predictions == y_data).sum().item()
        
    acc = 100 * correct / total
    return acc


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

best_val = 0.
criterion = SupConLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-5)
for epoch in range(1500):
    epoch_loss = 0.
    epoch_count = 0.
    model.train()
    for i, (x1,x2, y) in enumerate(train_loader):
        x1 = x1.to(device).float()
        x2 = x2.to(device).float()
        y = y.to(device).long()
        N = x1.size(0)
        optimizer.zero_grad()        

        inputs = torch.cat([x1,x2],dim=0)

        logits = model(inputs)
        features = torch.cat([logits[:N][:,None], logits[N:][:,None]],dim = 1)
        #print(features.shape, y.shape)
        loss = criterion(features, y[:,1])
        
        loss.backward()
        optimizer.step()

        epoch_loss += loss * N
        epoch_count += N
        # train_acc= (predictions == y[:,1]).sum().item() * 100 / y_train.size(0)
    epoch_loss /= epoch_count
    print(f"epoch: {epoch} Training loss: {epoch_loss:.4f}")

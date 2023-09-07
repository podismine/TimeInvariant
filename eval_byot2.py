from network_dataset import Task3Data
import torch
import sys
import yaml
import numpy as np
import os
from sklearn import preprocessing
from torch.utils.data.dataloader import DataLoader
from modeling_pretrain import BNTF, FT
from modeling_pretrain import  MLPHead
import random
import torch.nn as nn

config = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)

shuffle_seed = 1
batch_size = 16
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_dataset = Task3Data(shuffle_seed,is_train=True,is_test=False)
val_dataset = Task3Data(shuffle_seed,is_train=False,is_test=False)
test_dataset = Task3Data(shuffle_seed,is_train=False,is_test=True)


train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          num_workers=0, drop_last=False, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          num_workers=0, drop_last=False, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=1,
                          num_workers=0, drop_last=False, shuffle=False)

device='cuda'
feature_size = 1024
new_encoder = FT(feature_size).to(device)

load_params = torch.load(os.path.join('output_checkpoint_alldata_d12_h8_1024_t15/best_model.pth'),
                         map_location=torch.device(torch.device(device)))

if 'online_network_state_dict' in load_params:
    new_encoder.encoder.load_state_dict(load_params['online_network_state_dict'])
    print("Parameters successfully loaded.")

#encoder = torch.nn.Sequential(*list(encoder.children())[:-1])    
new_encoder = new_encoder.to(device)

def continus_mixup_data(*xs, y=None, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = y.size()[0]
    index = torch.randperm(batch_size).to(device)
    new_xs = [lam * x + (1 - lam) * x[index, :] for x in xs]
    y = lam * y + (1-lam) * y[index]
    return *new_xs, y

for name, p in new_encoder.named_parameters():
    if 'BNTF' in name:
        #pass
        p.requires_grad_(False)
        pass

optimizer = torch.optim.AdamW([f for f in new_encoder.parameters() if f.requires_grad], lr=1e-4,weight_decay=1e-5)
criterion = torch.nn.CrossEntropyLoss()
eval_every_n_epochs = 10

@torch.no_grad()
def evaluate(loader,model):
    model.eval()
    test_acc = 0.
    test_c = 0.
    for x,y in loader:
        x=x.to(device).float()
        y=y.to(device).long()
        logits = model(x)
        predictions = torch.argmax(logits, dim=1)
    
        total = y.size(0)
        if len(y.shape)>1:
            correct = (predictions == y[:,1]).sum().item() * 100
        else:
            correct = (predictions == y).sum().item() * 100

        test_acc += correct
        test_c += len(x)
    test_acc /= test_c
    return test_acc

@torch.no_grad()
def ea_evaluate(loader,model):
    model.eval()
    test_acc = 0.
    test_c = 0.
    for x,y in loader:
        batch = x.size(0)
        batch_pred = []
        all_pred = []
        for mini_batch in range(batch):
            mini_x = x[mini_batch].to(device).float()
            logits = model(mini_x)
            batch_pred.append(logits.exp().mean(0))
        batch_pred = torch.stack(batch_pred,0)
        y=y.to(device).long()
        predictions = torch.argmax(batch_pred, dim=1)
    
        total = y.size(0)
        if len(y.shape)>1:
            correct = (predictions == y[:,1]).sum().item() * 100
        else:
            correct = (predictions == y).sum().item() * 100

        test_acc += correct
        test_c += len(x)
    test_acc /= test_c
    return test_acc

best_val = 0.
for epoch in range(30):

    new_encoder.train()
    train_acc = 0.
    train_c = 0.
    for x,y in train_loader:
        y=y.to(device).float()
        x,y_mix=continus_mixup_data(x,y=y)
        x=x.to(device).float()
        y_mix=y_mix.to(device).float()
        optimizer.zero_grad()        

        logits = new_encoder(x)
        predictions = torch.argmax(logits, dim=1)
        
        loss = criterion(logits, y_mix)
        
        loss.backward()
        optimizer.step()
        train_acc += (predictions == y[:,1]).sum().item() * 100
        train_c += len(x)
    train_acc /= train_c

    val_acc = evaluate(val_loader, new_encoder)
    #val_acc = evaluate(x_val, y_val[:,1], logreg)

    if val_acc >= best_val:
        best_val = val_acc

        #test_acc = evaluate(x_test, y_test[:,1], logreg)
        test_acc = evaluate(test_loader, new_encoder)
        #test_acc = ea_evaluate(test_loader, new_encoder)

        print(f"epoch: {epoch} Training accuracy: {train_acc:.2f} Validation accuracy: {val_acc:.2f} Testing accuracy: {test_acc:.2f}")
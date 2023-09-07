import yaml
import torch
import numpy as np
import os
from network_dataset import Task3Data,Task3DataNetwork
from modeling_pretrain import MLPHead, BNTF, FT
from torch.utils.data.dataloader import DataLoader
from byol_trainer import BYOLTrainer
import torch
import torch.nn as nn
import random 

shuffle_seed = 0
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

# train_dataset = Task3DataNetwork(shuffle_seed,is_train=True,is_test=False)
# val_dataset = Task3DataNetwork(shuffle_seed,is_train=False,is_test=False)
# test_dataset = Task3DataNetwork(shuffle_seed,is_train=False,is_test=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          num_workers=0, drop_last=False, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          num_workers=0, drop_last=False, shuffle=False)

test_loader = DataLoader(test_dataset, batch_size=batch_size,
                          num_workers=0, drop_last=False, shuffle=False)

#config_d24_h20_1024_t15_abide
#config = yaml.load(open("./configs/config_d4_h4_1024_t15_abide.yaml", "r"), Loader=yaml.FullLoader)
config = yaml.load(open("./configs/small_abide.yaml", "r"), Loader=yaml.FullLoader)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
feature_size = config['network']['feature_dim']
depth = config['network']['depth']
heads = config['network']['heads']
dim_feedforward = config['network']['dim_feedforward']

encoder = FT(feature_size,depth,heads,dim_feedforward)

new_params = dict()

#load_params = torch.load(os.path.join('output_checkpoint_abide_d4_h4_1024_t15/best_model.pth'),
#                         map_location='cpu')['online_network_state_dict']

# for key,val in encoder.state_dict().items():
#     if 'g2' in key:
#         new_params[key] = val
#     else:
#         new_params[key] = load_params[key.replace("encoder.","")]


#encoder.load_state_dict(new_params,strict=True)

print("loading finished.")


encoder.cuda().train()

for name, p in encoder.named_parameters():
    if 'encoder' in name:
        #pass
        #p.requires_grad_(False)
        pass

optimizer = torch.optim.AdamW([f for f in encoder.parameters() if f.requires_grad], lr=1e-4,weight_decay=1e-5)
criterion = torch.nn.CrossEntropyLoss()
eval_every_n_epochs = 10

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
for epoch in range(250):

    encoder.train()
    train_acc = 0.
    train_c = 0.
    for step, (x,y) in enumerate(train_loader):
        y=y.to(device).float()
        x=x.to(device).float()
        x,y_mix=continus_mixup_data(x,y=y)
        optimizer.zero_grad()        
        logits = encoder(x)
        #print(x.shape, y_mix.shape,logits.shape)
        predictions = torch.argmax(logits, dim=1)
        
        loss = criterion(logits, y_mix)
        acc = (predictions == y[:,1]).sum().item() * 100 / len(y)
        loss.backward()
        optimizer.step()
        train_acc += acc * len(x)
        train_c += len(x)
        #print(f"{step}/{len(train_loader)} training loss: {loss:.5f} acc: {acc}")
    train_acc /= train_c

    val_acc = evaluate(val_loader, encoder)
    #val_acc = evaluate(x_val, y_val[:,1], logreg)

    if val_acc >= best_val:
        best_val = val_acc

        #test_acc = evaluate(x_test, y_test[:,1], logreg)
        test_acc = evaluate(test_loader, encoder)
        #test_acc = ea_evaluate(test_loader, new_encoder)

        print(f"epoch: {epoch} Training accuracy: {train_acc:.2f} Validation accuracy: {val_acc:.2f} Testing accuracy: {test_acc:.2f}")
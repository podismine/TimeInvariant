from network_dataset import Task3Data, Task3DataNetwork
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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed",'-s', type=int,default = 42)
args = parser.parse_args()

shuffle_seed = int(args.seed) #42
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

config = yaml.load(open("./configs/large_run3.yaml", "r"), Loader=yaml.FullLoader)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
feature_size = config['network']['feature_dim']
depth = config['network']['depth']
heads = config['network']['heads']
dim_feedforward = config['network']['dim_feedforward']

encoder = BNTF(feature_size,depth,heads,dim_feedforward).to(device)

load_params = torch.load(os.path.join(config['saving']['checkpoint_dir'],'best_model.pth'),
                         map_location='cpu')['online_network_state_dict']

encoder.load_state_dict(load_params)
print("Parameters successfully loaded.")

#encoder = torch.nn.Sequential(*list(encoder.children())[:-1])    
encoder = encoder.to(device)

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            torch.nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            torch.nn.Linear(32, output_dim),
            )
    def forward(self, x):
        return self.linear2(x)
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

#logreg = LogisticRegression(feature_size, 2)
logreg = LogisticRegression(100*8, 2)
logreg = logreg.to(device)

def get_features_from_encoder(encoder, loader,times = 1):
    
    x_train = []
    y_train = []

    # get the features from the pre-trained model
    encoder.eval()
    for _ in range(times):
        for i, (x, y) in enumerate(loader):
            x = x.to(device).float()
            y = y.to(device).long()
            with torch.no_grad():
                #feature_vector = encoder(x)
                bz, _, _, = x.shape

                for atten in encoder.attention_list:
                    x = atten(x)

                node_feature = encoder.dim_reduction(x)
                feature_vector = node_feature.reshape((bz, -1))
                x_train.extend(feature_vector.detach())
                y_train.extend(y.detach())

            
    x_train = torch.stack(x_train)
    y_train = torch.stack(y_train)
    return x_train, y_train

encoder.eval()
x_train, y_train = get_features_from_encoder(encoder, train_loader,10)
x_val, y_val = get_features_from_encoder(encoder, val_loader,1)
x_test, y_test = get_features_from_encoder(encoder, test_loader,1)


x_train = x_train.detach().cpu().numpy()
x_val = x_val.detach().cpu().numpy()
x_test = x_test.detach().cpu().numpy()


scaler = preprocessing.StandardScaler()
scaler.fit(x_train)


x_train = scaler.transform(x_train).astype(np.float32)
x_val = scaler.transform(x_val).astype(np.float32)
x_test = scaler.transform(x_test).astype(np.float32)

# from sklearn.svm import SVC 
# from sklearn.metrics import accuracy_score
# clf = SVC()
# clf.fit(x_train, y_train.detach().cpu().numpy())
# pred = clf.predict(x_test)
# print(accuracy_score(pred, y_test.detach().cpu().numpy()))
# exit()

x_train = torch.from_numpy(x_train).cuda().float()
x_val = torch.from_numpy(x_val).cuda().float()
x_test = torch.from_numpy(x_test).cuda().float()
    
print("Training data shape:", x_train.shape, y_train.shape)
print("Testing data shape:", x_test.shape, y_test.shape)

optimizer = torch.optim.Adam(logreg.parameters(), lr=1e-3,weight_decay=1e-5)
criterion = torch.nn.CrossEntropyLoss()
eval_every_n_epochs = 10
logreg.cuda().train()

x_train = x_train.cuda()
y_train = y_train.cuda()

x_val = x_val.cuda()
y_val = y_val.cuda()

x_test = x_test.cuda()
y_test = y_test.cuda()


def evaluate(x_data,y_data,model):
    model.eval()

    logits = logreg(x_data)
    predictions = torch.argmax(logits, dim=1)
    
    total = y_data.size(0)
    correct = (predictions == y_data).sum().item()
        
    acc = 100 * correct / total
    return acc

best_val = 0.
for epoch in range(1500):

    logreg.train()
    # zero the parameter gradients
    optimizer.zero_grad()        
    #x_train,mix_y_train=continus_mixup_data(x_train,y=y_train.float())

    logits = logreg(x_train)
    predictions = torch.argmax(logits, dim=1)
        
    loss = criterion(logits, y_train.float())
        
    loss.backward()
    optimizer.step()
    #train_acc= (predictions == y_train).sum().item() * 100 / y_train.size(0)
    train_acc= (predictions == y_train[:,1]).sum().item() * 100 / y_train.size(0)

    #val_acc = evaluate(x_val, y_val, logreg)
    val_acc = evaluate(x_val, y_val[:,1], logreg)

    if val_acc >= best_val:
        best_val = val_acc

        test_acc = evaluate(x_test, y_test[:,1], logreg)
        #test_acc = evaluate(x_test, y_test, logreg)

        print(f"epoch: {epoch} Training accuracy: {train_acc:.2f} Validation accuracy: {val_acc:.2f} Testing accuracy: {test_acc:.2f}")
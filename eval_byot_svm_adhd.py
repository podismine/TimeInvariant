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

config_path = "./configs/large_run1_train.yaml"
config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
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
# x_test, y_test = get_features_from_encoder(encoder, train_loader,5)
# x_val, y_val = get_features_from_encoder(encoder, val_loader,1)
# x_train, y_train = get_features_from_encoder(encoder, test_loader,1)

x_train, y_train = get_features_from_encoder(encoder, train_loader,1)
x_val, y_val = get_features_from_encoder(encoder, val_loader,1)
x_test, y_test = get_features_from_encoder(encoder, test_loader,1)

# x_train, y_train = x_train[:int(0.5 * len(x_train))], y_train[:int(0.5 * len(x_train))]
# print(x_train.shape, y_train.shape)

x_train = x_train.detach().cpu().numpy()
x_val = x_val.detach().cpu().numpy()
x_test = x_test.detach().cpu().numpy()


scaler = preprocessing.StandardScaler()
scaler.fit(x_train)


x_train = scaler.transform(x_train).astype(np.float32)
x_val = scaler.transform(x_val).astype(np.float32)
x_test = scaler.transform(x_test).astype(np.float32)

from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
clf = SVC()
clf.fit(x_train, y_train.detach().cpu().numpy()[:,1])
pred_val = clf.predict(x_val)
pred_test = clf.predict(x_test)
#print(accuracy_score(pred_val, y_val.detach().cpu().numpy()[:,1]))
acc = accuracy_score(pred_test, y_test.detach().cpu().numpy()[:,1])
cm = confusion_matrix(pred_test, y_test.detach().cpu().numpy()[:,1])
auc = roc_auc_score(pred_test, y_test.detach().cpu().numpy()[:,1])
sen = round(cm[1, 1] / float(cm[1, 1]+cm[1, 0]),4)
spe = round(cm[0, 0] / float(cm[0, 0]+cm[0, 1]),4)

res_string = f"acc: {acc:.4f} auc: {auc:.4f} sen: {sen:.4f} spe: {spe:.4f}"
print(res_string)
res_path = config_path.replace("./configs/", "")
with open(f"res/{res_path}.txt", 'a') as f:
    f.write(f"seed:[{shuffle_seed}] {res_string} \n")
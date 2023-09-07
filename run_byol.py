from network_dataset import Task1Data,Task3Data

import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from apex.parallel import DistributedDataParallel as DDP

import yaml

print(torch.__version__)
torch.manual_seed(0)

class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter(log_dir='byot_test')
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()
        best_train_loss = 99999.
        if not os.path.exists(model_checkpoints_folder):
            os.makedirs(model_checkpoints_folder)
        for epoch_counter in range(self.max_epochs):
            total_loss = 0.
            count = 0.
            for (batch_view_1, batch_view_2) in train_loader:

                batch_view_1 = batch_view_1.to(self.device).float()
                batch_view_2 = batch_view_2.to(self.device).float()
                loss = self.update(batch_view_1, batch_view_2)
                self.writer.add_scalar('loss', loss, global_step=niter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self._update_target_network_parameters()  # update the key encoder
                niter += 1
                total_loss += float(loss) * len(batch_view_1)
                count += len(batch_view_1)
            total_loss /= count
            print("End of epoch {}".format(epoch_counter), f' train loss: {total_loss:.4}')
            self.writer.add_scalar('epoch_loss', total_loss, global_step=epoch_counter)

            if total_loss <= best_train_loss:
                best_train_loss = total_loss
                self.save_model(os.path.join(model_checkpoints_folder, 'best_model.pth'))


        # save checkpoints
        
        self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()

    def save_model(self, PATH):
        
        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)


from modeling_pretrain import BNTF
from modeling_pretrain import  MLPHead
def main():
    config = yaml.load(open("./configs/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")


    train_dataset = Task1Data()
    #train_dataset = Task1Data(0,is_train=True,is_test=False)

    # online network
    feature_size = 1024
    online_network = BNTF(feature_size,4,4,1024).to(device)

    # predictor network
    predictor = MLPHead(feature_size,feature_size * 2,feature_size).to(device)

    # target encoder
    target_network = BNTF(feature_size,4,4,1024).to(device)

    optimizer = torch.optim.AdamW(list(online_network.parameters()) + list(predictor.parameters()),lr= 3e-4,weight_decay=1e-4)

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          **config['trainer'])

    trainer.train(train_dataset)


if __name__ == '__main__':
    main()
from pyarrow import Tensor
import torch
from torch import nn,optim
from egnn_clean import EGNN 
from divider import Dataset 
import numpy as np
import argparse
from sklearn.metrics import accuracy_score
# import datasetLoader


# PARSER ARGUMENTS
parser = argparse.ArgumentParser(description='DL course Project')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N',
                    help='experiment_name')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='AIDS/logs', metavar='N',
                    help='folder to output values') 
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='Learning Rate 1')
parser.add_argument('--nf', type=int, default=4, metavar='N',
                    help='Number of hidden features')
parser.add_argument('--n_layers', type=int, default=7, metavar='N',
                    help='Number of Layers')
parser.add_argument('--property', type=str, default='1', metavar='N',
                    help='label to predict: Active | Inactive')
parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                    help='number of workers for the dataloader')
parser.add_argument('--node_attr', type=int, default=0, metavar='N',
                    help='node_attr or not')
parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                    help='weight decay')
parser.add_argument('--hidden_nf', type=float, default=32, metavar='N',
                    help='Normalize coordinates Messages')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32

def main():
    print("\n###MAIN###\n")
    # Model and Stuffn
    model = EGNN(out_node_nf=1,in_node_nf=1, in_edge_nf=1, hidden_nf=32, device="cpu", n_layers=3,attention=False,normalize=False)
    # egnn = EGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_l1 = nn.L1Loss()
    
    # Generate Training and Testing datasets
    dataset = Dataset()
    train_dl, test_dl = dataset.generateData()

    train_losses,test_losses,train_accs,test_accs = [],[],[],[]

    for epoch in range(args.epochs):
        # train_preds,train_gts,test_preds,test_gts  = np.array([]),np.array([]),np.array([]),np.array([])
        train_loss, test_loss = 0,0

        model.train()
        for data in iter(train_dl):

            # Transforming tensors from data to numpy ndarrays beacuse the example did so  
            # edge_index = data.edge_index.view(len(data.edge_index[0]),2)

            h = torch.ones(len(data.x),1)#.view(len(data.x),1)
            print(f"h shape: {h.shape}")
            h, x = model(h, data.x, data.edge_index, data.edge_attr)

            # Compute Loss Value
            train_loss = loss_l1(h,data.y)
            # Backward Pass - update the Weights
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # save results for statistics
            # train_preds = np.concatenate((train_preds, h.flatten().numpy()))
            # train_gts = np.concatenate((train_gts, data.y.flatten().numpy()))

        model.eval()
        for data in test_dl:

            h = torch.ones([0]*len(data.x))
            h, x = model(h, data.x, data.edge_index, data.edge_attr)
            # Compute loss value
            test_loss = loss_l1(h, data.y)
            # Save results for statistics
            # test_preds = np.concatenate((test_preds, h.flatten().numpy()))
            # test_gts = np.concatenate((test_gts, y.flatten().numpy()))

        # Compute Stats 
        # if epoch % 1 == 0:
            # train_accs.append(accuracy_score(train_gts, train_preds))
            # test_accs.append(accuracy_score(test_gts, test_preds))
            # train_losses.append(train_loss.item())
            # test_losses.append(test_loss.item())



    

if __name__ == "__main__":
    main()

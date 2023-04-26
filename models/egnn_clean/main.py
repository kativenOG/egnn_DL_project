from egnn_clean import EGNN 
from torch import nn,optim
import numpy as np
import torch
import argparse
import datasetLoader
from sklearn.metrics import accuracy_score


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
parser.add_argument('--nf', type=int, default=128, metavar='N',
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
    model = EGNN(out_node_nf=1,in_node_nf=15, in_edge_nf=0, hidden_nf=args.nf, device=device, n_layers=args.n_layers,attention=False,normalize=False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    loss_l1 = nn.L1Loss()
    
    # Generate Training and Testing datasets
    gds = datasetLoader.generateDatasets() 
    train_dl, test_dl = gds.generate()
    print(train_dl)
    # train_iter,test_iter = iter(train_dl),iter(test_dl)

    train_losses,test_losses,train_accs,test_accs = [],[],[],[]

    for epoch in range(args.epochs):
        train_preds,train_gts,test_preds,test_gts  = np.array([]),np.array([]),np.array([]),np.array([])

        model.train()
        for data in iter(train_dl):
            h, x = model(h, x, data.edges, data.edge_attr)
            # compute loss value
            train_loss = loss_l1(h,data.y)
            # backward pass - update the weights
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            # save results for statistics
            train_preds = np.concatenate((train_preds, h.flatten().numpy()))
            train_gts = np.concatenate((train_gts, data.y.flatten().numpy()))

        model.eval()
        for data in test_dl:

            h, x = model(h, x, data.edges, data.edge_attr)
            # Compute loss value
            test_loss = loss_l1(h, data.y)
            # Save results for statistics
            test_preds = np.concatenate((test_preds, h.flatten().numpy()))
            test_gts = np.concatenate((test_gts, y.flatten().numpy()))

        # Compute Stats 
        if epoch % 1 == 0:
            train_accs.append(accuracy_score(train_gts, train_preds))
            test_accs.append(accuracy_score(test_gts, test_preds))
            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())



    

if __name__ == "__main__":
    main()


# # Dummy parameters
# batch_size = 8
# n_nodes = 4
# n_feat = 1
# x_dim = 3

# # Dummy variables h, x and fully connected edges
# h = torch.ones(batch_size *  n_nodes, n_feat)
# x = torch.ones(batch_size * n_nodes, x_dim)
# edges, edge_attr = get_edges_batch(n_nodes, batch_size)

# # Initialize EGNN
# egnn = EGNN(in_node_nf=n_feat, hidden_nf=32, out_node_nf=1, in_edge_nf=1)

# # Run EGNN
# h, x = egnn(h, x, edges, edge_attr)

# def train(epoch, loader, partition='train'):
    #     lr_scheduler.step()
    #     res = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    #     for i, data in enumerate(loader):
    #         if partition == 'train':
    #             model.train()
    #             optimizer.zero_grad()

    #         else:
    #             model.eval()

    #         batch_size, n_nodes, _ = data['positions'].size()
    #         atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, dtype)
    #         atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
    #         edge_mask = data['edge_mask'].to(device, dtype)
    #         one_hot = data['one_hot'].to(device, dtype)
    #         charges = data['charges'].to(device, dtype)
    #         nodes = qm9_utils.preprocess_input(one_hot, charges, args.charge_power, charge_scale, device)

    #         nodes = nodes.view(batch_size * n_nodes, -1)
    #         # nodes = torch.cat([one_hot, charges], dim=1)
    #         edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
    #         label = data[args.property].to(device, dtype)

    #         pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
    #                      n_nodes=n_nodes)

    #         if partition == 'train':
    #             loss = loss_l1(pred, (label - meann) / mad)
    #             loss.backward()
    #             optimizer.step()
    #         else:
    #             loss = loss_l1(mad * pred + meann, label)

    #         res['loss'] += loss.item() * batch_size
    #         res['counter'] += batch_size
    #         res['loss_arr'].append(loss.item())

    #         prefix = ""
    #         if partition != 'train':
    #             prefix = ">> %s \t" % partition

    #         if i % args.log_interval == 0:
    #             print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])))
    #     return res['loss'] / res['counter']

    # res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}
    # for epoch in range(0, args.epochs):
    #     train(epoch, dataloaders['train'], partition='train')
    #     if epoch % args.test_interval == 0:
    #         val_loss = train(epoch, dataloaders['valid'], partition='valid')
    #         test_loss = train(epoch, dataloaders['test'], partition='test')
    #         res['epochs'].append(epoch)
    #         res['losess'].append(test_loss)

    #         if val_loss < res['best_val']:
    #             res['best_val'] = val_loss
    #             res['best_test'] = test_loss
    #             res['best_epoch'] = epoch
    #         print("Val loss: %.4f \t test loss: %.4f \t epoch %d" % (val_loss, test_loss, epoch))
    #         print("Best: val loss: %.4f \t test loss: %.4f \t epoch %d" % (res['best_val'], res['best_test'], res['best_epoch']))


    #     json_object = json.dumps(res, indent=4)
    #     with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
    #         outfile.write(json_object)


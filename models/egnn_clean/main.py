from tqdm import tqdm 
import torch
from torch import nn,optim
from egnn_clean import EGNN 
from divider import Dataset 
import numpy as np 
import argparse
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# PARSER ARGUMENTS
parser = argparse.ArgumentParser(description='DL course Project')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N',
                    help='experiment_name')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no_cuda', action='store_true', default=False,
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

def stats(part,preds,gts,losses):
    # Accuracy 
    # preds = list(map(round,preds))
    # new_gts = list(map(np.argmax,gts))
    # new_preds = list(map(np.argmax,preds))
    # acc  = accuracy_score(y_pred=new_preds,y_true=new_gts) 
    loss_avg = sum(losses)/len(losses)  # Average Loss 
    print(f"{part}:  Avg_loss: {loss_avg:.3f}") #Accuracy:{acc:.3f}") 


def one_hot(y):
    new_y = torch.zeros(len(y),38)
    # print(new_y)
    # print(len(new_y[0]))
    # print(new_y.shape)
    for i,val in enumerate(y):
        new_y[i,int(val)] = 1 
    return new_y

def visualize(h, color):
    z = TSNE(n_components=2,perplexity=10).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

def main():
    print("\n###MAIN###\n")
    print("Device: {}".format(device))

    # Model and Stuff
    # model = EGNN(out_node_nf=1,in_node_nf=1, in_edge_nf=1, hidden_nf=32, device=device, n_layers=3,attention=False,normalize=False) # For L1loss
    model = EGNN(out_node_nf=38,in_node_nf=1, in_edge_nf=1, hidden_nf=32, device=device, n_layers=3,attention=False,normalize=False) # for Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss = nn.CrossEntropyLoss() # Better
    # loss = nn.L0Loss()
    
    # Generate Training and Testing datasets
    dataset = Dataset()
    train_dl, test_dl = dataset.generateData()

    # Softmax Example
    soft_max = nn.Softmax(dim=1)
    # input = torch.randn(2, 3)
    # output = soft_max(input)

    for epoch in range(args.epochs):

        train_losses,test_losses = [],[]
        train_preds,train_gts = [],[]
        test_preds,test_gts = [],[]
        
        model.train()
        for data in iter(train_dl):
            # Transforming tensors of edge_index (adjacency)
            edge_index = []
            edge_index.append(data.edge_index[0])
            edge_index.append(data.edge_index[1])
            h = torch.ones(len(data.x),1)#.view(len(data.x),1)
            h, x = model(h, data.x, edge_index, data.edge_attr)
            # visualize(h,data.y)

            new_y = one_hot(data.y)
            h_softmaxed = soft_max(h)
            print(model)
            print()
            print(h_softmaxed)
            exit()
            train_loss = loss(h_softmaxed,new_y)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Save Results for Statistics
            if epoch % 10 == 0:
                train_losses.append(train_loss.item())
                train_preds = np.concatenate((train_preds, h_softmaxed.flatten().detach().numpy()))
                train_gts = np.concatenate((train_gts, new_y.flatten().numpy()))

        if epoch % 10 == 0:
            model.eval()
            for data in test_dl:

                h = torch.ones(len(data.x),1)
                edge_index = []
                edge_index.append(data.edge_index[0])
                edge_index.append(data.edge_index[1])
                h, x = model(h, data.x, edge_index, data.edge_attr)

                new_y = one_hot(data.y)
                h_softmaxed = soft_max(h)
                test_loss = loss(h_softmaxed,new_y)

                # Save results for statistics
                if epoch % 10 == 0:
                    test_losses.append(test_loss.item())
                    test_preds = np.concatenate((test_preds, h_softmaxed.flatten().detach().numpy()))
                    test_gts = np.concatenate((test_gts, new_y.flatten().numpy()))

            # Compute Stats 
            print(f"Epoch: {epoch}")
            stats("Train",train_preds,train_gts,train_losses)
            stats("Test",test_preds,test_gts,test_losses)
            print()
                        
    print("Saving the Model")
    torch.save(model.state_dict,"./state_dict/egnn.pt")


    

if __name__ == "__main__":
    main()

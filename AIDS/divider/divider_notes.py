# Single data 
from torch_geometric.data import Data
data = Data(x=x, edge_index=edge_index, y=y)
# Create an array and then the DataLoader 
from torch_geometric.loader import DataLoader
data_list = [Data(...), ..., Data(...)]
loader = DataLoader(data_list, batch_size=32)
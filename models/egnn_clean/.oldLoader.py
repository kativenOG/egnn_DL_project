import numpy as np 
import datasets # hugging face 
import torch 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Loading AIDS dataset from hugging face
class generateDatasets():

    def __init__(self):
        all_data = datasets.list_datasets()
        target = [d for d in all_data if ("AIDS" in d) or ("aids" in d)][0]
        train_ds,test_ds= datasets.load_dataset(target,split=["full[:80%]","full[80%:]"])
        self.train_ds,self.test_ds= train_ds,test_ds

    def generate(self,shuffle=True,batch_size=20):
        train_list = [Data(graph) for graph in self.train_ds] # Taking only the train graphs 
        test_list = [Data(graph) for graph in self.test_ds] # Taking only the test graphs 
        dl_train = DataLoader(train_list,shuffle=shuffle,batch_size=batch_size)
        dl_test = DataLoader(test_list,shuffle=shuffle,batch_size=batch_size)
        for _ in dl_train:
            pass
        return dl_train,dl_test

# gds = generateDatasets() 
# self.train_ds, test_ds = gds.generate()
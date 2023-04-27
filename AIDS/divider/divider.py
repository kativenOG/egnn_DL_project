import torch 
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
import math 

def graph_stats(warehouse,action_dict):
    """
    Check the average number of nodes and labels 
    (they are provided on the dataset site, so when can check)
    """
    avg_nodes,avg_labels = 0,0
    for data in warehouse: 
       avg_nodes+= len(data.edge_set) if action_dict["Nodes"] == True else 0 
       avg_labels+= len(data.adjecency_and_labels) if action_dict["Edges"] == True else 0 
    # PRINTS:
    print("STATS:")
    if action_dict["Nodes"] == True:
        print(f"Avg number of Nodes: {avg_nodes/2000}")
    if action_dict["Edges"] == True: 
        print(f"Avg number of Labels (ALL): {avg_labels/2000}")
        print(f"Avg number of Labels (Counting each pair only once) : {avg_labels/4000}")


class storage: 
    """
    Container for a single graph during data extraction  
    """
    def __init__(self,n):
        self.n = n +1
        self.node_attributes_and_labels= [] # Node Attributes with respectives labels 
        self.adjecency_and_labels = [] # Adjecency matrix and edges label
        self.edge_set = set() # Nodes counter 

    def print(self):
        print(f"#####Storage Print#####")
        print(f"Normalized Edges with Labels: {self.adjecency_and_labels}")
        print(f"Node Attributes with labels: {self.node_attributes_and_labels}")
        print()
    
    def add_node(self,n):
        self.edge_set.add(n)

    def self_add_node_attr(self,attr_array):
        cols = list(attr_array.columns)
        edge_list = list(self.edge_set)
        edge_list.sort() # this way all attr arrays are already sorted 
        for node in edge_list:
            pos = node- 1
            df_line = attr_array.iloc[pos]
            val = [df_line[index] for index in cols] 
            self.node_attributes_and_labels.append(val)
        
    def add_edge(self,n1,n2,label):
        if (n1 in self.edge_set) or (n2 in self.edge_set):
            self.adjecency_and_labels.append(list([n1,n2,label]))
            return True 
        return False  
    
    def normalize_storage(self):
        # Trasform edge_set to list for iterating in order
        edge_list = list(self.edge_set)
        edge_list.sort() 
        # Generating the dict for normalizing the edges 
        node_dict,counter= {},0
        for node in edge_list:
            counter+=1
            node_dict[str(node)] = counter
        # Now normalize the edges using the node_dict 
        for i in range(len(self.adjecency_and_labels)):
            try:
                self.adjecency_and_labels[i][0] = node_dict[str(self.adjecency_and_labels[i][0])]
                self.adjecency_and_labels[i][1] =  node_dict[str(self.adjecency_and_labels[i][1])]
            except:
                print("####ERROR####")
                print(f"Node list: {edge_list}\n")
                print(f"Edges: {self.adjecency_and_labels}\n")
                print(f"PAIR: {self.adjecency_and_labels[i][0]},{self.adjecency_and_labels[i][1]}")
                exit()

    def extract_data(self):

        num_nodes = len(list(self.edge_set))
        node_appo = np.array(self.node_attributes_and_labels)
        edge_appo = np.array(self.adjecency_and_labels)

        x = torch.Tensor(node_appo[:,0:4])
        y =  torch.Tensor(node_appo[:,4].T).view(len(node_appo),1) # node labels 
        edge_index  = torch.Tensor(edge_appo[:,0:2]).view(2,len(edge_appo))
        edge_attr = torch.Tensor(edge_appo[:,2].T).view(len(edge_appo),1) # edge labels 
        

        data = Data(x=x,edge_attr=edge_attr,edge_index=edge_index,y=y)
        data.num_nodes = num_nodes
        # data.is_directed = False  # BOHHHH VEDIAMO 

        return data  

class Dataset: 

    def __init__(self):
        self.data_array = None  
        self.data_loader= None  

    def generateData(self):
        action_dict = {"Nodes": False,"Edges":False} # STATS DICT 

        # EDGES dataframes:
        e_adj_label= pd.read_csv("../AIDS_A.txt",header=None) # Matrice di Adiacenza 
        e_attr= pd.read_csv("../AIDS_edge_labels.txt",header=None)
        e_adj_label["label"] =  e_attr[0]
        
        # NODES dataframes:
        n_attr_label = pd.read_csv("../AIDS_node_attributes.txt",header=None)
        n_label= pd.read_csv("../AIDS_node_labels.txt",header=None)
        n_attr_label["label"] =  n_label[0]
        
        
        # List for Storing all the 2000 graphs 
        warehouse = [ storage(i) for i in range(2000)] 
        
        # Graph Indicator files that tells us every Node Graph 
        graph_indicator = pd.read_csv("../AIDS_graph_indicator.txt",header=None)
        # counter = 0 

        # Adding all graph nodes to storage set 
        print("Adding Nodes based on graph_indicator! ")
        for index,graph_n in graph_indicator.iterrows():
            val = int(graph_n[0])-1
            warehouse[val].add_node(int(index+1))
        print("Done!")
        
        # Adding all node attributes and node labels to every graph
        print("Adding node attributes and labels to their graphs ")
        for data in tqdm(warehouse):
            data.self_add_node_attr(n_attr_label)
        print("Done!")
    
        # STATS: 
        action_dict["Nodes"]= True 
        graph_stats(warehouse,action_dict)
    
        # Add edges to create a adjecency matrix for each cell 
        print("Create the adjacency matrix for each graph")
        pivot = math.ceil(len(e_adj_label)/2)
        inverse_warehouse  = warehouse[::-1]
        for index,edge in tqdm(e_adj_label.iterrows()):
            iterable =  warehouse if index<pivot else inverse_warehouse
            a,b,label = edge[0],edge[1],edge["label"]
            for instance in iterable:
                if instance.add_edge(a,b,label): break
        print("Done!")

        # STATS:
        action_dict["Edges"],action_dict["Nodes"] = True, False
        graph_stats(warehouse,action_dict)
    
        # Normalize node numbers 
        print("Normalizing Nodes! ")
        for instance in tqdm(warehouse):
            instance.normalize_storage() 
        print("Done!")


        print("Generate Dataloader! ")
        dataset = []
        for instance in tqdm(warehouse):
            dataset.append(instance.extract_data())
        data_loader = DataLoader(dataset=dataset,batch_size=20,shuffle=True)
        print("Done!")

        return data_loader 


# if __name__ == "__main__":
#     dataset = Dataset()
#     data_loader = dataset.generateData()
#     print("Iterating trough the DataLoader!")
#     for data in data_loader:
#         print(data)
#         break
#     exit()

import torch 
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
    
    def add_node(self,n):
        self.edge_set.add(n)

    def self_add_node_attr(self,attr_array):
        edge_list = list(self.edge_set)
        edge_list.sort() # this way all attr arrays are already sorted 
        for node in edge_list:
            self.node_attributes_and_labels.append(attr_array.iloc[node]) #-1])
        # print(f"Graph {self.n} Done")
        
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
                print(f"Node list: {edge_list}\n")
                print(f"Edges: {self.adjecency_and_labels}\n")
                print(f"PAIR: {self.adjecency_and_labels[i][0]},{self.adjecency_and_labels[i][1]}")
                exit()
    

class Dataset: 

    def __init__(self):
        self.data_array = None  
        self.data_loader= None  

    def generateData(self):
        action_dict = {"Nodes": False,"Edges":False}
        # EDGES dataframes:
        e_adj_label= pd.read_csv("AIDS_A.txt") # Matrice di Adiacenza 
        e_attr= pd.read_csv("AIDS_edge_labels.txt")
        e_adj_label["label"] =  e_attr["0"]
        
        # NODES dataframes:
        n_attr_label = pd.read_csv("AIDS_node_attributes.txt")
        n_label= pd.read_csv("AIDS_node_labels.txt")
        n_attr_label["label"] =  n_label["0"]
        
        
        # List for Storing all the 2000 graphs 
        warehouse = [ storage(i) for i in range(2000)] 
        
        # Graph Indicator files that tells us every Node Graph 
        graph_indicator = pd.read_csv("AIDS_graph_indicator.txt")
        # counter = 0 

        # Adding all graph nodes to storage set 
        warehouse[0].add_node(1)
        for index,graph_n in graph_indicator.iterrows():
            val = int(graph_n["1"])-1
            warehouse[val].add_node(int(index+2))
        print(warehouse[-1].edge_set)
        
        # Adding all node attributes and node labels to every graph
        for data in tqdm(warehouse):
            data.self_add_node_attr(n_attr_label)
    
        # Check stats
        action_dict["Nodes"]= True 
        graph_stats(warehouse,action_dict)
    
        # Add edges to create a adjecency matrix for each cell 
        pivot = math.ceil(len(e_adj_label)/2)
        inverse_warehouse  = warehouse[::-1]
        for index,edge in tqdm(e_adj_label.iterrows()):
            iterable =  warehouse if index<pivot else inverse_warehouse
            a,b,label = edge["1"],edge[" 2"],edge["label"]
            for instance in iterable:
                if instance.add_edge(a,b,label): break
        # STATS:
        action_dict["Edges"],action_dict["Nodes"] = True, False
        graph_stats(warehouse,action_dict)
    
        # Normalize node numbers 
        print("Normalizing Nodes! ")
        for instance in tqdm(warehouse):
            instance.normalize_storage() 
        print("Done!")
        
        return None 


if __name__ == "__main__":
    dataset = Dataset()
    data_loader = dataset.generateData()
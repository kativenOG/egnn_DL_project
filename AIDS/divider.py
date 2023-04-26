import pandas as pd 
import numpy as np 
from tqdm import tqdm 
# import random 
# import copy 

def graph_stats(array):
    # Graphs stats 
    counter,avg,minimum,maximum = 0,0,1000000,0
    for instance in summed:
        node_number =  len(instance.edge_set)
        minimum=  min(minimum,node_number)
        maximum=  max(maximum,node_number)
        avg += node_number
        if node_number==2:
            counter+=1
    avg = avg/len(summed)
    print(f"Average number of Nodes: {avg}")
    print(f"One link Arrays: {counter}")
    print(f"Minimum: {minimum}\nMaximum: {maximum}")


class storage:
    def __init__(self):
        self.node_attributes_and_labels= [] # Node Attributes with respectives labels 
        self.adjecency_and_labels = [] # Adjecency matrix and edges label
        self.edge_set = set() # Nodes counter 

    def add(self,edge,edge_label):
        self.adjecency_and_labels.append(list([edge["1"],edge[" 2"],edge_label["0"]]))
        self.edge_set.add(edge["1"])
        self.edge_set.add(edge[" 2"])
    
    def check(self,edge):
        if (edge["1"] in self.edge_set) or (edge[" 2"] in self.edge_set):
            return True 
        return False 

    def merge(self,storage):
        self.adjecency_and_labels = self.adjecency_and_labels + storage.adjecency_and_labels#list(set(self.adjecency_and_labels) | set(storage.adjecency_and_labels))
        self.edge_set = self.edge_set | storage.edge_set

    def normalize(self):
        edge_list = list(self.edge_set)
        edge_list.sort() 
        node_dict,counter= {},0
        for node in edge_list:
            counter+=1
            node_dict[str(node)] = counter
        print(f"Node Dict: {node_dict}")
        for i in range(len(self.adjecency_and_labels)):
            print(f"Pair: {self.adjecency_and_labels[i][0]} ,{self.adjecency_and_labels[i][1]}")
            self.adjecency_and_labels[i][0] = node_dict[str(self.adjecency_and_labels[i][0])]
            self.adjecency_and_labels[i][1] =  node_dict[str(self.adjecency_and_labels[i][1])]

    # def print(self):
    #     print(f"ATTRIBUTES AND LABELS: {self.node_attributes_and_labels}")
    
# Loading files with pandas dataframes 
# EDGES:
m_adiancenza = pd.read_csv("AIDS_A.txt")
edge_attributes= pd.read_csv("AIDS_edge_labels.txt")
# NODES:
node_attributes = pd.read_csv("AIDS_node_attributes.txt")
node_labels= pd.read_csv("AIDS_node_labels.txt")

# Generating graphs based on the edges (using the Adjecency Matrix)
print("Generating graphs based on the edges")
edge_divided_graps = []
appo = storage()
for (index1,edge),(index2,label) in zip(m_adiancenza.iterrows(),edge_attributes.iterrows()):
    a,b = edge["1"],edge[" 2"]
    if (a not in appo.edge_set)  and (b not in appo.edge_set): 
        control = False 
        for i,graph in enumerate(edge_divided_graps): # Check for nodes in all graphs 
            if (a in graph.edge_set) or (b in graph.edge_set):
                edge_divided_graps[i].add(edge,label)
                control = True 
                break 
        if control == False:  # create new graph if nodes are not found 
            edge_divided_graps.append(appo)
            appo = storage()
            appo.add(edge,label)
    else:
        appo.add(edge,label) 

edge_divided_graps.pop(0)
print(f"First graph: {edge_divided_graps[0].edge_set}")
print(f"Lenght: {len(edge_divided_graps)}")

# Non abbastanza ! facciamo intersezione tra i set per sommare due storage che contengono uno stesso nodo  
summed = []
check,counter= True,0
while check and counter<10:#and counter<6:
    check = False  
    # random.shuffle(edge_divided_graps)
    summed = []
    already_summed_set = set()
    for i,storage1 in enumerate(tqdm(edge_divided_graps)):
        if i not in already_summed_set: # skip if already summed 
            new_storage = storage1#copy.copy(storage1)
            pos = i+1
            for j,storage2 in enumerate(edge_divided_graps[pos:]):
                if not new_storage.edge_set.isdisjoint(storage2.edge_set):
                    new_storage.merge(storage2)
                    summ = pos+j
                    already_summed_set.add(summ)
                    check = True 
            summed.append(new_storage)
    edge_divided_graps = summed #copy.copy(summed)
    counter+=1
    print(f"Lenght: {len(summed)}")

# Add nodes and labels to storage graphs 
print("Adding Nodes! ")
attr = list(node_attributes.columns)
index = 0  
for (index1,features),(index2,label) in zip(node_attributes.iterrows(),node_labels.iterrows()):
    index+=1
    attr_and_label = [features[attr[0]],features[attr[1]],features[attr[2]],features[attr[3]],label["0"]]
    for storage in summed:
        if index in storage.edge_set:
            storage.node_attributes_and_labels.append(attr_and_label)
            break
print("Done!")


# Normalizing nodes for every graph
print("Normalizing Nodes! ")
for storage in tqdm(summed):
    storage.normalize() 
print("Done!")

graph_stats(summed)

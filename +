import pandas as pd 
import numpy as np 
import copy 

class storage:
    def __init__(self):
        # Node Attributes with respectives labels 
        self.node_attributes_and_labels= []
        # Adjecency matrix and edges label
        self.adjecency_and_labels = []
        # self.edge_label= []
        # Nodes counter 
        self.edge_set = set()

    def add(self,edge,edge_label):
        # self.edge_label.append(edge_label["0"])
        self.adjecency_and_labels.append(np.array([edge["1"],edge[" 2"],edge_label["0"]]))
        # adding nodes identifiers to set  
        self.edge_set.add(edge["1"])
        self.edge_set.add(edge[" 2"])
    
    def check(self,edge):
        if (edge["1"] in self.edge_set) or (edge[" 2"] in self.edge_set):
            return True 
        return False 

    def merge(self,storage):
        self.adjecency_and_labels = list(set(self.adjecency_and_labels ) | set(storage.adjecency_and_labels))
        self.edge_set = self.edge_set | storage.edge_set
    
# Loading files with pandas dataframes 
m_adiancenza = pd.read_csv("AIDS_A.txt")
edge_attributes= pd.read_csv("AIDS_edge_labels.txt")

# Generating graphs based on the edges (using the Adjecency Matrix)
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

print(len(edge_divided_graps))

# Non abbastanza ! facciamo intersezione tra i set per sommare due storage che contengono uno stesso nodo  
summed = []
for i,storage1 in enumerate(edge_divided_graps):
    new_storage = copy.copy(storage1)
    for storage2 in edge_divided_graps[i:]:
        if not new_storage.edge_set.isdisjoint(storage2.edge_set):
            new_storage.merge(storage2)
    summed.append(new_storage)

# print(len(summables))

# check = True 
# while check:
#     check = False 
#     summables_improved= []
#     for i,first in enumerate(summables):
#         not_found = True 
#         for j,second in enumerate(summables[i:]):
#             if len(set(first and second))!=0: 
#                 summables_improved.append((first | second ))
#                 check = True  
#                 not_found = False  
#         if not_found:
#             summables_improved.append(first)
#     summables = summables_improved 

# print(summables)

# node_list_graph_divided = []
# appo = set() 
# nodes = set()
# for index, row in m_adiancenza.iterrows():
#     a,b = row["1"],row[" 2"] 

#     # Nodes set for checking 
#     nodes.add(a)
#     nodes.add(b)

#     if (a not in appo)  and (b not in appo): # check if nodes are not in the last graph (both of them)
#         control = False 
#         for i,graph in enumerate(node_list_graph_divided): # Check for nodes in all graphs 
#             if (a in graph):
#                 node_list_graph_divided[i].append(b)
#                 control = True 
#                 break 
#             if  (b in graph):
#                 node_list_graph_divided[i].append(a)
#                 control = True 
#                 break 

#         if control == False:  # create new graph if nodes are not found 
#             node_list_graph_divided.append(list(appo))
#             appo = set()
#             appo.add(a)
#             appo.add(b)
#     else:
#         appo.add(a)
#         appo.add(b)

# # print(node_list_graph_divided)
# print(np.asarray(node_list_graph_divided))

# print("Checking node set: ")
# node_list = list(nodes) 
# node_list.sort()
# appo = 0
# for node in node_list:
#     confront = appo +1
#     if  confront != node: 
#         print("ERROR")
#         break
#     else:
#         print(node)
#     appo = node




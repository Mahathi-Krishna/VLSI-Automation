# Read the circuit file
import re
import numpy as np

from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import inv
from numpy.linalg import solve

filename = 'testcase18'
filepath = f'./benchmarks/{filename}.sp'
data = ""

nodes = set()   # Stores unique node combinations
node_index = {} # Stores  [node: index] for each node combination
G_data = []     # Conductance values with nodes in a matrix
V_data = []     # Voltage values with nodes in a matrix
I_data = []     # Current values with nodes in a matrix
row_mat = []    # Store the x-coordinates
col_mat = []    # Store the y-coordinates
val_mat = []    # Store the value
V_matrix = []   # Voltage matrix

# Read netlist:
with open(filepath, 'r') as file:
    data = file.readlines()

# Sort inorder:
def natural_sort_key(s):
    parts = re.split(r'(\d+)', s)  # Split by numbers while keeping them
    return [int(p) if p.isdigit() else p for p in parts]

# Parse netlist data:
for line in data:
    if not line.startswith(".") and len(line.split()) == 4:
        components = line.split()
        # Process Resistances:
        if(components[0].startswith("R")):
            G_data.append((components[1], components[2], 1/float(components[3])))
        # Process Currents:
        elif(components[0].startswith("I")):
            I_data.append((components[1], components[2], float(components[3])))
        # Process Voltages:
        elif(components[0].startswith("V")):
            V_data.append((components[1], components[2], float(components[3])))
        # Save node information:
        components[1] = components[2] if components[1] == '0' else components[1]
        components[2] = components[1] if components[2] == '0' else components[2]
        nodes.update([components[1], components[2]])

# Identify each node with an index:
nodes = sorted(nodes, key=natural_sort_key)
max_n = len(nodes)
node_index = {node:i for i, node in enumerate(nodes)}

# Form G-matrix:
G_mat_dict = {} # Sparse G-matrix as dictionary
for net1, net2, G in G_data:
    i,j = node_index[net1], node_index[net2]
    G_mat_dict[(i,i)] = (G_mat_dict[(i,i)] + G) if (i,i) in G_mat_dict else G
    G_mat_dict[(i,j)] = (G_mat_dict[(i,j)] - G) if (i,j) in G_mat_dict else -G
    G_mat_dict[(j,i)] = (G_mat_dict[(j,i)] - G) if (j,i) in G_mat_dict else -G
    G_mat_dict[(j,j)] = (G_mat_dict[(j,j)] + G) if (j,j) in G_mat_dict else G

# print(G_data)
# print(G_mat_dict)

# Form Current matrix:
I_matrix = np.zeros(max_n)
for net1, net2, I in I_data:
    if net2 == '0':
        i = node_index[net1]
        I_matrix[i] = -I
    elif net1 == '0':
        i = node_index[net2]
        I_matrix[i] = I

# Modified G-matrix:
if len(V_data) > 0:
    for net1, net2, V in V_data:
        I_matrix = np.append(I_matrix, V)
        if net2 == '0':
            i = node_index[net1]
            G_mat_dict[(i,max_n)] = 1
            G_mat_dict[(max_n,i)] = 1
        elif net1 == '0':
            i = node_index[net2]
            G_mat_dict[(i,max_n)] = 1
            G_mat_dict[(max_n,i)] = 1
        max_n += 1

# Separate rows, cols, vals from G_mat_dict:
for (row,col), value in G_mat_dict.items():
    row_mat.append(row)
    col_mat.append(col)
    val_mat.append(value)

G_matrix = coo_matrix((val_mat, (row_mat, col_mat))).tocsr()

V_matrix = spsolve(G_matrix, I_matrix)

print(len(V_matrix), len(V_data), len(nodes))

with open(f'{filename}.txt', 'w') as file:
    for node, voltage in zip(nodes, V_matrix):
        file.write(f"{node}\t{voltage}\n")
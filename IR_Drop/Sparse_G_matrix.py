# Class for sparse G-matrix
import re
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

class Sparse_G_matrix:
    # Process the data into a sparse matrix:
    def __init__(self, data):
        self.nodes = set()   # Stores unique node combinations
        self.node_index = {} # Stores  [node: index] for each node combination
        self.g_data = []     # Conductance values with nodes in a matrix
        self.v_data = []     # Voltage values with nodes in a matrix
        self.i_data = []     # Current values with nodes in a matrix
        self.row_mat = []    # Store the x-coordinates
        self.col_mat = []    # Store the y-coordinates
        self.val_mat = []    # Store the value
        self.v_vector = []   # Voltage matrix

        for line in data:
            if not line.startswith(".") and len(line.split()) == 4:
                components = line.split()
                # Process Resistances:
                if(components[0].startswith("R")):
                    self.g_data.append((components[1], components[2], 1/float(components[3])))
                # Process Currents:
                elif(components[0].startswith("I")):
                    self.i_data.append((components[1], components[2], float(components[3])))
                # Process Voltages:
                elif(components[0].startswith("V")):
                    self.v_data.append((components[1], components[2], float(components[3])))
                # Save node information:
                components[1] = components[2] if components[1] == '0' else components[1]
                components[2] = components[1] if components[2] == '0' else components[2]
                self.nodes.update([components[1], components[2]])

        # Identify each node with an index:
        self.nodes = sorted(self.nodes, key=self.sort_key)
        max_n = len(self.nodes)
        self.node_index = {node:i for i, node in enumerate(self.nodes)}

        # Form G-matrix:
        G_mat_dict = {} # Sparse G-matrix as dictionary
        for net1, net2, G in self.g_data:
            i,j = self.node_index[net1], self.node_index[net2]
            G_mat_dict[(i,i)] = (G_mat_dict[(i,i)] + G) if (i,i) in G_mat_dict else G
            G_mat_dict[(i,j)] = (G_mat_dict[(i,j)] - G) if (i,j) in G_mat_dict else -G
            G_mat_dict[(j,i)] = (G_mat_dict[(j,i)] - G) if (j,i) in G_mat_dict else -G
            G_mat_dict[(j,j)] = (G_mat_dict[(j,j)] + G) if (j,j) in G_mat_dict else G

        # Form Current matrix:
        i_vector = np.zeros(max_n)
        for net1, net2, I in self.i_data:
            if net2 == '0':
                i = self.node_index[net1]
                i_vector[i] = -I
            elif net1 == '0':
                i = self.node_index[net2]
                i_vector[i] = I

        # Modified G-matrix:
        if len(self.v_data) > 0:
            for net1, net2, V in self.v_data:
                i_vector = np.append(i_vector, V)
                if net2 == '0':
                    i = self.node_index[net1]
                    G_mat_dict[(i,max_n)] = 1
                    G_mat_dict[(max_n,i)] = 1
                elif net1 == '0':
                    i = self.node_index[net2]
                    G_mat_dict[(i,max_n)] = 1
                    G_mat_dict[(max_n,i)] = 1
                max_n += 1

        # Separate rows, cols, vals from G_mat_dict:
        for (row,col), value in G_mat_dict.items():
            self.row_mat.append(row)
            self.col_mat.append(col)
            self.val_mat.append(value)
        
        sparse_g_matrix = coo_matrix((self.val_mat, (self.row_mat, self.col_mat))).tocsr()

        self.i_vector = i_vector
        self.sparse_g_matrix = sparse_g_matrix
    
    # Sort inorder:
    def sort_key(self, s):
        parts = re.split(r'(\d+)', s)  # Split by numbers while keeping them
        return [int(p) if p.isdigit() else p for p in parts]

    # Compute the Voltage vector:
    def get_voltage_vector(self):
        self.v_vector = spsolve(self.sparse_g_matrix, self.i_vector)
    
    # Return G_matrix, Voltage & Current vectors:
    def getGImatrix(self):
        return self.sparse_g_matrix, self.i_vector, self.v_vector
    
    # Write the node voltages:
    def write_voltage_vector(self, outfilename):
        with open(f'{outfilename}', 'w') as file:
            for node, voltage in zip(self.nodes, self.v_vector):
                file.write(f"{node}\t{voltage}\n")
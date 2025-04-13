# Class for Netlist Parser:
import re
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class Netlist_Parser:
    # Process the netlist data and form a sparse G-matrix:
    def __init__(self, data):
        self.nodes = set()              # Stores unique node combinations
        self.node_index = {}            # Stores  [node: index] for each node combination
        self.g_data = []                # Conductance values with nodes in a matrix [(node1, node2, G),...]
        self.v_data = []                # Voltage values with nodes in a matrix [(node1, node2, V),...]
        self.i_data = []                # Current values with nodes in a matrix [(node1, node2, I),...]
        self.row_mat = []               # Store the x-coordinates
        self.col_mat = []               # Store the y-coordinates
        self.val_mat = []               # Store the value
        self.v_vector = []              # Voltage matrix
        self.x_coord_max = 0            # Stores the max x_coord
        self.y_coord_max = 0            # Stores the max y_coord
        self.x_max = 0                  # x_coord_max // 2000
        self.y_max = 0                  # y_coord_max // 2000
        self.area = 0                   # Total area of the netlist
        self.current_coord = []         # Stores the current data [[x, y, value],...]
        self.voltage_sources_grid = []  # Stores the location of all voltage sources
        self.metal_layers = set()       # Stores the metal layers

        for line in data:
            if not line.startswith(".") and len(line.split()) == 4:
                components = line.split()
                item = components[0]
                node1 = components[1]
                node2 = components[2]
                val = components[3]
                
                # Process the metal layer used:
                self.Save_Metal_Info(node1, node2)

                # Compute max X & Y coordinates:
                if(node1 == '0'):
                    x_coord1, x_coord2 = 0, int(node2.split('_')[-2])
                    y_coord1, y_coord2 = 0, int(node2.split('_')[-1])
                elif(node2 == '0'):
                    x_coord1, x_coord2 = int(node1.split('_')[-2]), 0
                    y_coord1, y_coord2 = int(node1.split('_')[-1]), 0
                else:
                    x_coord1, x_coord2 = int(node1.split('_')[-2]), int(node2.split('_')[-2])
                    y_coord1, y_coord2 = int(node1.split('_')[-1]), int(node2.split('_')[-1])
                
                self.x_coord_max = max(self.x_coord_max, x_coord1, x_coord2)
                self.y_coord_max = max(self.y_coord_max, y_coord1, y_coord2)

                # Process Resistances:
                if(item.startswith("R")):
                    self.g_data.append((node1, node2, 1/float(val)))
                
                # Process Currents:
                elif(item.startswith("I")):
                    self.i_data.append((node1, node2, float(val)))
                    # Store current details with converted coordinates:
                    x_coord = x_coord2//2000 if (x_coord1//2000 == 0) else x_coord1//2000
                    y_coord = y_coord2//2000 if (y_coord1//2000 == 0) else y_coord1//2000
                    self.current_coord.append((x_coord, y_coord, float(val)))
                
                # Process Voltages:
                elif(item.startswith("V")):
                    self.v_data.append((node1, node2, float(val)))
                    # Store voltage source details:
                    x_coord = x_coord2//2000 if (x_coord1//2000 == 0) else x_coord1//2000
                    y_coord = y_coord2//2000 if (y_coord1//2000 == 0) else y_coord1//2000

                    if (x_coord, y_coord) not in self.voltage_sources_grid:
                        self.voltage_sources_grid.append((x_coord, y_coord))
                
                # Save node information:
                node1 = node2 if node1 == '0' else node1
                node2 = node1 if node2 == '0' else node2
                self.nodes.update([node1, node2])

        # Calculate dimension and area:
        self.x_max = (self.x_coord_max // 2000) + 1
        self.y_max = (self.y_coord_max // 2000) + 1
        dim = f"Resolution: {self.x_max} x {self.y_max}"
        self.area = self.x_max * self.y_max
        print(dim, "Area: ", self.area)

        # Identify each node with an index:
        self.nodes = sorted(self.nodes, key=self.Sort_Key)
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
        self.v_vector = spsolve(self.sparse_g_matrix, self.i_vector)
    
    # Sort inorder:
    def Sort_Key(self, s):
        parts = re.split(r'(\d+)', s)  # Split by numbers while keeping them
        return [int(p) if p.isdigit() else p for p in parts]
    
    # Write the node voltages:
    def Write_Voltage_Vector(self, outfilename):
        with open(f'{outfilename}', 'w') as file:
            for node, voltage in zip(self.nodes, self.v_vector):
                file.write(f"{node}\t{voltage}\n")
    
    # Save the metal layer information:
    def Save_Metal_Info(self, node1, node2):
        metal1, metal2 = '', ''
        if(node1 != '0'):
            metal1 = node1.split('_')[1]
            self.metal_layers.update([metal1])
        if(node2 != '0'):
            metal2 = node2.split('_')[1]
            self.metal_layers.update([metal2])
    
    # Process for Current Map:
    def Process_Current_Map(self):
        self.current_map = np.zeros((self.x_max, self.y_max))
        for x, y, val in self.current_coord:
            self.current_map[x, y] += val
    
    # Process for IR Drop and Effective Voltage distances:
    def Process_IR_Drop(self):
        self.ir_drop_mat = np.zeros((self.x_max, self.y_max))
        for node, voltage in zip(self.nodes, self.v_vector):
            if(node.split('_')[1] == 'm1'):
                x, y = int(node.split('_')[-2]), int(node.split('_')[-1])
                x = x // 2000
                y = y // 2000
                self.ir_drop_mat[x, y] = max(self.ir_drop_mat[x, y], (1.1 - voltage))
    
    # Process for Effective Distance to Voltage Sources:
    def Process_Volt_Dist(self):
        self.eff_dist_volt = np.zeros((self.x_max, self.y_max))
        for i in range(self.x_max):     # for each row (along y-axis)
            for j in range(self.y_max): # for each column (along x-axis)
                distances = [(1/np.hypot(j - vx, i - vy)) for vx, vy in self.voltage_sources_grid]
                d_eff = 1/sum(distances)
                self.eff_dist_volt[j, i] = d_eff
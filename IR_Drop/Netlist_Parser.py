# Class for Netlist Parser:
import re
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from collections import defaultdict

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
        self.x_coo_max = 0              # Stores the max x_coord
        self.y_coo_max = 0              # Stores the max y_coord
        self.x_max = 0                  # x_coord_max // 2000
        self.y_max = 0                  # y_coord_max // 2000
        self.area = 0                   # Total area of the netlist
        self.current_coo = []           # Stores the current data [[x, y, value],...]
        self.voltage_sources_grid = []  # Stores the location of all voltage sources
        self.pdn_map = []               # For PDN Map
        self.m4_nodes = set()           # Stores unique metal strip's coordinates based on direction
        self.dbu = 2000                 # Grid units x & y to be scaled down by
        self.block_size = 100           # Bin size
        self.pdn_blocks = {}
        self.pdn_templates = {(600, 1000): 0, (301, 599): 1, (201, 300): 2, (0,200): 3}

        for line in data:
            if not line.startswith(".") and len(line.split()) == 4:
                component = line.split()
                item = component[0]
                node1 = component[1]
                node2 = component[2]
                val = component[3]
                
                # Compute max X & Y coordinates:
                if(node1 == '0'):
                    x_coo1, x_coo2 = 0, int(node2.split('_')[-2])
                    y_coo1, y_coo2 = 0, int(node2.split('_')[-1])
                elif(node2 == '0'):
                    x_coo1, x_coo2 = int(node1.split('_')[-2]), 0
                    y_coo1, y_coo2 = int(node1.split('_')[-1]), 0
                else:
                    x_coo1, x_coo2 = int(node1.split('_')[-2]), int(node2.split('_')[-2])
                    y_coo1, y_coo2 = int(node1.split('_')[-1]), int(node2.split('_')[-1])
                
                # Locate and save individual bins:
                block_x1_min = int(((x_coo1 / self.dbu) // self.block_size) * self.block_size)
                block_x1_max = block_x1_min + 100

                block_x2_min = int(((x_coo2 / self.dbu) // self.block_size) * self.block_size)
                block_x2_max = block_x2_min + 100

                block_y1_min = int(((y_coo1 / self.dbu) // self.block_size) * self.block_size)
                block_y1_max = block_y1_min + 100

                block_y2_min = int(((y_coo2 / self.dbu) // self.block_size) * self.block_size)
                block_y2_max = block_y2_min + 100

                # Update bins dictionary with unique bin coordinates:
                if (((block_x1_min, block_x1_max), (block_y1_min, block_y1_max)) not in self.pdn_blocks.keys()):
                    self.pdn_blocks[((block_x1_min, block_x1_max), (block_y1_min, block_y1_max))] = []
                
                if (((block_x2_min, block_x2_max), (block_y2_min, block_y2_max)) not in self.pdn_blocks.keys()):
                    self.pdn_blocks[((block_x2_min, block_x2_max), (block_y2_min, block_y2_max))] = []
                
                # Check and save max X & Y coordinates:
                self.x_coo_max = max(self.x_coo_max, x_coo1, x_coo2)
                self.y_coo_max = max(self.y_coo_max, y_coo1, y_coo2)

                # Process Resistances:
                if(item.startswith("R")):
                    self.g_data.append((node1, node2, 1/float(val)))
                    
                    # Process m4 metals:
                    node1_metal = node1.split('_')[1]
                    node2_metal = node2.split('_')[1]
                    if (node1_metal == 'm4' and node2_metal == 'm4'):
                        self.m4_nodes.update([node1, node2])
                        self.pdn_blocks[((block_y1_min, block_y1_max), (block_x1_min, block_x1_max))].append(item)
                        self.pdn_blocks[((block_y2_min, block_y2_max), (block_x2_min, block_x2_max))].append(item)
                
                # Process Currents:
                elif(item.startswith("I")):
                    self.i_data.append((node1, node2, float(val)))
                    # Store current details with converted coordinates:
                    x_coord = x_coo2//self.dbu if (x_coo1//self.dbu == 0) else x_coo1//self.dbu
                    y_coord = y_coo2//self.dbu if (y_coo1//self.dbu == 0) else y_coo1//self.dbu
                    self.current_coo.append((x_coord, y_coord, float(val)))
                
                # Process Voltages:
                elif(item.startswith("V")):
                    self.v_data.append((node1, node2, float(val)))
                    # Store voltage source details:
                    x_coord = x_coo2//self.dbu if (x_coo1//self.dbu == 0) else x_coo1//self.dbu
                    y_coord = y_coo2//self.dbu if (y_coo1//self.dbu == 0) else y_coo1//self.dbu

                    if (x_coord, y_coord) not in self.voltage_sources_grid:
                        self.voltage_sources_grid.append((x_coord, y_coord))
                
                # Save node information:
                node1 = node2 if node1 == '0' else node1
                node2 = node1 if node2 == '0' else node2
                self.nodes.update([node1, node2])

        # Calculate dimension and area:
        self.x_max = (self.x_coo_max // self.dbu) + 1
        self.y_max = (self.y_coo_max // self.dbu) + 1
        dim = f"Resolution: {self.x_max} x {self.y_max}"
        self.area = self.x_max * self.y_max
        print(dim, "Area: ", self.area)

        # Update bins upper bound with X_max & Y_max:
        for key in list(self.pdn_blocks):
            x1, x2 = key[0]
            y1, y2 = key[1]
            if(x2 > self.x_max and y2 > self.y_max):
                self.pdn_blocks[((x1, self.x_max), (y1, self.y_max))] = self.pdn_blocks.pop(((x1, x2), (y1, y2)))
            elif(x2 > self.x_max):
                self.pdn_blocks[((x1, self.x_max), (y1, y2))] = self.pdn_blocks.pop(((x1, x2), (y1, y2)))
            elif(y2 > self.y_max):
                self.pdn_blocks[((x1, x2), (y1, self.y_max))] = self.pdn_blocks.pop(((x1, x2), (y1, y2)))

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
    
    # Current Map:
    def Process_Current_Map(self):
        self.current_map = np.zeros((self.x_max, self.y_max))
        for x, y, val in self.current_coo:
            self.current_map[x, y] += val
    
    # IR Drop Map:
    def Process_IR_Drop(self):
        self.ir_drop_mat = np.zeros((self.x_max, self.y_max))
        for node, voltage in zip(self.nodes, self.v_vector):
            if(node.split('_')[1] == 'm1'):
                x, y = int(node.split('_')[-2]), int(node.split('_')[-1])
                x = x // self.dbu
                y = y // self.dbu
                self.ir_drop_mat[x, y] = max(self.ir_drop_mat[x, y], (1.1 - voltage))
        
        # Smooth the ir_drop_mat:
        kernel_size = 5
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)
        mask = self.ir_drop_mat != 0
        # Convolve both the data and the mask
        smoothed_data = convolve(self.ir_drop_mat, kernel, mode='constant', cval=0.0)
        normalization = convolve(mask.astype(np.float64), kernel, mode='constant', cval=0.0)
        # To avoid divide-by-zero and normalize only valid regions
        with np.errstate(divide='ignore', invalid='ignore'):
            self.ir_drop_mat = np.where(normalization > 0, smoothed_data / normalization, 0)
    
    # Effective Distance to Voltage Sources Map:
    def Process_Volt_Dist(self):
        self.eff_dist_volt = np.zeros((self.x_max, self.y_max))
        epsilon = 1e-12 # To prevent divide by zero error
        for i in range(self.x_max):     # for each row (along y-axis)
            for j in range(self.y_max): # for each column (along x-axis)
                distances = [ 1 / (np.hypot(j - vx, i - vy) + epsilon) for vx, vy in self.voltage_sources_grid]
                d_eff = 1/sum(distances)
                self.eff_dist_volt[j, i] = d_eff
    
    # Sort nodes into each block:
    def Sort_By_Block(self):
        str_data = ''
        m4_nodes = defaultdict(list)
        x_max, y_max = self.x_max, self.y_max
        x_blocks = math.ceil(x_max / self.block_size)
        y_blocks = math.ceil(y_max / self.block_size)
        # print("212: ", x_blocks, y_blocks)

        # Store the m4 node's coordinates as per block
        for node in self.m4_nodes:
            x = int(node.split('_')[-2]) // self.dbu
            y = int(node.split('_')[-1]) // self.dbu
            x_block = x // self.block_size
            y_block = y // self.block_size
            # str_data = str_data + f"{x_block, y_block}: {x, y}\n"
            m4_nodes[(x_block, y_block)].append(node) # m4_nodes[(x_block, y_block)].append((x, y))
       
        # Iterate in desired block order
        sorted_nodes = []
        for x_block in range(x_blocks):
            for y_block in range(y_blocks):
                block = (x_block, y_block)
                if block in m4_nodes:
                    m4_nodes[block] = sorted(m4_nodes[block], key=self.Sort_Key)
                    sorted_nodes.append((block, m4_nodes[block]))

        # Save the sorted nodes:
        self.sorted_m4_nodes = sorted_nodes
    
    # PDN Density map:
    def Process_PDN_Map(self):
        x_max = self.x_max
        y_max = self.y_max
        self.pdn_density_map = np.zeros((x_max, y_max))  # Initialize the PDN Density map with 0s

        for (x_range, y_range), res in self.pdn_blocks.items():
            t_val = -1
            val = len(res)
            for (min_val, max_val), template_val in self.pdn_templates.items():
                if min_val <= val <= max_val:
                    t_val = template_val
                    break

            # Get range for plot
            x_start, x_end = x_range
            y_start, y_end = y_range

            # Clamp to design size to avoid out-of-bounds
            x_end = min(x_end, x_max)
            y_end = min(y_end, y_max)

            self.pdn_density_map[y_start:y_end, x_start:x_end] = t_val
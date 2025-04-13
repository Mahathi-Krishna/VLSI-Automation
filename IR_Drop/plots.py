# For plotting and saving the plots:
import numpy as np
import matplotlib.pyplot as plt

# Form Current map:
def form_current_map(current_coord, x_max, y_max):
    current_map = np.zeros((x_max, y_max))
    for x, y, val in current_coord:
        current_map[x, y] += val
    
    plt.figure(figsize=(6,6))
    plt.imshow(current_map, cmap='jet', origin='upper', aspect='equal')
    plt.colorbar()
    plt.title("Current Map")
    plt.xlabel("X Coordinate (μm)")
    plt.ylabel("Y Coordinate (μm)")
    plt.tight_layout()
    plt.show()

# Form IR Drop map:
def form_ir_drop_map(nodes, v_vector, x_max, y_max):
    ir_drop_mat = np.zeros((x_max, y_max))
    for node, voltage in zip(nodes, v_vector):
        if(node.split('_')[1] == 'm1'):
            x, y = int(node.split('_')[-2]), int(node.split('_')[-1])
            x = x // 2000
            y = y // 2000
            ir_drop_mat[x, y] = max(ir_drop_mat[x, y], (1.1 - voltage))

    plt.figure(figsize=(6,6))
    plt.imshow(ir_drop_mat, cmap='jet', origin='upper', aspect='equal')
    plt.colorbar()
    plt.title("IR Drop Map")
    plt.xlabel("X Coordinate (μm)")
    plt.ylabel("Y Coordinate (μm)")
    plt.tight_layout()
    plt.show()
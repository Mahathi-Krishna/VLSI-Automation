# For plotting and saving the plots:
import matplotlib.pyplot as plt

fig_size = (6,6)

def Custom_Plot(input_mat, map_title):
    plt.figure(figsize = fig_size)
    plt.imshow(input_mat, cmap='jet', origin='upper', aspect='equal')
    plt.colorbar()
    plt.title(map_title)
    plt.xlabel("X Coordinate (μm)")
    plt.ylabel("Y Coordinate (μm)")
    plt.tight_layout()
    plt.show()

def Custom_Plot_PDN(input_mat, map_title):
    plt.figure(figsize = fig_size)
    plt.imshow(input_mat, cmap='jet', origin='upper', aspect='equal', vmin=0, vmax=3)
    plt.colorbar()
    plt.title(map_title)
    plt.xlabel("X Coordinate (μm)")
    plt.ylabel("Y Coordinate (μm)")
    plt.tight_layout()
    plt.show()
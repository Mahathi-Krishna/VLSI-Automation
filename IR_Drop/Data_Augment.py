# Function for augmenting and saving the data:
import os
import cv2
import numpy as np

from plotter import *

def Data_Augment (filename, output_path, feature_path, label_path, current_map, voltage_map, pdn_map, ir_drop_map, train_or_test='test'):

    # Plot the four different maps:
    # Custom_Plot(current_map, "Current Map")
    # Custom_Plot(voltage_map, "Effective Distance to Voltage Source Map")
    # Custom_Plot_PDN(pdn_map, "PDN Density Map")
    # Custom_Plot(ir_drop_map, "IR Drop")

    # Check and create directories:
    os.makedirs(feature_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    # Resize the maps using cv2:
    c_map = cv2.resize(current_map, (256, 256), interpolation=cv2.INTER_AREA)
    v_map = cv2.resize(voltage_map, (256, 256), interpolation=cv2.INTER_AREA)
    p_map = cv2.resize(pdn_map, (256, 256), interpolation=cv2.INTER_AREA)
    ir_map = cv2.resize(ir_drop_map, (256, 256), interpolation=cv2.INTER_AREA)


    # 1. Save the normal version of matrices:
    out_path = os.path.join(output_path, f'current_map_{filename}.csv')
    np.savetxt(out_path, current_map, delimiter=',')

    out_path = os.path.join(output_path, f'voltage_source_map_{filename}.csv')
    np.savetxt(out_path, voltage_map, delimiter=',')

    out_path = os.path.join(output_path, f'pdn_density_map_{filename}.csv')
    np.savetxt(out_path, pdn_map, delimiter=',')

    out_path = os.path.join(output_path, f'ir_drop_map_{filename}.csv')
    np.savetxt(out_path, ir_drop_map, delimiter=',')

    # Save the stacked features and labels as npy files:
    # # Features:
    #     index 0 - Current map
    #     index 1 - Voltage source map
    #     index 2 - PDN map
    out_path = os.path.join(feature_path, f"feature_{filename}.npy")
    np.save(out_path, np.stack([c_map, v_map, p_map], axis=0).astype(np.float32))

    out_path = os.path.join(label_path, f"label_{filename}.npy")
    np.save(out_path, ir_map.astype(np.float32))

    # To generate augmented data during training:
    if train_or_test == 'train':
        # 2. Save the vertically filpped versions of matrices:
        out_path = os.path.join(output_path, f'current_map_1_{filename}.csv')
        np.savetxt(out_path, np.flipud(current_map), delimiter=',')

        out_path = os.path.join(output_path, f'voltage_source_map_1_{filename}.csv')
        np.savetxt(out_path, np.flipud(voltage_map), delimiter=',')

        out_path = os.path.join(output_path, f'pdn_density_map_1_{filename}.csv')
        np.savetxt(out_path, np.flipud(pdn_map), delimiter=',')

        out_path = os.path.join(output_path, f'ir_drop_map_1_{filename}.csv')
        np.savetxt(out_path, np.flipud(ir_drop_map), delimiter=',')

        # Save the stacked features and labels as npy files:
        out_path = os.path.join(feature_path, f"feature_1_{filename}.npy")
        np.save(out_path, np.stack([np.flipud(c_map), np.flipud(v_map), np.flipud(p_map)], axis=0).astype(np.float32))

        out_path = os.path.join(label_path, f"label_1_{filename}.npy")
        np.save(out_path, np.flipud(ir_map).astype(np.float32))


        # 3. Save the horizontally filpped version of matrices:
        out_path = os.path.join(output_path, f'current_map_2_{filename}.csv')
        np.savetxt(out_path, np.fliplr(current_map), delimiter=',')

        out_path = os.path.join(output_path, f'voltage_source_map_2_{filename}.csv')
        np.savetxt(out_path, np.fliplr(voltage_map), delimiter=',')

        out_path = os.path.join(output_path, f'pdn_density_map_2_{filename}.csv')
        np.savetxt(out_path, np.fliplr(pdn_map), delimiter=',')

        out_path = os.path.join(output_path, f'ir_drop_map_2_{filename}.csv')
        np.savetxt(out_path, np.fliplr(ir_drop_map), delimiter=',')

        # Save the stacked features and labels as npy files:
        out_path = os.path.join(feature_path, f"feature_2_{filename}.npy")
        np.save(out_path, np.stack([np.fliplr(c_map), np.fliplr(v_map), np.fliplr(p_map)], axis=0).astype(np.float32))

        out_path = os.path.join(label_path, f"label_2_{filename}.npy")
        np.save(out_path, np.fliplr(ir_map).astype(np.float32))


        # 4. Save +90 rotated version of matrices
        out_path = os.path.join(output_path, f'current_map_3_{filename}.csv')
        np.savetxt(out_path, np.rot90(current_map, k=-1), delimiter=',')

        out_path = os.path.join(output_path, f'voltage_source_map_3_{filename}.csv')
        np.savetxt(out_path, np.rot90(voltage_map, k=-1), delimiter=',')

        out_path = os.path.join(output_path, f'pdn_density_map_3_{filename}.csv')
        np.savetxt(out_path, np.rot90(pdn_map, k=-1), delimiter=',')

        out_path = os.path.join(output_path, f'ir_drop_map_3_{filename}.csv')
        np.savetxt(out_path, np.rot90(ir_drop_map, k=-1), delimiter=',')

        # Save the stacked features and labels as npy files:
        out_path = os.path.join(feature_path, f"feature_3_{filename}.npy")
        np.save(out_path, np.stack([np.rot90(c_map, k=-1), np.rot90(v_map, k=-1), np.rot90(p_map, k=-1)], axis=0).astype(np.float32))

        out_path = os.path.join(label_path, f"label_3_{filename}.npy")
        np.save(out_path, np.rot90(ir_map, k=-1).astype(np.float32))


        # 5. Save +270 rotated version of matrices
        out_path = os.path.join(output_path, f'current_map_4_{filename}.csv')
        np.savetxt(out_path, np.rot90(current_map, k=2), delimiter=',')

        out_path = os.path.join(output_path, f'voltage_source_map_4_{filename}.csv')
        np.savetxt(out_path, np.rot90(voltage_map, k=2), delimiter=',')

        out_path = os.path.join(output_path, f'pdn_density_map_4_{filename}.csv')
        np.savetxt(out_path, np.rot90(pdn_map, k=2), delimiter=',')

        out_path = os.path.join(output_path, f'ir_drop_map_4_{filename}.csv')
        np.savetxt(out_path, np.rot90(ir_drop_map, k=2), delimiter=',')

        # Save the stacked features and labels as npy files:
        out_path = os.path.join(feature_path, f"feature_4_{filename}.npy")
        np.save(out_path, np.stack([np.rot90(c_map, k=2), np.rot90(v_map, k=2), np.rot90(p_map, k=2)], axis=0).astype(np.float32))

        out_path = os.path.join(label_path, f"label_4_{filename}.npy")
        np.save(out_path, np.rot90(ir_map, k=2).astype(np.float32))


        # 6. Save -90 rotated version of matrices
        out_path = os.path.join(output_path, f'current_map_5_{filename}.csv')
        np.savetxt(out_path, np.rot90(current_map, k=1), delimiter=',')

        out_path = os.path.join(output_path, f'voltage_source_map_5_{filename}.csv')
        np.savetxt(out_path, np.rot90(voltage_map, k=1), delimiter=',')

        out_path = os.path.join(output_path, f'pdn_density_map_5_{filename}.csv')
        np.savetxt(out_path, np.rot90(pdn_map, k=1), delimiter=',')

        out_path = os.path.join(output_path, f'ir_drop_map_5_{filename}.csv')
        np.savetxt(out_path, np.rot90(ir_drop_map, k=1), delimiter=',')

        # Save the stacked features and labels as npy files:
        out_path = os.path.join(feature_path, f"feature_5_{filename}.npy")
        np.save(out_path, np.stack([np.rot90(c_map, k=1), np.rot90(v_map, k=1), np.rot90(p_map, k=1)], axis=0).astype(np.float32))

        out_path = os.path.join(label_path, f"label_5_{filename}.npy")
        np.save(out_path, np.rot90(ir_map, k=1).astype(np.float32))
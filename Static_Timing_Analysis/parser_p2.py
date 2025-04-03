import argparse     #For Command line interfacing

from circuit_parser import *
from nldm_parser import *
from traversal import *

# parser = argparse.ArgumentParser() 
# parser.add_argument("-r","--read_nldm",help="reads  the nldm file",action = "store_true")   #To parse and populate the datastructure for the NLDM file and find slew
# parser.add_argument("-c","--read_ckt",help="reads the bench file",action="store_true")      #To parse and populate the datastructure for the becnch file
# parser.add_argument('input_ckt_file',help='Path to the input file')                         # Circuit file
# parser.add_argument('input_nldm_file',help='Path to the input file')                        # nldm file

# args = parser.parse_args()
circuit_filepath = './c17.bench' #args.input_ckt_file    # Getting the file mentioned in the command line
nldm_filepath = './sample_NLDM.lib' # args.input_nldm_file      # Getting the file mentioned in the command line

str_data = ""

# Main function calls
if 1==1: #args.read_nldm and args.read_ckt:
    
    nldm(nldm_filepath)
    delay()
    slew()
    read_ckt(circuit_filepath)

    # Create a Q and append all the input nodes to begin with the top traversal:
    for i in circuit_input_lines:
        Q.append(circuit_input_lines[i])
        tempQ.append(circuit_input_lines[i])
    while Q:
        v = Q.popleft()
        topologicaltraversal(v)

    # print('Traversal:\n', tempQ)

    max_time = backtrack()

    str_data = f"Circuit Delay: {(max_time * 1000) :.4f} ps\n\n"
    str_data = str_data + "Gate Slacks:\n"
    for val in gate_obj_dict.values():
        mystr = f"{val.name}: {val.slack * 1000 :.5f} ps"
        str_data = str_data + str(mystr) + '\n'

    str_data = str_data + f"\nCritical Path:\n{', '.join(longest_path)}"

    fn_w_circuit_file(0, 'test.txt', 'w', str_data)
    # phase-2 end
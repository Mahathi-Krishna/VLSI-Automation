from circuit_parser import *
from nldm_parser import *
from traversal import *

str_data = ""

# Main function calls
nldm()
delay()
slew()
read_ckt()

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

fn_w_circuit_file(0, circuitfile, 'w', str_data)
# phase-2 end
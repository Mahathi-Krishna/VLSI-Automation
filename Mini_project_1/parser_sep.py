from circuit_parser import *
from nldm_parser import *
from traversal import *

out_node = ""
max_time = 0.0
req_time = 0.0

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

# Find the output node with longest delay:
for gate in circuit_output_lines:
    gate = circuit_output_lines[gate]
    out_time = gate_obj_dict.get(gate).max_out_arr_time
    if out_time > max_time:
        max_time = out_time
        out_node = gate

# Start backtraversal from the output node to find the delay path:
backtraversal(out_node, max_time)
longest_path.reverse()

req_time = max_time * 1.1
print(f"Circuit Delay: {(max_time * 1000) : .4f}ps, {req_time}")
print(f"Critical Path:\n{', '.join(longest_path)}")

# Need to remove
str_data = "" + '\n\n'
for val in gate_obj_dict.values():
    mystr = val.__dict__
    str_data = str_data + str(mystr) + '\n'
fn_w_circuit_file(0, circuitfile, 'a', str_data)
# phase-2 end
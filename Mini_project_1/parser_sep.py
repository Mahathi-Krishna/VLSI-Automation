from circuit_parser import *
from nldm_parser import *
from traversal import *

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

print(f"Circuit Delay: {(max_time * 1000) : .4f} ps")
print(f"Critical Path:\n{', '.join(longest_path)}")

# Need to remove
str_data = "" + '\n\n'
for val in gate_obj_dict.values():
    mystr = val.__dict__
    str_data = str_data + str(mystr) + '\n'
    # print('name:', val.name, 'delay:', val.delay, 'in_time:', val.in_arr_time, '\nout_time:', val.out_arr_time, 'max:', val.max_out_arr_time, '\n')
fn_w_circuit_file(0, circuitfile, 'a', str_data)
# phase-2 end
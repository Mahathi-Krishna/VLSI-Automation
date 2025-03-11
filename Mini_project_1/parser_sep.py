from circuit_parser import *
from nldm_parser import *

nldm()
delay()
slew()
read_ckt()

# phase-2 start
# For topological traversal
def topsort(v):
    # Calculate delay & slew of current node
    # Update the tau_in and in_arr_time of its next neighbor in forward direction
    gate_name = v
    if(gate_name.split('-')[0] != 'INPUT'):
        a_plus_di = 0.0
        curr_path = ""
        cload = gate_obj_dict.get(gate_name).cload
        for tau in gate_obj_dict.get(gate_name).tau_in:
            tau_pin = tau
            new_path = tau_pin
            tau = gate_obj_dict.get(gate_name).tau_in.get(tau_pin)
            arr_time =  gate_obj_dict.get(gate_name).in_arr_time.get(tau_pin)
            delay, slew = get_delay_slew(gate_name, cload, tau)
            a_plus_di_new = arr_time + delay
            gate_obj_dict.get(gate_name).delay[tau_pin] = delay
            gate_obj_dict.get(gate_name).pathslew[tau_pin] = slew
            gate_obj_dict.get(gate_name).out_arr_time[tau_pin] = a_plus_di_new
            if(a_plus_di_new > a_plus_di):
                a_plus_di = a_plus_di_new
                curr_path = new_path
        gate_obj_dict.get(gate_name).max_out_arr_time = a_plus_di
        gate_obj_dict.get(gate_name).tau_out = gate_obj_dict.get(gate_name).pathslew.get(curr_path)

    # Update tau_in and in_arr_time of fan-out neighbors:
    for gate in gate_obj_dict.get(gate_name).outputs:
        gate_obj_dict.get(gate).tau_in[gate_name] = gate_obj_dict.get(gate_name).tau_out # phase-2
        gate_obj_dict.get(gate).in_arr_time[gate_name] = gate_obj_dict.get(gate_name).max_out_arr_time # phase-2
    
    # Traverse through each of the fanout gates and add them to the queue if all the input arrival times are known
    for index, gate in enumerate(gate_obj_dict.values()):
        if (gate_name in gate.inputs) and (gate.visited == 0):
            if (len(gate.inputs) == len(gate.tau_in)):
                gate.visited = 1
                Q.append(gate.name)
                tempQ.append(gate.name)

# Create a Q and append all the input nodes to begin with the top traversal:
for i in circuit_input_lines:
    Q.append(circuit_input_lines[i])
    tempQ.append(circuit_input_lines[i])
while Q:
    v = Q.popleft()
    topsort(v)

print(tempQ)

# Need to remove
str_data = "" + '\n\n'
for val in gate_obj_dict.values():
    mystr = val.__dict__
    str_data = str_data + str(mystr) + '\n'
fn_w_circuit_file(0, circuitfile, 'a', str_data)
# phase-2 end
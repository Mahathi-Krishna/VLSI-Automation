from collections import deque
from circuit_parser import *

out_node = ""
max_time = 0.0
req_time = 0.0
min_slack = 0.0
longest_path = []

# Topological traversal of nodes to update delay and slew values:
def topologicaltraversal(Q):
    # Calculate delay & slew of current node
    # Update the tau_in and in_arr_time of its next neighbor in forward direction
    
    while Q:
        v = Q.popleft()
        gate_name = v

        # If the node is already visited, do nothing:
        if (gate_obj_dict.get(gate_name).visited == 1):
            continue

        if(gate_name.split('-')[0] != 'INPUT'):
            fanin = len(gate_obj_dict.get(gate_name).inputs)
            a_plus_di = 0.0
            curr_path = ""
            cload = gate_obj_dict.get(gate_name).cload
            for tau in gate_obj_dict.get(gate_name).tau_in:
                tau_pin = tau
                new_path = tau_pin
                tau = gate_obj_dict.get(gate_name).tau_in.get(tau_pin)
                arr_time =  gate_obj_dict.get(gate_name).in_arr_time.get(tau_pin)
                delay, slew = get_delay_slew(gate_name, cload, tau)

                if(fanin > 2):
                    delay = delay * (fanin / 2)
                    slew = slew * (fanin / 2)
            
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
            if (gate_name in gate.inputs):
                if (len(gate.inputs) == len(gate.tau_in)):
                    Q.append(gate.name)
    
        # Update OUTPUT nodes:
        if(gate_name.split('-')[0] != 'INPUT' and gate_name.split('-')[1] in circuit_output_lines):
            output_node = circuit_output_lines[gate_name.split('-')[1]]
            gate_obj_dict.get(output_node).visited = 1
            gate_obj_dict.get(output_node).inputs.append(gate_name)
            gate_obj_dict.get(output_node).out_arr_time[gate_name] = gate_obj_dict.get(gate_name).max_out_arr_time
            gate_obj_dict.get(output_node).max_out_arr_time = gate_obj_dict.get(gate_name).max_out_arr_time
        
        gate_obj_dict.get(gate_name).visited = 1 # Mark current node as visited

# Backtrack from output node to input node to find the longest delay path:
def longestpath(out_node, gate_value, delay_or_slack):
    longest_path.append(out_node)

    if out_node.split('-')[0] == 'INPUT':  # Stop if input node is reached
        return

    if delay_or_slack == 'delay':  # Check with delay
        next_node = next(
            (key for key, value in gate_obj_dict.get(out_node).out_arr_time.items() if value == gate_value),
            out_node  # Default to out_node if no match is found
        )
        gate_value = gate_obj_dict.get(next_node).max_out_arr_time
        longestpath(next_node, gate_value, delay_or_slack)  # Recursive call

    elif delay_or_slack == 'slack':  # Check with slack
        next_node = None
        min_slack = float('inf')
        min_slack_node = None

        for nextnode in gate_obj_dict.get(out_node).inputs:
            nextnode_slack = round(gate_obj_dict.get(nextnode).slack, 5)

            # Primary condition: Exact slack match
            if nextnode_slack == round(gate_value, 5):
                next_node = gate_obj_dict.get(nextnode).name
                gate_value = nextnode_slack
                break  # Stop searching after an exact match

            # Secondary condition: Track node with minimum slack
            if nextnode_slack < min_slack:
                min_slack = nextnode_slack
                min_slack_node = gate_obj_dict.get(nextnode).name

        # If no exact match, explore the node with minimum slack
        if next_node is None and min_slack_node is not None:
            next_node = min_slack_node
            gate_value = min_slack

        # Recursive call if a valid next node exists
        if next_node is not None:
            longestpath(next_node, gate_value, delay_or_slack)
        else:
            print(f"Stopping recursion: No valid next node from {out_node}")

# Find the output node with longest delay:
def backtrack():
    global req_time
    global max_time
    global min_slack

    # To find max required time:
    for gate in circuit_output_lines:
        gate = circuit_output_lines[gate]
        Q.append(gate)
        out_time = gate_obj_dict.get(gate).max_out_arr_time
        if out_time > max_time:
            max_time = out_time
            out_node = gate
    req_time = max_time * 1.1
    
    # Calculate required time and slack:
    calculateslack()

    # To find min slack:
    for gate in circuit_output_lines:
        gate = circuit_output_lines[gate]
        slack = gate_obj_dict.get(gate).slack
        if slack < min_slack:
            min_slack = slack
            out_node = gate
    
    # Start from the output node with max delay to find the delay path:
    # longestpath(out_node, max_time, 'delay')

    # Start from the output node with minimum slack to find the delay path:
    longestpath(out_node, min_slack, 'slack')

    longest_path.reverse()
    return max_time

# Calculate Required time & slack details of each gate:
def calculateslack():
    while Q:
        gate = Q.popleft()
        if(gate_obj_dict.get(gate).visited == 2):
            continue

        if(gate.split('-')[0] == 'OUTPUT'):
            gate_obj_dict.get(gate).min_required_time = req_time
            gate_obj_dict.get(gate).slack = gate_obj_dict.get(gate).min_required_time - gate_obj_dict.get(gate).max_out_arr_time
            for fanin in gate_obj_dict.get(gate).inputs:
                Q.append(fanin)
        
        elif(gate.split('-')[0] not in ['INPUT','OUTPUT'] and gate.split('-')[1] in circuit_output_lines.keys()):
            gate = circuit_intermediate_outputs[gate.split('-')[1]]
            gate_obj_dict.get(gate).min_required_time = req_time
            gate_obj_dict.get(gate).slack = gate_obj_dict.get(gate).min_required_time - gate_obj_dict.get(gate).max_out_arr_time

            for fanin in gate_obj_dict.get(gate).inputs:
                if (fanin not in Q and gate_obj_dict.get(fanin).visited != 2):
                    delay = gate_obj_dict.get(gate).delay.get(fanin)
                    gate_req_time = gate_obj_dict.get(gate).min_required_time
                    gate_obj_dict.get(fanin).required_time[gate] = gate_req_time - delay
                    if len(gate_obj_dict.get(fanin).required_time) == len(gate_obj_dict.get(fanin).outputs):
                        gate_obj_dict.get(fanin).min_required_time = min(gate_obj_dict.get(fanin).required_time.values())
                        gate_obj_dict.get(fanin).slack = gate_obj_dict.get(fanin).min_required_time - gate_obj_dict.get(fanin).max_out_arr_time
                        Q.append(fanin)
        
        elif(gate.split('-')[0] not in ['INPUT','OUTPUT'] and gate.split('-')[1] in circuit_intermediate_outputs.keys()):
            gate = circuit_intermediate_outputs[gate.split('-')[1]]
            
            if len(gate_obj_dict.get(gate).required_time) == len(gate_obj_dict.get(gate).outputs):
                gate_obj_dict.get(gate).min_required_time = min(gate_obj_dict.get(gate).required_time.values())
                gate_obj_dict.get(gate).slack = gate_obj_dict.get(gate).min_required_time - gate_obj_dict.get(gate).max_out_arr_time

            for fanin in gate_obj_dict.get(gate).inputs:
                
                if (fanin not in Q and gate_obj_dict.get(fanin).visited != 2):
                    delay = gate_obj_dict.get(gate).delay.get(fanin)
                    gate_req_time = gate_obj_dict.get(gate).min_required_time

                    if (fanin.split('-')[0] == 'INPUT'):
                        gate_obj_dict.get(fanin).required_time[gate] = gate_req_time - delay
                        gate_obj_dict.get(fanin).min_required_time = min(gate_obj_dict.get(fanin).required_time.values())
                        gate_obj_dict.get(fanin).slack = gate_obj_dict.get(fanin).min_required_time - gate_obj_dict.get(fanin).max_out_arr_time

                    else:
                        delay = gate_obj_dict.get(gate).delay.get(fanin)
                        gate_req_time = gate_obj_dict.get(gate).min_required_time
                        gate_obj_dict.get(fanin).required_time[gate] = gate_req_time - delay
                        if len(gate_obj_dict.get(fanin).required_time) == len(gate_obj_dict.get(fanin).outputs):
                            gate_obj_dict.get(fanin).min_required_time = min(gate_obj_dict.get(fanin).required_time.values())
                            gate_obj_dict.get(fanin).slack = gate_obj_dict.get(fanin).min_required_time - gate_obj_dict.get(fanin).max_out_arr_time
                            Q.append(fanin)
        
        gate_obj_dict.get(gate).visited = 2 # Mark current node as visited
from circuit_parser import *
from nldm_parser import *


nldm()
delay()
slew()
read_ckt()

# phase-2 start
# For topological traversal
# def topsort(v):
#     for index, gate in enumerate(gate_obj_list):
#         if (v in gate.inputs) and (gate.visited == 0) :
#             gate.visited = 1
#             Q.append(gate.outname)
#             tempQ.append(gate.outname)

# for i in circuit_input_lines:
#     Q.append(i)
#     tempQ.append(i)

# while Q:
#     v = Q.popleft()
#     topsort(v)

# print(tempQ)

# for i in gate_obj_list:
#     print(i.name, i.inputs, i.cin, i.outputs, i.cload)
#    print(i.__dict__)
# phase-2 end
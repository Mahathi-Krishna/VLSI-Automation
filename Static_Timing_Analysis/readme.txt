VLSI Automation - Mini Project 1 - Static Timing Analysis:
The objective of this project is to compute the given circuit's characteristics using python.

Description:
The project consists of:
*.lib file - contains the properties of each gate i.e. load capacitance, delay and slew matrices.
*.bench file - contains the circuit/netlist details i.e. inputs, outputs and gates required for the circuit.
main_sta.py file reads the .lib and .bench files and outputs a file depending on the function called containing the requested details.

The following are the functions of interest:
--read_ckt - reads the *.bench file and writes a "ckt_details.txt" file with the circuit's delay, slack values and the longest delay path.
--read_nldm - reads the .lib file and writes:
    1. "delay_LUT.txt" file which contains the delay values along with the delay matrix for each gate.
    2. "slew_LUT.txt" file which contains the input delay, slew values along with the slew matrix for each gate.

Note: Please make sure that the files are present in the same directory as the main_sta.py file.
If files are in a different directory, then pass the file path as argument in the command line.

Getting Started:
1. Install python version 3.7 or later.

2. Install virtual environment:
>> pip install virtualenv

3. Create a virtual environment for the project:
>> python3.7 -m venv <name of venv>

4. Activate the environment by sourcing it:
>> source <name of venv>/bin/activate

5. Install the required packages specified in the "requirements.txt" file:
>> pip3 install -r requirements.txt

6. Run the main_sta.py file:
syntax: python3.7 main_sta.py --read_ckt <.bench file> --read_nldm sample_NLDM.lib
>> python3.7 main_sta.py --read_ckt c17.bench --read_nldm sample_NLDM.lib


An original work of:
Copyright © 2025 by Mahathi Krishna Ravi Shankar and Vishnu Ram Jawaharram. All rights reserved.
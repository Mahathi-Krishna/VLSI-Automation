VLSI Automation - Mini Project 2 - IR Drop Solver:
The objective of this project is to predict the IR Drop of a design using machine learning approach.
Trained a UNet model on 100 datapoints and used the model to predict the IR Drop of a new testbench.


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

6. To generate the four CSV files (Current Map, Voltage Source Map, PDN Density Map, IR Drop Map) for a netlist:
Syntax: py data_generation.py -spice_netlist <path to spice file> -voltage_file <path to voltage file> –output <path to directory where four csvs are saved.>
>> py data_generation.py -spice_netlist ./Datapoints/testcase1.sp -voltage_file ./Datapoints/testcase1.voltage –output ./CSV_Files

7. To train the machine learning model:
Syntax: py training.py -input <path to directory .sp and .voltage files> -output <path to ML model will be saved its name>
>> py training.py -input ./Datapoints -output ./models

8. To predict the IR Drop of a netlist using the saved ML model:
Syntax: py inference.py -spice_file <path to spice file> -ml_model <path to ML model> -output <path to generated ir_drop_map>
>> py inference.py -spice_file ./Benchmarks/testcase1.sp -ml_model ./models/unet.pth -output ./Prediction


Note:
In (7): It is assumed that voltage files are stored with a .voltage extension and in the same folder as the .sp files.
In (7): The script generates the csv files for all the datapoints in -input path and then starts training the model.
In (8): -output only takes the path, no need to give the file name.


An original work of:
Copyright © 2025 by Mahathi Krishna Ravi Shankar and Vishnu Ram Jawaharram. All rights reserved.
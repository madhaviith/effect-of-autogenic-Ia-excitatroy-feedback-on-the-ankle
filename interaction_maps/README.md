## Automatic generation of spinal connectome using curated data of spinal neurons, conection rules, muscles information. Refer figure 2 of the manuscript and supplementary-A for the algorithm and definitions of inputs(.xlsx sheets). 

This folder contains the codes to generate ipsilateral spinal connectome from the curated **spinal_neurons.xlsx, connection_rules.xlsx, movement_types.xlsx, muscles.xlsx** and generates figure in **figure 5** of the mauscript that describes the spinal afferent based muscle-muscle interactions.  

### Software and hardware requirements:

2. Linux desktop (preferrably installed with Ubuntu)

Both code development and testing was carried out on **Ubuntu-18.04.6-LTS desktop with intel-core i7 CPU and NVIDIA quadro P1000 graphics and 16GB RAM space**. 

The above hardware requirements are for the indicative purposes and any desktop latest versions of ubuntu should be compatible with the current codes. We have not tested on the other linux platforms  like mint os or cent os etc. But the presented codes should run if the compatible python software and required libraries are installed. 

1. Python 3.7 

Required libraries:  pandas, numpy, os, seaborn, matplotlib, json, openpyxl 


### Run instructions:

##### Step1: Open the terminal and run interactions_maps.py code as shown below

~$ python iteraction_maps.py 

##### Step2: wait for the comletion of code and graph windows to pop-up.  

Refer readme.pdf for the same intructions 

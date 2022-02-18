#########################################################################################################
#  This code is part of project NEUROiD developed at Spine Labs
#  Dept. of BME,
#  IIT Hyderabad
#  www.iith.ac.in/~mohanr
#
#  Revision History
#  -----------------------
#   Name             Date            Changes
#   ---------       -------     ----------------------
#   Raghu S Iyengar  Aug 2018      Initial Creation
#   Madhav Vinodh     2020         1. redesigned the code to handle multiple muscles based on the osim model and number of msucles 
#                                  2. implemented find and search based afferent feedback calculation, 
#                                  3. new function to send the activations to the actuators,   
#
#############################################################################################################################################
# The file implements a glue layer between neuroid and OpenSim
# This file implements a TCP server that waits for a client connection
# main.py code that performs the NEURON simulation acts as client and
# connects to this server to perform one step of OpenSim simulation
#
# This script loads the OpenSim model, initializes and waits for client
#  message to peform every step of simulation
# 
# Input:
# 1. lower limb model (/lower_limb_2010/ArnoldHamner_LegsTorsoArms_v2.1.osim)
# 2. osim_neuroid_map- to connect the muscle proprioceptors with afferent neurons  (../osim_musc_drg_map.xlsx)
# 3. activations_dict - received from the main.py client at every time step
#
# Output:
# 1. muscle_lengths.xlsx, joint_angles.xlsx, afferent_firing.xlsx, muscles_velocity.xlsx files are written to the output sumulation directory once the code is executed
# 2. firings_to_neuroid - sends afferent firing to main.py at every time step
#
##################################################################################################################################################
import numpy as np
import sys,math
import json, socket
import opensim as osim
from matplotlib import pyplot as plt
import os
import pandas as pd

cur_path = os.getcwd()
print(cur_path)

#server-socket
def create_neuroid_socket():
    """
    function: create_neuroid_socket

    Creates socket and starts the server
    """
    s = socket.socket()         # Create a socket object
    host = socket.gethostname() # Get local machine name
    print(host)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = 50003               # Reserve a port for your service.
    s.bind((host, port))        # Bind to the port
    s.listen(5)                 # Now wait for client connection.
    return s

# function to receve muscle activation value from the spinal motoneurons
def read_act_from_neuroid(s):
    """
    function: read_act_from_neuroid

    Accept client connection from neuroid and read muscle activations
    """
    c, addr = s.accept()     # Establish connection with client.
    jsstr = c.recv(102400)
    #print jsstr
    jsmsg = jsstr.decode('utf-8')
    #return c,jsmsg['flex']['act'], jsmsg['ext']['act']
    jsmsg = eval(jsmsg)
    return c,jsmsg

#function to send calculated affrent firing to the spinal neuronal network 
def write_afferent_to_neuroid(c,firing_to_neuroid_dict):
    jsmsg = str(firing_to_neuroid_dict)
    #c.send(json.dumps(jsmsg))
    c.send(jsmsg.encode('utf-8'))
    c.close()

#function to extract activations dict received from neuroid into a list 
def extract_muscle_ids(muscle_activations):
    muscles_ids = []
    for each in muscle_activations: 
        muscles_ids.append(each["id"])
    return muscles_ids
    
def extract_muscle_acts(muscle_ids):
    muscles_activations = []
    for each in muscle_ids: 
        muscles_activations.append(each["act"])
    return muscles_activations

#function to extract the name strings of the osim muscles from the ids extracted from the activations_dict
def extract_muscles_names(osim_muscles,muscles_ids):    
    neuroid_muscles_list = []
    for each in muscles_ids: 
        neuroid_muscles_list.append(osim_muscles.get(each).getName())
    return neuroid_muscles_list

#pass the list of the joint_indices that need to be locked
def lock_joints(model,joint_indices_list):
    joints = model.getJointSet()
    for joint_index in joint_indices_list:
        joint = joints.get(joint_index)
        for t in range(joint.numCoordinates()):
            joint.get_coordinates(t).setLocked(state,True) 
        print(joints.get(joint_index).getName(),"is locked")

#pass the list of  muscle activation received from neuroid in the following  


#fucntion to set the activation value directly by calling the prescribeControlForActuator module of the brain <brain is PrescribedController>
#format [{'id': osim_musle_id, 'act' : muscle_activation_value_from_neuroid}]
def set_activation_value1(muscl_activation):
    """
    function: actuate
      model - instance of OpenSim model
      controller - instance of Controller for OpenSim model
      action - activation for flexor and extensor sent as list of 2 elements

    This function sets the activation for flexor and extensor muscles in
     OpenSim model
    """
    for each in muscl_activation:
        muscle_id = each['id']
        muscle_act = float(each['act'])
        func2 = osim.Constant(muscle_act)
        brain.prescribeControlForActuator(muscle_id,func2)
                
def update_muscles_params(osim_state,osim_muscles,muscles_current_l,muscles_force,muscles_vel,muscles_act,muscles_id,iterator):
    for each in muscles_id:
        osim_muscle_name = osim_muscles.get(each).getName()
        muscles_current_l.loc[iterator,osim_muscle_name] = osim_muscles.get(each).getFiberLength(osim_state)
        muscles_force.loc[iterator,osim_muscle_name]     = osim_muscles.get(each).getActiveFiberForce(osim_state) #important change to calculate force
        muscles_act.loc[iterator,osim_muscle_name]       = osim_muscles.get(each).getActivation(osim_state)
        muscles_vel.loc[iterator,osim_muscle_name]       = osim_muscles.get(each).getFiberVelocity(osim_state)
        
#input is osim-state and osim_model_muscles, and muscles_ids
#update only those muscles that are querried for via osim_drg map
#input# osim_state, osim_muscles, muscles_id = list of only requested muscles,
#<can be deprecated>
def update_muscle_params(osim_state,osim_model_muscles,muscles_id):
    #create empty dataframes for length, velocity and force
    muscle_current_l = pd.DataFrame(0,index=[0],columns = muscles_list)
    muscle_vel = pd.DataFrame(0,index=[0],columns = muscles_list)
    muscle_force = pd.DataFrame(0,index=[0],columns = muscles_list)
    for each in muscles_id:
        osim_muscle_name = osim_muscles.get(each).getName()
        muscle_current_l.loc[:,osim_muscle_name] = osim_muscles.get(each).getFiberLength(osim_state)
        muscle_vel.loc[:,osim_muscle_name] = osim_muscles.get(each).getFiberVelocity(osim_state)
        muscle_force.loc[:,osim_muscle_name] = osim_muscles.get(each).getFiberForce(osim_state) #verify this later
    return muscle_current_l, muscle_vel, muscle_force 

#evaluate afferent feedback based on the afferent neuron tag
def eval_aff_feedback(muscles_current_l,osim_drg_map,iterator):
        
    def cal_IaAfferent(muscle_velocity,muscle_stretch,base_firing):
        Ia_firing = base_firing + 2*(muscle_stretch) + (4.3*((muscle_velocity)**0.6)) 
        #generic formula
        #Ia_firing = base_firing + muscle_stretch + (muscle_velocity)**0.6
        return int(Ia_firing)
    
    def cal_IIAfferent(muscle_stretch,base_firing):
        II_firing = base_firing + (muscle_stretch)
        return int(II_firing)
    
    def cal_IbAfferent(mus_force,mus_max_iso_force,base_firing):
        #Ib_firing = 200 *(musc_activ*(max_iso_force + (56.3*(muscle_stretch)) + (2.81*muscle_velocity))/max_iso_force) 
        
        
        #generic formula 
        Ib_firing = base_firing * (mus_force/mus_max_iso_force) 
        print(mus_force, mus_max_iso_force, Ib_firing)
        return int(Ib_firing)
    
    
    #function finds and evaluates affernt firing rate when a drg_neuron and all muscle_params are given
    def find_cal_aff_firing(target_drg_neuron,source_muscle,iterator):  
        #Pool the tags into a list
        tag_list = target_drg_neuron.split("_") 
        #muscle_length,muscle_vel and maximum isometric force for a particular muscle
        mus_max_iso_force   = muscles_iso_force.loc[0,source_muscle]
        mus_force = muscles_force.loc[iterator,source_muscle]
        mus_vel = muscles_vel.loc[iterator,source_muscle]
        mus_opti_l   = muscles_opt_l.loc[0,source_muscle]
        #musc_activ      = muscles_act.loc[iterator,source_muscle]
        mus_cur_len  = muscles_current_l.loc[iterator,source_muscle]
        mus_str        = mus_cur_len - mus_opti_l   
        mus_stretch  = 1000*(max(0,mus_str)) # considering only stretch not shortening
        
        #to consider only stretch velocity for caluclation of Ia afferent firing
        if mus_vel < 0:
            mus_vel_val = 0
        
        else: 
            mus_vel_val = 1000*mus_vel 
            
        #to consider only ocntraction force for caluclation of Ia afferent firing
        
        if mus_force < 0:
            mus_force_val = 0
        else: 
            mus_force_val = mus_force
        
        #identify any of the following in the tag and calculate the appropriate firing
        if (target_drg_neuron.find("Ia") != -1) :
            base_firing = 0

            firing = cal_IaAfferent(mus_vel_val,mus_str,base_firing)
            
        if (target_drg_neuron.find("II") != -1) :
            base_firing = 10
            firing = cal_IIAfferent(mus_str,base_firing)
            
        if (target_drg_neuron.find("Ib") != -1) :
            base_firing = 120
            firing = cal_IbAfferent(mus_force_val,mus_max_iso_force,base_firing)
            print(firing)
            
        return firing
        #muscles_prev_l = muscles_current_l.loc[0,source_muscle]
        
        
    #create a aff_firing_df with empty 'firing_in_Hz' column and that is eventually will be sent to neuroid 
    #extract only those drg with proprioception 'on' cases
    firing_to_neuroid = osim_drg_map.loc[osim_drg_map['proprioception']=='on'].copy()
    firing_to_neuroid["firing_in_Hz"] = np.NaN
    firing_to_neuroid = firing_to_neuroid[['target_drg','muscle_name','firing_in_Hz']]
    drgs = firing_to_neuroid['target_drg'].tolist()
    if firing_to_neuroid.shape[0] != 0:
        for each in drgs:
            target_drg_neuron = firing_to_neuroid[firing_to_neuroid['target_drg'] == each]['target_drg'].tolist()[0]
            source_muscle = firing_to_neuroid[firing_to_neuroid['target_drg'] == each]['muscle_name'].tolist()[0]
            firing = find_cal_aff_firing(target_drg_neuron,source_muscle,j)
            #update firings to dataframe 
            firings_dat.loc[iterator,target_drg_neuron] = firing
            #use this to send to neuroid
            firing_to_neuroid.loc[firing_to_neuroid['target_drg'] == each ,'firing_in_Hz'] = firing  

    return firing_to_neuroid 

def update_joint_vals(osim_state,joint_name,joint_id,joint_angles_dict):
    joint = joints.get(joint_id) 
    if joint.numCoordinates() == 1:
        joint_val = joint.getCoordinate().getValue(state)
        joint_angles_dict[joint_name].append(joint_val)
    else:
        #store multi coordinate values
        #variable_dict = {}
        for j in range(joint.numCoordinates()):
            joint_val = joint.get_coordinates(j).getValue(state)
            jnt_name = joint_name + "_" + str(j)
            joint_angles_dict[jnt_name].append(joint_val)


#load osim muscles and their default control here
load_osim_musc = pd.read_excel(cur_path + "/../osim_muscles.xlsx",sheet_name='osim_muscles')
neuroid_osim_muscles=load_osim_musc['muscle_name']

#load osim joints and their status here 
load_osim_joints = pd.read_excel(cur_path + "/../osim_muscles.xlsx",sheet_name='osim_joints')

#load_osim_neuroid map to map the muscle proprioceptors with the afferent neurons 
load_osim_drg_map = pd.read_excel(cur_path + "/../osim_musc_drg_map.xlsx",sheet_name="osim_musc_drg_map")

#load osim model
cwd = os.getcwd()
model_name = "lower_limb_2010/ArnoldHamner_LegsTorsoArms_v2.1.osim"
model = osim.Model(model_name)

#set gravity to zero
gravity_vec = osim.Vec3(0,0,0)
model.setGravity(gravity_vec)

model.setUseVisualizer(True)
joints = model.getJointSet()
muscles= model.getMuscles()

#create a prescribed controller 
brain = osim.PrescribedController()

# Add the controller to the model
model.addController(brain)
muscles_list = []
for j in range(model.getMuscles().getSize()):    
            func = osim.Constant(0.0)
            muscles_list.append(model.getMuscles().get(j).getName())
            #print("muscle index: {} and muscle name :{}".format(j,model.getMuscles().get(j).getName()))
            brain.addActuator(model.getMuscles().get(j))
            #set muscle activations to zero in the bigenning
            brain.prescribeControlForActuator(j, func)

#empty dataframes to store values of muscle's previous lengths, velocity, force, 
#all dataframes are of shape 1xlen(osim.model.muscles)
#optimum length(optimal_fibre_length) is constant param of a particular muscles 
#each muscles optimum lenghts initialized as zero dataframe

muscles_iso_force = pd.DataFrame(0,index=[0], columns = muscles_list)
muscles_opt_l = pd.DataFrame(0,index=[0],columns = muscles_list)

#fill the iso_force dataframe with the maximum isometric force of each muscle
for i in range(muscles.getSize()):
    mus_name = muscles.get(i).getName()
    muscles_iso_force.loc[:,mus_name] = muscles.get(i).get_max_isometric_force()
    muscles_opt_l.loc[:,mus_name]     = muscles.get(i).get_optimal_fiber_length()

#empty dataframes to store values of muscles previous lengths, velocity, force during simulation
#all dataframes are of shape 1xlen(osim.model.muscles)
muscles_prev_l = pd.DataFrame(0,index=[0],columns = muscles_list)
muscles_current_l = pd.DataFrame(0,index=[0],columns = muscles_list)
muscles_velocity = pd.DataFrame(0,index=[0],columns = muscles_list)
muscles_vel = pd.DataFrame(0,index=[0],columns = muscles_list)
muscles_force = pd.DataFrame(0,index=[0],columns = muscles_list)
muscles_act = pd.DataFrame(0,index=[0],columns = muscles_list)
firings_dat = pd.DataFrame(0,index=[0],columns = load_osim_drg_map['target_drg'].to_list())

#load the requested drgs, source_opensim_muscles, opensim_muscle_ids to the lists from osim_drg_map
get_osim_muscle_list = load_osim_drg_map.muscle_name.unique().tolist()
get_target_drg_list = load_osim_drg_map.target_drg.unique().tolist()
get_osim_muscle_id_list = load_osim_drg_map.index_number.unique().tolist() #obtain this list from xlsx

# fxn to update the parameters of all muscles (irrespective of the requested muscles) in loop
#input #empty dataframes: muscles_current_l, muscles_vel, muscles_force and osim_state, muscles_id = list of all osim_muscle_ids
#output# updated dataframes 


#initialization
state = model.initSystem()

#set posture II (refer supplementary-B )
joints.get(10).get_coordinates(0).setValue(state,0.1*math.pi) #hip joint left   
joints.get(12).get_coordinates(0).setValue(state,0.2*math.pi) #knee joint left
joints.get(0).get_coordinates(0).setValue(state,0.5*math.pi) # make it to lie on ground supine


model.equilibrateMuscles(state)

#obtain indeces of the joints to be locked from the user 
joint_indices_list = load_osim_joints.loc[load_osim_joints['joint_lock_state'] =='yes']['index_number'].tolist()

#joint names to track
joint_names_list = load_osim_joints.loc[load_osim_joints['joint_lock_state'] =='no']['joint_name'].tolist()
print(joint_names_list)
#print joint_indices_list
#print load_osim_joints.loc[load_osim_joints['joint_lock_state'] =='no']['index_number'].tolist()
#extended joint angles dict
#empty lists for each type of joints
joint_angles_dict = {}
for each in joint_names_list:
    num_coord = load_osim_joints.loc[load_osim_joints['joint_name'] == each]['num_coordinates'].tolist()[0]
    #print num_coord
    if num_coord > 1:
        for i in range(num_coord):
            joint_angle_name = each + "_" + str(i)
            joint_angles_dict[joint_angle_name] = []
    if num_coord == 1:
        joint_angles_dict[each] = []

#lock joints 
lock_joints(model,joint_indices_list)

#add model to manager
manager = osim.Manager(model)
manager.setIntegratorMethod(osim.Manager.IntegratorMethod_Verlet)
manager.setIntegratorAccuracy(3e-2)

#initialize the state
manager.initialize(state)


total_time_ms = 150.0 #in milli seconds
total_time_s = total_time_ms/1000.0
print(total_time_ms)
time_iterator = 0.0
stepsize = 0.05/1000*5

n_socket = create_neuroid_socket()
firings_to_df = pd.DataFrame()
time_array = np.linspace(0,total_time_s,int(total_time_s/stepsize))
#print time_array[2000]
firings = []
j=0 #iterator
# Wait for client connection in a loop

#empty dataframes to track variables
joint_angles = pd.DataFrame(0,index = time_array, columns = joint_names_list)
joint_angles = joint_angles.astype(float)
#print joint_angles.loc[time_array[1999],'ankle_l']
#(time_iterator<total_time_s)

while (time_iterator<total_time_s):
    # Get Activations from NEUROiD
    print(j, time_iterator, total_time_s, type(time_iterator), type(total_time_s))
    if j >= int(total_time_s/stepsize):
        print("yes")
        break
    else: print("no....")
    
    #conn, activation_dict = read_act_from_neuroid(n_socket)
    conn, activations_dict = read_act_from_neuroid(n_socket)

        
    #Actuate the muscles in OpenSim model and advance the simulation 
    set_activation_value1(activations_dict)
    
    #set time step
    state.setTime(j*stepsize)
   
    #integrate
    state = manager.integrate(stepsize*(j+1))
 
    #assign the osim_drg_map
    osim_drg_map = load_osim_drg_map.copy() 
    #print osim_drg_map[osim_drg_map['proprioception']=='on']
    
    #extract muscle_ids and muslce names from the muscle_activation dict sent from the neuroid
    muscles_ids = extract_muscle_ids(activations_dict)
    muscles_activations = extract_muscle_acts(activations_dict)
    neuroid_muscles_list = extract_muscles_names(muscles,muscles_ids)
    
       
    ##update the muscles parameters at each time step
    update_muscles_params(state,muscles,muscles_current_l,muscles_force,muscles_vel,muscles_act,muscles_ids,j)
    for each in joint_names_list:
            joint_var_id =  load_osim_joints.loc[load_osim_joints['joint_name'] == each]['index_number'].tolist()[0]
            #print "cord Id", joint_var_id
            if joint_var_id == 0:
                pass
            else:
                update_joint_vals(state,each,joint_var_id,joint_angles_dict)

    #calculate the afferent feedback                      
    aff_feedback = eval_aff_feedback(muscles_current_l,osim_drg_map,j)
    
    #firings_to_df = firings.append(aff_feedback)
    firings_to_neuroid = aff_feedback.to_dict(orient='dict') 
    
    #firings.append(firings_to_neuroid)
    print(firings_to_neuroid['firing_in_Hz'])

    ##send the afferent feedback to neuroid
    ##handle all proprioception "off" cases where the firings_to_neuroid will render an empty dataframe 
    write_afferent_to_neuroid(conn,firings_to_neuroid['firing_in_Hz'])
    
    j = j+1
    time_iterator = time_iterator + stepsize
    
print("while_loop finished") 

#push the osim variables to the designated folders   
#get the current simulation folder and push the data to that simulation folder

#muscle_max_iso_force
max_iso_force_file_name = cur_path + '/output/mus_max_iso_force.xlsx'
muscles_iso_force.to_excel(max_iso_force_file_name)


#muscle force to xlsx
forces_file_name = cur_path + '/output/muscle_forces.xlsx'
muscles_force.to_excel(forces_file_name)


#muscle length to xlsx
lengths_file_name = cur_path + '/output/muscle_lengths.xlsx'
muscles_current_l.to_excel(lengths_file_name)

#joint_angles to xlsx
joint_angles_df = pd.DataFrame.from_dict(joint_angles_dict, orient  = 'index')
joint_angles_df = joint_angles_df.T
joints_file_name = cur_path + '/output/joint_angles.xlsx'
joint_angles_df.to_excel(joints_file_name)

#write the firing dataframe to a xlsx sheet
firings_to_df = pd.DataFrame(firings)
firings_file_name = cur_path + '/output/afferent_firing.xlsx'
firings_dat.to_excel(firings_file_name)

#write the velocity dataframe to a xlsx sheet
velocity_file_name = cur_path + '/output/muscles_velocity.xlsx'
muscles_velocity.to_excel(velocity_file_name)

#write the velocity dataframe to a xlsx sheet
vel_file_name = cur_path + '/output/muscles_vel.xlsx'
muscles_vel.to_excel(vel_file_name)



  

print("end of code execution")
 




#########################################################################################################################################################
#This code imports the output data(muscle lenghts, muscle actiavtions, joint angles, afferent feedback, voltage traces) from the simulation folders
#
#And plots the data
#
#The data plotted are :
#    
#1. Ankle angles due ta stimulation during simulations without feedback vs autogenic Ia excitatory feedback
#2. Muscle activation of ta, Sol, gasmed, gaslat muscles due tp ta stimulation during simulations without feedback vs autogenic Ia excitatory feedback
#3. Voltages of segmental motoneurons of ta, sol, gasmed, gaslat due ta stimulation during simulations without feedback vs autogenic Ia excitatory feedback 
#4. Segmental afferent activity due to ta, sol, gasmed, gaslat due to ta stimualtion msucles during simulations without feedback vs autogenic Ia excitatory feedback 
#5. Muscle lenghts changes of ta, sol, gaslat, gasmed due to ta stimulation during simulations without feedback vs autogenic Ia excitatory feedback
#####################################################################################################################################################################

import os, json
import pandas as pd
import numpy as np
import math
import fnmatch
import matplotlib.pyplot as plt
cwd = os.getcwd()



# set open loop and close loop simulation folders
o_folname = cwd + "/sim_o/"
c_folname = cwd + "/sim_c/"

# load data open loop 
o_muscle_act_dat = pd.read_excel(o_folname + "/muscle_act_data_.xlsx",index_col=0 )
o_muscle_lengths = pd.read_excel(o_folname + "/muscle_lengths.xlsx",index_col=0)
o_joint_angles = pd.read_excel(o_folname + '/joint_angles.xlsx',  index_col=0)
o_time = np.load(o_folname + "/save_time.npy")
o_aff_firing = pd.read_excel(o_folname + "/afferent_firing.xlsx",index_col=0 )

# load data close loop (Ia-autogenic excitation)
c_muscle_act_dat = pd.read_excel(c_folname + "/muscle_act_data_.xlsx",index_col=0 )
c_muscle_lengths = pd.read_excel(c_folname + "/muscle_lengths.xlsx",index_col=0)
c_joint_angles = pd.read_excel(c_folname + '/joint_angles.xlsx',  index_col=0)
c_time = np.load(c_folname + "/save_time.npy")
c_aff_firing = pd.read_excel(c_folname + "/afferent_firing.xlsx",index_col=0 )



#Add time to joint angles dataframe and convert them to degrees
#open_loop data
o_joint_angles['time in milliseconds'] = o_time[0::5]
o_joint_angles.set_index(['time in milliseconds'], inplace=True)
o_joint_angles_deg = o_joint_angles.multiply(57.2958)

#close loop data
c_joint_angles['time in milliseconds'] = c_time[0::5]
c_joint_angles.set_index(['time in milliseconds'], inplace=True)
c_joint_angles_deg = c_joint_angles.multiply(57.2958)


#new data frame

df = pd.DataFrame(columns=['no_feedback','autogenic Ia feedback'])
df['time in ms'] = o_joint_angles_deg.index
df.set_index(['time in ms'], inplace = True)
df['autogenic Ia feedback'] = c_joint_angles_deg['ankle_l'].values
df['no_feedback'] = o_joint_angles_deg['ankle_l'].values

#plot joint angles
ax0 = df.plot(xlim=(0,150),grid=True) #ylim=(-40,20)
ax0.set_ylabel('Plantarflexion <------ Ankle angle(in deg) ------> Dorsiflexion ', fontweight='bold')
ax0.set_xlabel('time in ms', fontweight='bold')
ax0.set_title('joint angle')


#plot muscle activations 
fig1, ax1 = plt.subplots(2,2,figsize=(5,4))
fig1.suptitle('Muscle activations', fontsize=16)

ax1[0][0].plot(o_time,o_muscle_act_dat['tibant_l'],label='no feedback')
ax1[0][0].plot(c_time,c_muscle_act_dat['tibant_l'],label='autogenic Ia feedback')
ax1[0][0].set_ylim(-0.1,1)
ax1[0][0].legend(loc="upper right")
ax1[0][0].set_xlabel("time in ms",fontweight='bold')
ax1[0][0].set_ylabel("muscle activation(0-1)",fontweight='bold')
ax1[0][0].set_title("ta", fontweight='bold')
ax1[0][0].set_xticks(np.arange(0,150,20))


ax1[0][1].plot(o_time,o_muscle_act_dat['soleus_l'],label='no feedback')
ax1[0][1].plot(c_time,c_muscle_act_dat['soleus_l'],label='autogenic Ia feedback')
ax1[0][1].set_ylim(-0.1,1)
ax1[0][1].legend(loc="upper right")
ax1[0][1].set_xlabel("time in ms",fontweight='bold')
ax1[0][1].set_ylabel("muscle activation(0-1)",fontweight='bold')
ax1[0][1].set_title("sol", fontweight='bold')
ax1[0][1].set_xticks(np.arange(0,150,20))

ax1[1][0].plot(o_time,o_muscle_act_dat['gaslat_l'],label='no feedback')
ax1[1][0].plot(c_time,c_muscle_act_dat['gaslat_l'],label='autogenic Ia feedback')
ax1[1][0].set_ylim(-0.1,1)
ax1[1][0].legend(loc="upper right")
ax1[1][0].set_xlabel("time in ms",fontweight='bold')
ax1[1][0].set_ylabel("muscle activation(0-1)",fontweight='bold')
ax1[1][0].set_title("gaslat", fontweight='bold')
ax1[1][0].set_xticks(np.arange(0,150,20))

ax1[1][1].plot(o_time,o_muscle_act_dat['gasmed_l'],label='no feedback')
ax1[1][1].plot(c_time,c_muscle_act_dat['gasmed_l'],label='autogenic Ia feedback')
ax1[1][1].set_ylim(-0.1,1)
ax1[1][1].legend(loc="upper right")
ax1[1][1].set_xlabel("time in ms",fontweight='bold')
ax1[1][1].set_ylabel("muscle activation(0-1)",fontweight='bold')
ax1[1][1].set_title("gasmed", fontweight='bold')
ax1[1][1].set_xticks(np.arange(0,150,20))


#plot voltages of close loop simulations
voltage_dict_c = {}
#mypath = os.getcwd() 
#print mypath
pattern = '*_voltage_*.npy'
files = os.listdir(o_folname) 
#print files
for each in files:
    if fnmatch.fnmatch(each, pattern):
        name = each.strip("save_")
        name = name.strip("0_") 
        var_name = name.strip(".npy")
        voltage_dict_c[var_name]= np.load(c_folname + each)
        
#plot voltages in on graph 
fig2, ax2 = plt.subplots(9,figsize=(5,4))
fig2.suptitle('voltages of segmental motoneurons with autogenic Ia feedback', fontsize=8,fontweight='bold' )

ax2[0].plot(c_time,voltage_dict_c[voltage_dict_c.keys()[0]][:,0]),#label=voltage_dict_c.keys()[0])
ax2[0].set_title(voltage_dict_c.keys()[0],fontsize = 'xx-small',fontweight='bold')
ax2[1].plot(c_time,voltage_dict_c[voltage_dict_c.keys()[1]][:,0])
ax2[1].set_title(voltage_dict_c.keys()[1],fontsize = 'xx-small',fontweight='bold')
ax2[2].plot(c_time,voltage_dict_c[voltage_dict_c.keys()[2]][:,0])
ax2[2].set_title(voltage_dict_c.keys()[2],fontsize = 'xx-small',fontweight='bold')
ax2[3].plot(c_time,voltage_dict_c[voltage_dict_c.keys()[3]][:,0])
ax2[3].set_title(voltage_dict_c.keys()[3],fontsize = 'xx-small',fontweight='bold')
ax2[4].plot(c_time,voltage_dict_c[voltage_dict_c.keys()[4]][:,0])
ax2[4].set_title(voltage_dict_c.keys()[4],fontsize = 'xx-small',fontweight='bold')
ax2[5].plot(c_time,voltage_dict_c[voltage_dict_c.keys()[5]][:,0])
ax2[5].set_title(voltage_dict_c.keys()[5],fontsize = 'xx-small',fontweight='bold')
ax2[6].plot(c_time,voltage_dict_c[voltage_dict_c.keys()[6]][:,0])
ax2[6].set_title(voltage_dict_c.keys()[6],fontsize = 'xx-small',fontweight='bold')
ax2[7].plot(c_time,voltage_dict_c[voltage_dict_c.keys()[7]][:,0])
ax2[7].set_title(voltage_dict_c.keys()[7],fontsize = 'xx-small',fontweight='bold')
ax2[8].plot(c_time,voltage_dict_c[voltage_dict_c.keys()[8]][:,0])
ax2[8].set_title(voltage_dict_c.keys()[8],fontsize = 'xx-small',fontweight='bold')
ax2[8].set_xlabel("time in ms",fontweight='bold')



#plot voltages open loop simulation
voltage_dict_o = {}
#mypath = os.getcwd() 
#print mypath
pattern = '*_voltage_*.npy'
files = os.listdir(o_folname) 
#print files
for each in files:
    if fnmatch.fnmatch(each, pattern):
        name = each.strip("save_")
        name = name.strip("0_") 
        var_name = name.strip(".npy")
        voltage_dict_o[var_name]= np.load(o_folname + each)
        
#plot voltages in on graph 
fig3, ax3 = plt.subplots(9,figsize=(5,4))
fig3.suptitle('voltages of segmental motoneurons without feedback', fontsize=8,fontweight='bold' )

ax3[0].plot(o_time,voltage_dict_o[voltage_dict_o.keys()[0]][:,0]),#label=voltage_dict_c.keys()[0])
ax3[0].set_title(voltage_dict_o.keys()[0],fontsize = 'xx-small',fontweight='bold')
ax3[1].plot(o_time,voltage_dict_o[voltage_dict_o.keys()[1]][:,0])
ax3[1].set_title(voltage_dict_o.keys()[1],fontsize = 'xx-small',fontweight='bold')
ax3[2].plot(o_time,voltage_dict_o[voltage_dict_o.keys()[2]][:,0])
ax3[2].set_title(voltage_dict_o.keys()[2],fontsize = 'xx-small',fontweight='bold')
ax3[3].plot(o_time,voltage_dict_o[voltage_dict_o.keys()[3]][:,0])
ax3[3].set_title(voltage_dict_o.keys()[3],fontsize = 'xx-small',fontweight='bold')
ax3[4].plot(o_time,voltage_dict_o[voltage_dict_o.keys()[4]][:,0])
ax3[4].set_title(voltage_dict_o.keys()[4],fontsize = 'xx-small',fontweight='bold')
ax3[5].plot(o_time,voltage_dict_o[voltage_dict_o.keys()[5]][:,0])
ax3[5].set_title(voltage_dict_o.keys()[5],fontsize = 'xx-small',fontweight='bold')
ax3[6].plot(o_time,voltage_dict_o[voltage_dict_o.keys()[6]][:,0])
ax3[6].set_title(voltage_dict_o.keys()[6],fontsize = 'xx-small',fontweight='bold')
ax3[7].plot(o_time,voltage_dict_o[voltage_dict_o.keys()[7]][:,0])
ax3[7].set_title(voltage_dict_o.keys()[7],fontsize = 'xx-small',fontweight='bold')
ax3[8].plot(o_time,voltage_dict_o[voltage_dict_o.keys()[8]][:,0])
ax3[8].set_title(voltage_dict_o.keys()[8],fontsize = 'xx-small',fontweight='bold')
ax3[8].set_xlabel("time in ms",fontweight='bold')



#load affrent feedback data

Ia_aff_c = c_aff_firing.columns.tolist()[0:9]
Ia_aff_o = o_aff_firing.columns.tolist()[0:9]


#plot afferent firing data of close loop 

fig4, ax4 = plt.subplots(9,figsize=(5,4))
fig4.suptitle('passive segmental Ia afferent activity', fontsize=8,fontweight='bold' )

for i in range(len(Ia_aff_c)):
    ax4[i].plot(c_time[0::5],c_aff_firing[Ia_aff_c[i]])
    ax4[i].set_title(Ia_aff_c[i],fontweight='bold')
ax4[8].set_xlabel("time in ms",fontweight='bold')

#plot afferent firing data of open loop 

fig5, ax5 = plt.subplots(9,figsize=(5,4))
fig5.suptitle('Active segmental Ia afferent  activity', fontsize=8,fontweight='bold' )

for i in range(len(Ia_aff_o)):
    ax5[i].plot(o_time[0::5],o_aff_firing[Ia_aff_o[i]])
    ax5[i].set_title(Ia_aff_o[i],fontweight='bold')
ax5[8].set_xlabel("time in ms",fontweight='bold')



#plot muscle lengths

fig6,ax6 = plt.subplots(4)
#ax2[i].plot(time[0::5],c_muscle_lengths['tibant_l'],label='tibant')
ax6[0].plot(o_time[0::5],o_muscle_lengths['tibant_l'],label='no feedback')
ax6[0].plot(c_time[0::5],c_muscle_lengths['tibant_l'],label='autogenic Ia feedback')
ax6[0].legend(loc=3, fontsize = 'xx-small')
ax6[0].set_title("ta", fontweight = 'bold') 
ax6[0].set_ylabel('length in m',fontweight = 'bold')

ax6[1].plot(o_time[0::5],o_muscle_lengths['soleus_l'],label='no feedback')
ax6[1].plot(c_time[0::5],c_muscle_lengths['soleus_l'],label='autogenic Ia feedback')
ax6[1].legend(loc=3, fontsize = 'xx-small')
ax6[1].set_title("sol", fontweight = 'bold') 
ax6[1].set_ylabel('length in m',fontweight = 'bold')

ax6[2].plot(o_time[0::5],o_muscle_lengths['gaslat_l'],label='no feedback')
ax6[2].plot(c_time[0::5],c_muscle_lengths['gaslat_l'],label='autogenic Ia feedback')
ax6[2].legend(loc=3, fontsize = 'xx-small')
ax6[2].set_title("gaslat", fontweight = 'bold') 
ax6[2].set_ylabel('length in m',fontweight = 'bold')


ax6[3].plot(o_time[0::5],o_muscle_lengths['gasmed_l'],label='no feedback')
ax6[3].plot(c_time[0::5],c_muscle_lengths['gasmed_l'],label='autogenic Ia feedback')
ax6[3].legend(loc=3, fontsize = 'xx-small')
ax6[3].set_title("gasmed", fontweight = 'bold') 
ax6[3].set_xlabel('time in ms',fontweight = 'bold')
ax6[3].set_ylabel('length in m',fontweight = 'bold')

plt.show()


#!/usr/bin/env python
# coding: utf-8
############################################################################################################
#  This code is part of project NEUROiD developed at Spine Labs
#  Dept. of BME,
#  IIT Hyderabad
#  www.iith.ac.in/~mohanr
#
#  Revision History
#  -----------------------
#   Name             Date            Changes
#  ----------       -------     ----------------------
#   Madhav vinodh   01/01/2019      Initial Creation
############################################################################################################
# Autogeneration of spinal circuits using curated data 
############################################################################################################
#
# 1. This code parses the information from spinal_neurons.xlsx, muscles.xlsx, movement_types.xlsx, connection_rules.xlsx 
#    and generates lumbosacral-ipsilateral spinal connectome (see figure 2 of main text). 
# 
# 2. This code also generates Figure 5 of the manuscript, whcih creates the spinal neurons based msucle-muscle 
#    interactions maps 
# 
# Outputs:
#
# cell_groups.xlsx: cell groups that are utilized by NEUROiD engine to create simulatable computational models
#                   of spinal neurons
# nc_explicit.xlsx: Describes the source and target neurons (bascially the S(kxk) of Figure 2 and Figure 1)
#
#
# Refer supplementary-A and figure-2 of manuscript for more details on how the algorithm works
#
############################################################################################################
 
#import the required libraries 

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
cwd = os.getcwd()
cwd = cwd 

#funtions to import files
def get_movements_input_file():
    return cwd + '/movement_types.xlsx'

def get_muscles_input_file():
    return cwd + '/muscles.xlsx'

def get_spinal_neurons_input_file():
    return cwd + '/spinal_neurons.xlsx'

def get_connection_rules_input_file():
    return cwd + '/connection_rules.xlsx'

#funtions to export output files
def get_nc_explicit_file():
    return cwd + '/nc_explicit.xlsx'

def get_cellgroups_file():
    return cwd + '/cell_groups.xlsx'


#load spinal neurons data to dictionaries
movements_roster = pd.read_excel(get_movements_input_file(),sheet_name='movement_types')
muscles_roster = pd.read_excel(get_muscles_input_file(),sheet_name='muscles')
neurons_roster = {}
neurons_roster['motoneurons'] = []
neurons_roster['interneurons'] = []
neurons_roster['drg_neurons'] = []
load_motoneurons = pd.read_excel(get_spinal_neurons_input_file(),sheet_name='motoneurons')
for a in range(0,len(load_motoneurons)):
    tempdf = load_motoneurons.iloc[a].to_dict() 
    neurons_roster['motoneurons'].append(tempdf)

load_interneurons = pd.read_excel(get_spinal_neurons_input_file(),sheet_name='interneurons')
for a in range(0,len(load_interneurons)):
    tempdf = load_interneurons.iloc[a].to_dict() 
    neurons_roster['interneurons'].append(tempdf)

load_drg_neurons = pd.read_excel(get_spinal_neurons_input_file(),sheet_name='drg_neurons')
for a in range(0,len(load_drg_neurons)):
    tempdf = load_drg_neurons.iloc[a].to_dict() 
    neurons_roster['drg_neurons'].append(tempdf)

#load connection_rules and append them to connections_rosters
connections_roster = []
load_connection_rules = pd.read_excel(get_connection_rules_input_file(),sheet_name='connection_rules')
connections = list(load_connection_rules["connection_type"])
for a in range(0,len(connections)):
    tempdf = load_connection_rules.iloc[a].to_dict() 
    connections_roster.append(tempdf)

connections_info = {}
connections_info["self"] = []
connections_info["agonist"] = []
connections_info["antagonist"] = []
for coninfo in connections_roster:
    for mgrp in coninfo['target_muscle_group'].split(","):
        #print mgrp
        cinfo = { 'src': coninfo['source'], 'dest': coninfo['target'], 'props':coninfo['props'], 'side':coninfo['side'] }
        connections_info[mgrp].append(cinfo)

# function to output the segments list that a particular muscle receives input from
def segments_for_muscle(muscle_name):
    segments = load_motoneurons.loc[load_motoneurons["target_muscle_name"]==muscle_name]['segments'].tolist()[0].split(",")
    #print segments
    return segments
def get_muscles_for_movement(movement_type):
    muscles_for_movement = []
    for each in muscles_roster['muscle_name']:
        each_df = muscles_roster.loc[muscles_roster['muscle_name']==each]
        movements = each_df['movement_type'].tolist()[0].split(",")
        for movement in movements:
            if movement == movement_type:
                muscles_for_movement.append(each)
    return muscles_for_movement

#function to output the antagonist movement types for a given movement types
def get_antagonist_movement(movement_type):
    joint,movement = movement_type.split("_")
    antagonist_movement = joint +"_"+ movements_roster.loc[movements_roster['movement_type']==movement]['antagonist_movement'].tolist()[0]
    return antagonist_movement    

#function to construct a dictionary for any type of spinal cells/neurons
def make_cell_group(neuron_type,target_muscle_name,segment,side,numcells,lamina,transmitter,hoc_template_name):
    #print segment
    #print target_muscle_name
    cell_group = {'axonr': '',
                 'axons': '',
                              'cell_group_region': lamina+"_"+side,
                              'cell_packing_adapter': json.dumps({'number': str(numcells), 'pdf': 'random'}),
                              'cell_type': neuron_type,
                              'dends': '',
                              'funcs': '',
                              'name': lamina+"_"+side+"_"+neuron_type+"_"+target_muscle_name,
                              'neuro_rx': '',
                              'neuro_tx': transmitter,
                              'sub_cell_type': target_muscle_name,
                              'seg_info':segment,
                              'side':side,
                              'hoc_template':hoc_template_name}
    return cell_group

#function to construct the dictionary of spinal motoneurons
def create_moto_cellgroups(moto_roster):
    motoneuron_cellgroups = []
    for each in moto_roster:
        lamina = each['lamina']
        neuron_type = each['neuron_type']
        target_muscle_name = each['target_muscle_name']
        transmitter = each['transmitter']
        numcells = each['numcells']
        segments_list = each['segments'].split(",")
        spinal_side_list = []
        hoc_template_name = each['hoc_template']
        if each['side(L/R/both)'] == 'both':
            spinal_side_list = ['L','R']
        else:
            spinal_side_list = each['side(L/R/both)']
            
        for each_side in spinal_side_list:
            for each_segment in segments_list:
                motoneuron_cellgroups.append(make_cell_group(neuron_type,target_muscle_name,each_segment,each_side,numcells,lamina,transmitter,hoc_template_name))
    return(motoneuron_cellgroups)

#function to construct the dictionary of spinal interneurons 
def create_inter_cellgroups(inter_roster):
    interneuron_cellgroups = []
    for each in inter_roster:
        lamina = each['lamina']
        neuron_type = each['neuron_type']
        transmitter = each['transmitter']
        numcells = each['numcells']
        hoc_template_name = each['hoc_template']
        target_muscles_list = []
        segments_list =[]
        if (each['target_muscle_name'] == 'All') & (each['segments'] == 'mirror'):
            target_muscles_list = list(load_motoneurons['target_muscle_name'])
            #print 1
        else:
            target_muscles_list = each['target_muscle_name']
            segments_list = each['segments'].split(",")
            
        
        spinal_side_list = []
        if each['side(L/R/both)'] == 'both':
            spinal_side_list = ['L','R']
        else:
            spinal_side_list = each['side(L/R/both)']
            
        for each_side in spinal_side_list:
            for each_target_muscle_name in target_muscles_list:
                #print each_target_muscle_name
                segments_list = segments_for_muscle(each_target_muscle_name)
                #print segments_list
                for each_segment in segments_list:
                    interneuron_cellgroups.append(make_cell_group(neuron_type,each_target_muscle_name,each_segment,each_side,numcells,lamina,transmitter,hoc_template_name))
    return(interneuron_cellgroups)

#function to construct the dictionary of spinal drg_neuron (sensory neurons of spinal dorsal root ganglions)
def create_drg_cellgroups(drg_roster):
    drg_cellgroups = []
    for each in drg_roster:
        #lamina = each['lamina']
        neuron_type = each['neuron_type']
        transmitter = each['transmitter']
        numcells = each['numcells']
        hoc_template_name = each['hoc_template']
        target_muscles_list = []
        segments_list =[]
        if (each['target_muscle_name'] == 'All') & (each['segments'] == 'mirror'):
            target_muscles_list = list(load_motoneurons['target_muscle_name'])
            #print 1
        else:
            target_muscles_list = each['target_muscle_name']
            segments_list = each['segments'].split(",")
            
        
        spinal_side_list = []
        if each['side(L/R/both)'] == 'both':
            spinal_side_list = ['L','R']
        else:
            spinal_side_list = each['side(L/R/both)']
            
        for each_side in spinal_side_list:
            for each_target_muscle_name in target_muscles_list:
                #print each_target_muscle_name
                segments_list = segments_for_muscle(each_target_muscle_name)
                for each_segment in segments_list:
                    lamina = each_segment + "_" + each['lamina']
                    drg_cellgroups.append(make_cell_group(neuron_type,each_target_muscle_name,each_segment,each_side,numcells,lamina,transmitter,hoc_template_name))
    return(drg_cellgroups)

#fucntion to construct the dictionary of spinal connections 
def make_connection(s_neuron,s_neuron_subtype,s_neuron_lamina,s_neuron_seg,d_neuron,d_neuron_subtype,d_neuron_lamina,d_neuron_seg,props,transmitter): 
    net_conn = { "source_segment": "Human_"+s_neuron_seg ,
                             "source_cellgroup": s_neuron_lamina+"_"+s_neuron+"_"+s_neuron_subtype,
                             "dest_segment": "Human_"+d_neuron_seg,
                             "dest_cellgroup": d_neuron_lamina+"_"+d_neuron+"_"+d_neuron_subtype,
                             "axon_route":'',
                             "nc_props": props,
                             "transmitter":transmitter,
                              }
    return net_conn

#function to connect a pair of cell_groups defined by its source cell_pool, target_cell _pool, connect_side, 
#output is a dict of a conenction
def connect_cell_groups(source_cell_pool,target_cell_pool,connect_side,props):
    net_conn = []
    if connect_side == 'ipsilateral':
        #print source_cell_pool
        for each_source_cell_group in source_cell_pool:
            for each_target_cell_group in target_cell_pool:
                if each_source_cell_group['side'] == each_target_cell_group['side']:
                    net_con = make_connection(s_neuron = each_source_cell_group['cell_type'],
                                                          s_neuron_subtype = each_source_cell_group['sub_cell_type'],
                                                            s_neuron_lamina = each_source_cell_group['cell_group_region'],
                                                            s_neuron_seg = each_source_cell_group['seg_info'],
                                                            d_neuron = each_target_cell_group['cell_type'], 
                                                            d_neuron_subtype = each_target_cell_group['sub_cell_type'],
                                                            d_neuron_lamina = each_target_cell_group['cell_group_region'],
                                                            d_neuron_seg = each_target_cell_group['seg_info'],
                                                            transmitter = each_source_cell_group['neuro_tx'],
                                                            props = props)
                    net_conn.append(net_con)
    if connect_side == 'contralateral':
        for each_source_cell_group in source_cell_pool:
            for each_target_cell_group in target_cell_pool:
                if (each_source_cell_group['side'] == L) & (target_cell_group['side'] == R):
                    net_con = make_connection(s_neuron = each_source_cell_group['cell_type'],
                                                            s_neuron_subtype = each_source_cell_group['sub_cell_type'],
                                                            s_neuron_lamina = each_source_cell_group['cell_group_region'],
                                                            s_neuron_seg = each_source_cell_group['seg_info'],
                                                            d_neuron = each_target_cell_group['cell_type'], 
                                                            d_neuron_subtype = each_target_cell_group['sub_cell_type'],
                                                            d_neuron_lamina = each_target_cell_group['cell_group_region'],
                                                            d_neuron_seg = each_target_cell_group['seg_info'],
                                                            props = props)
                    net_conn = net_conn + net_con
                if (each_source_cell_group['side'] == R) & (target_cell_group['side'] == L):
                    net_con = make_connection(s_neuron = each_source_cell_group['cell_type'],
                                                            s_neuron_subtype = each_source_cell_group['sub_cell_type'],
                                                            s_neuron_lamina = each_source_cell_group['cell_group_region'],
                                                            s_neuron_seg = each_source_cell_group['seg_info'],
                                                            d_neuron = each_target_cell_group['cell_type'], 
                                                            d_neuron_subtype = each_target_cell_group['sub_cell_type'],
                                                            d_neuron_lamina = each_target_cell_group['cell_group_region'],
                                                            d_neuron_seg = each_target_cell_group['seg_info'],
                                                            props = props)
                    net_conn = net_conn + net_con
    return net_conn

#function to generate the entire connectome defined by dicntionary of cell_groups, dictionary of connection rules, and modality of connection 
def connect(cell_groups_dataframe,template,connection_modality):
    #source_cell_groups = {}
    #target_cell_groups = {}
    ncs = []
    source_neuron = template['src']
    target_neuron = template['dest']
    connect_side = template['side']
    source_neuron_pool = cell_groups_dataframe.loc[(cell_groups_dataframe['cell_type']==source_neuron)]
    target_neuron_pool = cell_groups_dataframe.loc[(cell_groups_dataframe['cell_type']==target_neuron)]
    #print target_neuron_pool['sub_cell_type']
    source_sub_cell_types = set(cell_groups_dataframe['sub_cell_type'].tolist())
    #net_conn = None
    if connection_modality == 'self':
           for sub_cell_type in source_sub_cell_types:
                #print sub_cell_type
                #target_neuron_pool['sub_cell_type'] == sub_cell_type
                target_cell_groups = target_neuron_pool.loc[(target_neuron_pool['sub_cell_type']==sub_cell_type)]
                source_cell_groups = source_neuron_pool.loc[source_neuron_pool['sub_cell_type']==sub_cell_type]
                source_cell_groups = list(source_cell_groups.T.to_dict().values())
                target_cell_groups = list(target_cell_groups.T.to_dict().values())
                #print source_cell_groups
                #nc.append(connect_cell_groups(source_cell_groups,target_cell_groups,connect_side,template['props']))
                net_conn = connect_cell_groups(source_cell_groups,target_cell_groups,connect_side,template['props'])
                #print net_conn
                if net_conn != None:
                    for each in net_conn:
                        if each not in ncs:
                            ncs.append(each)
    if connection_modality == 'agonist':
        for sub_cell_type in source_sub_cell_types:
            #check what movements the sub_cell_type i.e agnistic to muscle_name involves in  
            movements = muscles_roster.loc[muscles_roster['muscle_name']==sub_cell_type]['movement_type'].tolist()[0].split(",")
            for each_movement_type in movements:
                agonist_muscles_list = get_muscles_for_movement(each_movement_type)
                #to make sure that current muscle group is removed so only gives agonist list of muscles
                agonist_muscles_list.remove(sub_cell_type)
                
                if agonist_muscles_list == None:
                    pass
                else:
                        for each_agonist_muscle in agonist_muscles_list:
                            target_cell_groups = target_neuron_pool.loc[target_neuron_pool['sub_cell_type']==each_agonist_muscle]
                            source_cell_groups = source_neuron_pool.loc[source_neuron_pool['sub_cell_type']==sub_cell_type]
                            source_cell_groups = list(source_cell_groups.T.to_dict().values())
                            target_cell_groups = list(target_cell_groups.T.to_dict().values())
                            #nc.append(connect_cell_groups(source_cell_groups,target_cell_groups,connect_side,template['props']))
                            net_conn = connect_cell_groups(source_cell_groups,target_cell_groups,connect_side,template['props'])
                            #print net_conn
                            if net_conn != None:
                                for each in net_conn:
                                    if each not in ncs:
                                        ncs.append(each)
    if connection_modality == 'antagonist':
        for sub_cell_type in source_sub_cell_types:
            #check what movementsn the sub_cell_type i.e agnistic to muscle_name involves in  
            movements = muscles_roster.loc[muscles_roster['muscle_name']==sub_cell_type]['movement_type'].tolist()[0].split(",")
            #print movements
            for each_movement_type in movements:
                antagonist_muscles_list = get_muscles_for_movement(get_antagonist_movement(each_movement_type))
                #print antagonist_muscles_list
                for each_antagonist_muscle in antagonist_muscles_list:
                    target_cell_groups = target_neuron_pool.loc[target_neuron_pool['sub_cell_type']==each_antagonist_muscle]
                    source_cell_groups = source_neuron_pool.loc[source_neuron_pool['sub_cell_type']==sub_cell_type]
                    source_cell_groups = list(source_cell_groups.T.to_dict().values())
                    target_cell_groups = list(target_cell_groups.T.to_dict().values())
                    net_conn = connect_cell_groups(source_cell_groups,target_cell_groups,connect_side,template['props'])
                    if net_conn != None:
                        for each in net_conn:
                            if each not in ncs:
                                ncs.append(each)
    return ncs


#generate cell_groups and write to cell_groups.xlsx
writer1 = pd.ExcelWriter(get_cellgroups_file())
cell_groups_info = create_drg_cellgroups(neurons_roster['drg_neurons'])+create_moto_cellgroups(neurons_roster['motoneurons'])+create_inter_cellgroups(neurons_roster['interneurons'])
cellgroup_cols = ["name","cell_type","sub_cell_type","cell_group_region","axons","dends","cell_packing_adapter","axonr","funcs","neuro_tx","neuro_rx"]
tempdf = pd.DataFrame(cell_groups_info)
emptydf = pd.DataFrame()
emptydf.to_excel(writer1,sheet_name="common")
for each in set(tempdf['seg_info'].tolist()):
    seg_name="Human_" + each
    df = pd.DataFrame(tempdf.loc[tempdf['seg_info']==each])
    df.to_excel(writer1,sheet_name=seg_name,columns=cellgroup_cols,index=False)
writer1.save()
writer1.close()

#generate connections and write the connections to nc_explicit.xlsx
total_conns = []
nc = []
writer2 = pd.ExcelWriter(get_nc_explicit_file())
netcon_cols = ["source_segment","source_cellgroup","dest_segment","dest_cellgroup","axon_route","nc_props","transmitter"]
for each_modality in list(connections_info.keys()):
    for each_template in connections_info[each_modality]:
        connexions = (connect(tempdf,each_template,each_modality))
        nc.extend(connexions)

for each in nc:
    if each not in total_conns:
        total_conns.append(each)
nc_df = pd.DataFrame(total_conns)
nc_df.to_excel(writer2,sheet_name="common",columns=netcon_cols,index=False)
writer2.save()
writer2.close()

print ("cell_groups are generated and connections are made")
print ("now generating the interaction maps")
    
load_mus = pd.read_excel("muscles_for_segs1.xlsx")
muscles_seg_order = load_mus['muscles'].to_list()


def find_tar_neuron(src_neuron,total_conn):
    tar_neuron_list = []
    for each in total_conn:
        if each['source_cellgroup'] == src_neuron:
            tar_neuron_list.append(each['dest_segment']+"_"+each['dest_cellgroup'])
    return tar_neuron_list


#code to draw Ia interaction matrices based on individual spinal connections
syn = "Ia"

#create an empty dataframe i.e an adjacency matrix with muscles names.
d1 = pd.DataFrame(0, index = muscles_seg_order, columns = muscles_seg_order)
d1_ = pd.DataFrame(0, index = muscles_seg_order, columns = muscles_seg_order)
#iterate over connection to find the desired source neuron
for each in total_conns:
    if each['source_cellgroup'].split('_')[3] == syn:
        source_muscle = each['source_cellgroup'].split('_')[4]
        dest_muscle = each['dest_cellgroup'].split('_')[3]
        target_celltype = each['dest_cellgroup'].split('_')[2]
        source_seg = each['source_cellgroup'].split('_')[0]
        tar_seg = each['dest_segment'].split('_')[1]
        #print tar_seg
        
        #monosynaptic based Ia scorring
        if target_celltype == "aMot":
            if source_muscle == dest_muscle:
                
                #using the total count
                d1.at[source_muscle,dest_muscle]=d1.at[source_muscle,dest_muscle]+1
            else:
                d1.at[source_muscle,dest_muscle]=d1.at[source_muscle,dest_muscle]+1

        #disynaptic structures, find the final target aMot
        if target_celltype == "IaIn": 
            #find the aMot targets of IaIn
            tar_list = find_tar_neuron(each['dest_cellgroup'],total_conns)
            for each in tar_list:
                if each.split("_")[4] == 'aMot':
                    dest_muscle = each.split("_")[5] 
                    tar_seg = each.split("_")[1]
                    #print each
                    if source_muscle != dest_muscle:
                        #using counts
                        d1_.at[source_muscle,dest_muscle]=d1_.at[source_muscle,dest_muscle]-1


# Set up the matplotlib figure
f1, ax = plt.subplots(figsize=(11, 9))
ax.set_title("Ia-Excitatory", fontsize = 20)

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(d1, cmap=cmap, center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 1})

#plt.show()
#f.savefig('Ia_excitatory.png', bbox_inches='tight', pad_inches=0.1)


# Set up the matplotlib figure
f2, ax1 = plt.subplots(figsize=(11, 9))
ax1.set_title("Ia-Inhibitory", fontsize = 20)

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(d1_, cmap=cmap, center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 1})
#plt.show()
#f.savefig('Ia_inhibitory.png', bbox_inches='tight', pad_inches=0.1)


#code to draw the Ib interaction matrices based on individual spinal connections
import numpy as np

#insert required synergy to build for visualization
syn = "Ib"

#create an empty dataframe i.e an adjacency matrix with muscles names.
d2 = pd.DataFrame(0, index = muscles_seg_order, columns = muscles_seg_order)
d2_ = pd.DataFrame(0, index = muscles_seg_order, columns = muscles_seg_order)
#iterate over connection to find the desired source neuron
for each in total_conns:
    if each['source_cellgroup'].split('_')[3] == syn:
        source_muscle = each['source_cellgroup'].split('_')[4]
        dest_muscle = each['dest_cellgroup'].split('_')[3]
        target_celltype = each['dest_cellgroup'].split('_')[2]
        source_seg = each['source_cellgroup'].split('_')[0]
        tar_seg = each['dest_segment'].split('_')[1]
        #print tar_seg

        #disynaptic structures
        if target_celltype == "IbEx": 
            tar_list = find_tar_neuron(each['dest_cellgroup'],total_conns)
            for each in tar_list:
                if each.split("_")[4] == 'aMot':
                    dest_muscle = each.split("_")[5] 
                    tar_seg = each.split("_")[1]
                    if source_muscle != dest_muscle:
                        d2.at[source_muscle,dest_muscle] = d2.at[source_muscle,dest_muscle] + 1
        if target_celltype == "IbIn": 
            #find the aMot targets of IaIn
            tar_list = find_tar_neuron(each['dest_cellgroup'],total_conns)
            for each in tar_list:
                if each.split("_")[4] == 'aMot':
                    dest_muscle = each.split("_")[5] 
                    tar_seg = each.split("_")[1]
                    if source_muscle == dest_muscle:
                        d2_.at[source_muscle,dest_muscle] = d2_.at[source_muscle,dest_muscle] - 1


# Set up the matplotlib figure
f2, ax2 = plt.subplots(figsize=(11, 9))
#title = "Ib synergy matrix"
ax2.set_title("Ib-Excitatory", fontsize = 20)

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(d2, cmap=cmap, center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 1})
#plt.show()
#f.savefig('Ib_excitatory.png', bbox_inches='tight', pad_inches=0.1)


# Set up the matplotlib figure
f3, ax3 = plt.subplots(figsize=(11, 9))
#title = "Ib synergy matrix"
ax3.set_title("Ib-Inhibitory", fontsize = 20)

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(d2_, cmap=cmap, center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 1})
#plt.show()
#f.savefig('Ib_inhibitory.png', bbox_inches='tight', pad_inches=0.1)


#code to draw the renshaw interation matrices of the individual spinal connections
import numpy as np
#load total circuit stack 
#total_connections = moto_run(total_muscles)
#create an empty dataframe i.e an adjacency matrix with muscles names.
d3 = pd.DataFrame(0, index = muscles_seg_order, columns = muscles_seg_order)
for each in total_conns:
    if each['source_cellgroup'].split('_')[2] == 'aMot':
        source_muscle = each['source_cellgroup'].split('_')[3]
        dest_muscle = each['dest_cellgroup'].split('_')[3]
        target_celltype = each['dest_cellgroup'].split('_')[2]
        
        #find aMot targets for each
        if target_celltype == "Ren":
            tar_list = find_tar_neuron(each['dest_cellgroup'],total_conns)
            for each in tar_list:
                if each.split("_")[4] == 'aMot':
                    dest_muscle = each.split("_")[5] 
                    tar_seg = each.split("_")[1]
                    #print each
                    if source_muscle == dest_muscle:
                        #d3.at[source_muscle,dest_muscle]=d3.at[source_muscle,dest_muscle]+2*((-1)-calc_seg_score(source_seg,tar_seg)) 
                        d3.at[source_muscle,dest_muscle]=d3.at[source_muscle,dest_muscle]-1
                    if source_muscle != dest_muscle:
                        #d3.at[source_muscle,dest_muscle]=d3.at[source_muscle,dest_muscle]+(-1)-calc_seg_score(source_seg,tar_seg)
                        d3.at[source_muscle,dest_muscle]=d3.at[source_muscle,dest_muscle]-1


# Set up the matplotlib figure
f4, ax4 = plt.subplots(figsize=(11, 9))
#title = "Renshaw synergy matrix"
ax4.set_title("Renshaw", fontsize = 20)

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

#mask to draw only lower triangle
mask = np.zeros_like(d3)
mask[np.triu_indices_from(mask,k=1)] = True
# Draw the heatmap with the mask and correct aspect rati0
sns.heatmap(d3, cmap=cmap,center=0, mask=False,annot=False,
            square=False, linewidths=1, cbar_kws={"shrink": 1})

plt.show()
#f.savefig('Ren_interaction.png', bbox_inches='tight', pad_inches=0.1)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:31:29 2021

@author: alvarezguido, OlmedoMatias
GITHUB: https://github.com/alvarezguido
"""

"""
SYNOPSIS
This is a Satellite AIS simulator based on SimPy library.

"""


import simpy
import math
import random
import re
import os
import datetime
import sys
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap as Basemap



class myVessel():
    def __init__(self):
        self.id = None
        self.bw = None

        self.ts = None
        self.T = None
        
        self.sent = 0
        self.lost = 0
        self.collided = 0 
        self.extracted = 0
        self.Prx = 0

        self.totalLost = 0
        self.totalColl = 0
        self.totalExtracted = 0


def LoadAllFiles(path_leo_xyz, path_leo_lla, path_xyz, path_lla):
    leo_pos = np.loadtxt(path_leo_xyz, skiprows=1, delimiter=',', usecols=(1,2,3))
    ## WHERE:
        ## leo_pos[i,j]:
            ## i --> the step time in sat pass
            ## j --> 0 for x-position, 1 for y-position, 2 for z-position
    
    leo_pos_lla = np.loadtxt(path_leo_lla, skiprows=1, delimiter=',', usecols=(1,2,3))
    ## WHERE:
        ## leo_pos[i,j]:
            ## i --> the step time in sat pass
            ## j --> 0 for latitud, 1 for longitud, 2 for altitud

    # Read all the files
    sites_pos_xyz = np.loadtxt(path_xyz, skiprows=1, delimiter=',', usecols=(1,2,3))
    # WHERE:
        # sites_pos_xyz[i,j]:
            # i --> the node i
            # j --> 0 for x-position, 1 for y-position, 2 for z-position

    sites_pos_lla = np.loadtxt(path_lla, skiprows=1, delimiter=',', usecols=(1,2,3))
    # WHERE:
        # sites_pos_lla[i,j]:
            # i --> the node i
            # j --> 0 for latitud, 1 for longitud, 2 for altitud

    return leo_pos, leo_pos_lla, sites_pos_xyz, sites_pos_lla

def LoadAntennaGain(path_antenna_sat, path_antenna_vessel):
    G_sat_Array = np.loadtxt(path_antenna_sat, skiprows=1, delimiter=',', usecols=(1,2,3,4,5,6,7), max_rows=90)
    ## WHERE:
        ## G_sat_Array[i,j]:
            ## i --> theta
            ## j --> phi
    G_sat_Array = np.mean(G_sat_Array, axis=1)

    G_vessel_Array = np.loadtxt(path_antenna_vessel, skiprows=1, delimiter=',', usecols=(1,2,3,4,5,6,7), max_rows=90)
    ## WHERE:
        ## G_vessel_Array[i,j]:
            ## i --> theta
            ## j --> phi
    G_vessel_Array = np.mean(G_vessel_Array, axis=1)
    G_vessel_Array = np.append(G_vessel_Array, np.zeros(90)-120)

    return G_sat_Array, G_vessel_Array

def LoadIntoGraph(nrVessels):
    global sites_pos_xyz, sites_pos_lla
    # Generate a fully connected graph
    G = nx.complete_graph(nrVessels)

    # Load all the attributes to the nodes
    for i in range(nrVessels):
        G.nodes[i]['pos_xyz']       = sites_pos_xyz[i]
        G.nodes[i]['pos_lla']       = sites_pos_lla[i]
        G.nodes[i]['freq']          = 162.5e6 # Inicialmente se toma un valor medio, pero luego se debe ajustar segun el canal que se seleccione
        G.nodes[i]['clase']         = random.choice(['A', 'B'])
        txPow = 12.5 if G.nodes[i]['clase'] == 'A' else 2
        G.nodes[i]['txPow']         = 10*np.log10(txPow) + 30 #dBm


        G.nodes[i]['obj']           = myVessel()
        G.nodes[i]['obj'].id        = i
        G.nodes[i]['obj'].bw        = random.choice([12.5, 25]) #Khz
        G.nodes[i]['obj'].ts        = []
        G.nodes[i]['obj'].T         = 1 # Cantidad de TS usados por minuto
        G.nodes[i]['obj'].clase     = G.nodes[i]['clase']

        G.nodes[i]['obj'].sent      = 0
        G.nodes[i]['obj'].lost      = 0
        G.nodes[i]['obj'].collided  = 0
        G.nodes[i]['obj'].extracted = 0
        G.nodes[i]['obj'].Prx       = 0
        
        G.nodes[i]['obj'].totalLost         = 0 
        G.nodes[i]['obj'].totalColl         = 0 
        G.nodes[i]['obj'].totalExtracted    = 0 
        
    return G

def FilterEdges(G, umbral):
    umbral_pow2 = umbral*umbral #Since we use power raised equation
    for i, j in G.edges():
        #G.edges[i, j]['dst'] = np.sqrt(((G.nodes[i]['pos_xyz'] - G.nodes[j]['pos_xyz'])**2).sum())
        G.edges[i, j]['dst'] = ((G.nodes[i]['pos_xyz'] - G.nodes[j]['pos_xyz'])**2).sum()
        if G.edges[i, j]['dst'] >= umbral_pow2:
            G.remove_edge(i, j)   
    return G

def InitTS(G, nrTS):
    # Inicializacion (un time slot por barco, equivale a que todos los barcos transmiten una vez por minuto)
    for node_i in range(G.number_of_nodes()):
        candidates = list(range(0,nrTS))
        random.shuffle(candidates)
        used = []
        for neighbor in G.neighbors(node_i):
            used += G.nodes[neighbor]['obj'].ts
            for neighbor_of_neighbor in G.neighbors(neighbor):
                used += G.nodes[neighbor_of_neighbor]['obj'].ts
        
        for i in used:
            if i in candidates:
                candidates.remove(i)
        if len(candidates) > 0:
            seleccionado = candidates[0]
        else:
            seleccionado = []
        G.nodes[node_i]['obj'].ts = [seleccionado]

        #print('nodo: ' + str(node_i))
        #print('ts usados por el entorno: ' + str(used))
        #print('ts seleccionado: ' + str(seleccionado))
    return G

def PlotMap(G, leo_pos_lla):
    # Define a basemap
# =============================================================================
#     m = Basemap(projection='merc',
#                 llcrnrlon=-90, 
#                 llcrnrlat=-60, 
#                 urcrnrlon=-20, 
#                 urcrnrlat=0,
#                 lat_ts=0,
#                 resolution='i',
#                 suppress_ticks=True)
# =============================================================================
    global nrVessels
    plt.figure()
    m = Basemap(width=6000000,
                height=4500000,
                resolution='c',
                projection='aea',
                lat_0=11, #35
                lon_0=-80,
                lon_1=-65) #40
    
    # convert lat and lon to map projection
    lats = np.stack(list(dict(G.nodes.data('pos_lla')).values()), axis=0)[:,0]
    lons = np.stack(list(dict(G.nodes.data('pos_lla')).values()), axis=0)[:,1]
    mx,my = m(lons, lats)

    # put map projection coordinates in pos dictionary
    pos=dict(enumerate(np.stack([mx.tolist(), my.tolist()], axis=1)))
    # draw
    #nx.draw_networkx(G,pos,node_size=200,node_color='blue')
    nx.draw(G, with_labels=False, node_size=50, node_color='red', font_weight='bold', pos=pos)

    # draw ground track
    leo_lats = leo_pos_lla[:,0]
    leo_lons = leo_pos_lla[:,1]
    leo_mx,leo_my = m(leo_lons, leo_lats)
    step=20
    plt.plot(leo_mx[0::step], leo_my[0::step], 'c--',linewidth=3.0)
    

    # Now draw the map
    #m.drawcountries()
    #m.drawstates()
    m.bluemarble()
    plt.show()
    plt.savefig(str(nrVessels)+"_nodes.jpg")


def calcDistance (leo_pos, sites_pos):
    dist_sat = np.zeros((sites_pos.shape[0],3,leo_pos.shape[0]))
    t = 0 # <- Se usa para algo?
    for i in range(leo_pos.shape[0]):
        t+=1
        dist_sat [:,:,i] = leo_pos[i,:] - sites_pos
    ## WHERE:
        ## dist_sat[i,j,k]:
            ## i --> the node i
            ## j --> 0 for x-position, 1 for y-position, 2 for z-position
            ## k --> the step time in sat pass
    return dist_sat

def calcSlant (dist_sat):
    #### FOR COMPUTE DISTANCE MAGNITUDE (ABS) FROM END-DEVICE TO SAT PASSING BY ####
    distance = np.zeros((sites_pos.shape[0],leo_pos.shape[0]))
    distance[:,:] = (dist_sat[:,0,:]**2 + dist_sat[:,1,:]**2 + dist_sat[:,2,:]**2)**(1/2)
    ## WHERE:
        ## distance[i,j]:
            ## i --> the node i
            ## j --> the step time in sat pass
    return distance  

def calcLpl(distance):
    ##MATRIX FOR LINK BUDGET Lpl ###
    Lpl = np.zeros((sites_pos.shape[0],leo_pos.shape[0])) 
    Lpl = 20*np.log10(distance*1000) + 20*np.log10(freq) - 147.55 #DISTANCE MUST BE IN METERS
    ## WHERE:
        ## Lpl[i,j]:
            ## i --> the node i
            ## j --> the step time in sat pass 
    return Lpl

def calcAngles():
    global sites_pos, leo_pos
    delta = np.zeros((sites_pos.shape[0],3,leo_pos.shape[0]))
    for i in range(leo_pos.shape[0]):
        delta[:,:,i] = leo_pos[i,:] - sites_pos

    Angle_ZenithToSat = np.zeros((sites_pos.shape[0], leo_pos.shape[0]))
    for i in range(leo_pos.shape[0]):
        Angle_ZenithToSat[:,i] = np.rad2deg(np.arccos(np.sum(sites_pos*delta[:,:,i], axis=1)/np.sqrt(np.sum(sites_pos*sites_pos, axis=1)*np.sum(delta[:,:,i]*delta[:,:,i], axis=1))))

    delta = delta*-1
    Angle_NadirToVessel = np.zeros((sites_pos.shape[0], leo_pos.shape[0]))
    for i in range(leo_pos.shape[0]):
        Angle_NadirToVessel[:,i] = np.rad2deg(np.arccos(np.sum(leo_pos[i,:]*delta[:,:,i], axis=1)/np.sqrt(np.sum(leo_pos[i,:]*leo_pos[i,:], axis=0)*np.sum(delta[:,:,i]*delta[:,:,i], axis=1))))
    Angle_NadirToVessel = 180 - Angle_NadirToVessel

    return Angle_ZenithToSat, Angle_NadirToVessel

def G_Sat_FromAngle(angle):
    global G_sat_Array
    aux_angle = int(abs(round(angle)))
    G_sat = G_sat_Array[aux_angle]
    return G_sat

def G_Vessel_FromAngle(angle):
    global G_vessel_Array
    aux_angle = int(abs(round(angle)))
    G_Vessel = G_vessel_Array[aux_angle]
    return G_Vessel

def linkBudget (vesselid, instantoftime):
    global sensibility, Lpl, Angle_NadirToVessel, Angle_ZenithToSat

    Angle_Sat = Angle_NadirToVessel[vesselid, instantoftime]
    Angle_vessel = Angle_ZenithToSat[vesselid, instantoftime]
    G_sat = G_Sat_FromAngle(Angle_Sat)
    G_vessel = G_Vessel_FromAngle(Angle_vessel)

    Prx = G.nodes[vesselid]["txPow"] + G_sat + G_vessel - Lpl[vesselid, instantoftime]
    if Prx <= sensibility:
        return 1, Prx
    else:
        return 0, Prx

def captureEffect (vessel, other):
    global captureThreshold
    print ("{:3.5f} || Prx: Vesssel {} is {:3.2f} dBm, Vessel {} is {:3.2f} dBm; difference is {:3.2f} dBm".format(env.now, vessel.id, vessel.Prx, other.id, other.Prx, abs(vessel.Prx-other.Prx)))
    if abs(vessel.Prx - other.Prx) < captureThreshold:
        print( "{:3.5f} || Collision power both Vessel {} and Vessel {}".format(env.now,vessel.id, other.id))
        return (vessel, other)
    elif vessel.Prx - other.Prx < captureThreshold:
        print ("{:3.5f} || Vessel {} has overpowered Vessel {}".format(env.now,other.id, vessel.id))
        return (vessel,)
    else:
        print ("{:3.5f} || Vessel {} has overpowered Vessel {}".format(env.now,vessel.id, other.id))
        return (other,)
        

def checkCollision (vessel):
    global packetsAtBS
    col = 0
    if packetsAtBS:
        print ("{:3.5f} || >> FOUND overlap... Vessel {} others: {}".format(env.now,vessel.id, len(packetsAtBS)))
        for other in packetsAtBS:
            if other.id != vessel.id:
                print ("{:3.5f} || >> Vessel {} overlapped with Vessel {}. Let's check Capture Effect...".format(env.now,vessel.id, other.id))
                c = captureEffect(vessel, other)
                for p in c:
                    p.collided = 1
                    if p == vessel:
                        col = 1
        return col
    return col
    

def transmit (env, vessel):
    global nrTS, logs

    while True:
        count = 0
        for i, nextTS in enumerate(vessel.ts):
            if vessel in packetsAtBS:
                print ("{:3.5f} || ERROR: packet is already in...".format(env.now))
            else:
                if count == 0:
                    if vessel.ts[-1] < 2250:
                        wait = nextTS * 26.67e-3 #back off to next TS
                    else:
                        wait = (nextTS - 2250) * 26.67e-3 #back off to next TS
                    yield env.timeout(wait)
                else:
                    lastTS = vessel.ts[i-1]
                    wait = (nextTS - lastTS -1) * 26.67e-3  #back off to next TS
                    yield env.timeout(wait)
                print ("{:3.5f} || Vessel {} begins to transmit".format(env.now,vessel.id))
                vessel.sent = vessel.sent + 1
                if vessel in packetsAtBS:
                    print ("{:3.5f} || ERROR: packet is already in...".format(env.now))
                else:
                    isLost, Prx = linkBudget(vessel.id, math.ceil(env.now))
                    vessel.Prx = Prx
                    if isLost == 1:
                        print ("{:3.5f} || Vessel {}: Transmission is Lost due Link Budget".format(env.now,vessel.id))
                        vessel.lost = 1     # Comentario: Cuando da lost yo opino que no se haga el packetsAtBS.append(vessel)
                    if (checkCollision(vessel) == 1):
                        vessel.collided = 1
                    else:
                        vessel.extracted = 1
                    packetsAtBS.append(vessel)
                    yield env.timeout(26.67e-3) # Comentario: toda la comparacion se hace al inicio del ts. imagino que hay preferencia por el orden de ejecucion, el primer obj que se analiza
                                                # pueda dar como resultado (lost=0, collided=0 (logico porque es el primero), extracted=1), el segundo obj puede colisionar con el primero 
                                                # (la funcion checkCollision verifica esto, y altera el flag collided del primer obj y del segundo), 
                                                # puede darse que por captureEffect el segundo obj sea el que se extraiga, hasta aca todo perfecto, pero quien revierte el flag extracted del primer objeto?
                                                # En el archivo log esto se resulve solo, pero los contadores totalLost, totalColl, totalExtracted, para mi no reflejan lo que realmente sucede.
                    
            if (vessel in packetsAtBS):
                packetsAtBS.remove(vessel)
                
            vessel.totalLost += vessel.lost
            vessel.totalColl += vessel.collided
            vessel.totalExtracted += vessel.extracted
            
            ##print to logfile
            if vessel.lost:
                logs.append("{:3.3f},{},{:3.3f},{:3.3f},{},PL".format(env.now,vessel.id, distance[vessel.id, math.ceil(env.now)], elev[vessel.id, math.ceil(env.now)], vessel.clase))
            elif vessel.collided:
                logs.append("{:3.3f},{},{:3.3f},{:3.3f},{},PC".format(env.now,vessel.id, distance[vessel.id, math.ceil(env.now)], elev[vessel.id, math.ceil(env.now)], vessel.clase))
            elif vessel.extracted:
                logs.append("{:3.3f},{},{:3.3f},{:3.3f},{},PE".format(env.now,vessel.id, distance[vessel.id, math.ceil(env.now)], elev[vessel.id, math.ceil(env.now)], vessel.clase))
            else:
                print ("ERROR")
            
            vessel.lost = 0
            vessel.collided = 0
            vessel.extracted = 0
            count = count +1
        
        if nrTS == 2250:
            #yield env.timeout((nrTS - vessel.ts[-1] - 1)*26.67e-3)
            yield env.timeout(60 - (vessel.ts[-1] +1 )*26.67e-3)
        elif nrTS == 4500:
            if vessel.ts[-1] < 2250:
                if vessel.ts[-1] == 0 or vessel.ts[-1] == 2249:
                    yield env.timeout(60 - (vessel.ts[-1])*26.67e-3)
                else:
                    yield env.timeout(60 - (vessel.ts[-1])*26.67e-3 - 26.67e-3)
                #yield env.timeout((nrTS - vessel.ts[-1] - 1)*26.67e-3)
            else:
                if vessel.ts[-1] == 4499:
                    yield env.timeout(60 - (vessel.ts[-1] - 2251)*26.67e-3 - 26.67e-3)
                else:
                    yield env.timeout(60 - (vessel.ts[-1] - 2251)*26.67e-3 -2*26.67e-3)
                #yield env.timeout((nrTS - 2250 - vessel.ts[-1] - 1)*26.67e-3)



# ------ GLOBAL VARIABLES ------
name = "s-ais-results"

RANDOM_SEED = 0
random.seed(RANDOM_SEED)

nrVessels = 2000

#path_leo_xyz = "./wider_scenario_2/LEO-XYZ-Pos.csv"
path_leo_xyz = "./panama_scenario/"+str(nrVessels)+"_nodes/SAT_XYZ.csv"

#path_xyz = "./wider_scenario_2/SITES-XYZ-Pos.csv"
path_xyz = "./panama_scenario/"+str(nrVessels)+"_nodes/SITES_XYZ.csv"

path_lla = "./panama_scenario/"+str(nrVessels)+"_nodes/SITES_LAT_LON.csv"

path_leo_lla = "./panama_scenario/"+str(nrVessels)+"_nodes/SAT_LLA.csv"

# =============================================================================
# #path_leo_xyz = "./wider_scenario_2/LEO-XYZ-Pos.csv"
# path_leo_xyz = "./panama_scenario/100_nodes/SAT_XYZ.csv"
# 
# #path_xyz = "./wider_scenario_2/SITES-XYZ-Pos.csv"
# path_xyz = "./panama_scenario/100_nodes/SITES_XYZ.csv"
# 
# path_lla = "./panama_scenario/100_nodes/SITES_LAT_LON.csv"
# 
# path_leo_lla = "./panama_scenario/100_nodes/SAT_LLA.csv"
# =============================================================================


simulation_time = 600     #59 seconds of simulation

nrTS = 4500
umbral = 100           #[m] # ToDo: Cambiar nombre
# Link Budget Params
sensibility = -118
freq = 162.5e6  #took the media of two channels

# Radiation pattern of antennas (Antenna gain for a vessel and for the satellite)
path_antenna_sat = './Antenna/AntennaGain_CrossDipoleOnCubesat.csv'
path_antenna_vessel = './Antenna/AntennaGain_VesselVHFmonopole.csv'
G_sat_Array, G_vessel_Array = LoadAntennaGain(path_antenna_sat, path_antenna_vessel)

#Capture effect
captureThreshold = 30 #6 dB of overpower


logs = []
packetsAtBS = []
# ------------------------------

#### Call some functions to load neccesary files####
#################################################################
# Load the position of the satellite and the position of all vessels
leo_pos, leo_pos_lla, sites_pos_xyz, sites_pos_lla = LoadAllFiles(path_leo_xyz, path_leo_lla, path_xyz, path_lla)

# Reduce the number of ships (only for tests)
sites_pos_xyz = sites_pos_xyz[0:nrVessels,:]
sites_pos_lla = sites_pos_lla[0:nrVessels,:]

# Load all the data into a graph
G = LoadIntoGraph(nrVessels)

# Show the graph
# nx.draw(G, with_labels=True, font_weight='bold', pos=np.stack(list(dict(G.nodes.data('pos_xyz')).values()), axis=0)[:,0:2])

# Filter edges
G = FilterEdges(G, umbral)

# Init all TS
G = InitTS(G, nrTS)

# =============================================================================
# # Show the graph into a map
PlotMap(G, leo_pos_lla)
# =============================================================================


# Pass the site pos of the graph to a numpy array
sites_pos = np.stack(list(dict(G.nodes.data('pos_xyz')).values()), axis=0)

# Calculate distance to satellite for x,y,z for all nodes
dist_sat = calcDistance (leo_pos, sites_pos)

# Estimate slant-range distance to satellite for al nodes
distance = calcSlant(dist_sat)

# Estimate Path Loss for all nodes at any time
Lpl = calcLpl(distance)

# Calculate the angle from zenith to satellite (vessel), and the angle from nadir to vessel (satellite)
Angle_ZenithToSat, Angle_NadirToVessel = calcAngles()

elev = np.degrees(np.arcsin(550/distance)) # ToDo: Descartar
#################################################################

env = simpy.Environment()


for i in range(nrVessels):    
    env.process(transmit(env,G.nodes[i]['obj']))
        
    
env.run(until=simulation_time)

sent = sum(G.nodes[n]['obj'].sent for n in G.nodes)

# Write to file
folder = name+'_s'+str(RANDOM_SEED)

if not os.path.exists(folder):
    os.makedirs(folder)
fname = "./"+folder+"/" + str(name)+"_"+str(nrVessels)+"_s"+str(RANDOM_SEED)+".csv"
                    
with open(fname,"w") as myfile:
    myfile.write("\n".join(logs))
myfile.close()


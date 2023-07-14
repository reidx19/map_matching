#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 15:14:58 2022

@author: tannerpassmore
"""

#%% imports
import osmnx as ox
import pandas as pd
import geopandas as gpd
import networkx as nx
from leuvenmapmatching.matcher.distance import DistanceMatcher
from leuvenmapmatching.map.inmem import InMemMap
from leuvenmapmatching import visualization as mmviz
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
from shapely import geometry, ops
import os
import pickle
#from gps_utils import rdp
import time

import itertools
from operator import itemgetter

import geopandas as gpd
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString

from pathlib import Path

from tqdm import tqdm


#%% functions

#find nearest point
def ckdnearest(gdfA, gdfB, gdfB_cols=['edge_sequence','A_B']):
    #reset index
    gdfA = gdfA.reset_index(drop=True)
    gdfB = gdfB.reset_index(drop=True)
    A = np.concatenate(
        [np.array(geom.coords) for geom in gdfA.geometry.to_list()])
    B = [np.array(geom.coords) for geom in gdfB.geometry.to_list()]
    B_ix = tuple(itertools.chain.from_iterable(
        [itertools.repeat(i, x) for i, x in enumerate(list(map(len, B)))]))
    B = np.concatenate(B)
    ckd_tree = cKDTree(B)
    dist, idx = ckd_tree.query(A, k=1)
    idx = itemgetter(*idx)(B_ix)
    gdf = pd.concat(
        [gdfA, gdfB.loc[idx, gdfB_cols].reset_index(drop=True),
          pd.Series(dist, name='dist_ft')], axis=1)
    return gdf

#%% run

#export filepath
export_fp = Path.home() / 'Downloads/cleaned_trips'

#load dict of traces (replace with database)
with (export_fp/'simp_dict.pkl').open('rb') as fh:
    simp_dict = pickle.load(fh)

#load existing matches/if none then create a new dict
if (export_fp/'matched_traces.pkl').exists():
    with (export_fp/'matched_traces.pkl').open('rb') as fh:
        matched_traces = pickle.load(fh)
else:
    matched_traces = dict()

#load network
network_fp = r"C:\Users\tpassmore6\Documents\TransitSimData\networks\final_network.gpkg"
edges = gpd.read_file(network_fp,layer="links")
nodes = gpd.read_file(network_fp,layer="nodes")

# create network graph needed for map matching
map_con = InMemMap("marta_osm", use_latlon = False)

#redo the latlon columns
nodes['lat'] = nodes.geometry.y
nodes['lon'] = nodes.geometry.x

#NOTE Leuven uses LATLON not LONLAT

#add edges and nodes to leuven graph network object
for idx, row in nodes.iterrows():
    map_con.add_node(row['N'], (row['lat'], row['lon']))
for idx, row in edges.iterrows():
    map_con.add_edge(row['A'], row['B'])
 
#set up matching (find explanations for each parameter)
matcher = DistanceMatcher(map_con,
                     max_dist=1000,  # ft
                     min_prob_norm=0.001,
                     non_emitting_length_factor=0.75,
                     non_emitting_states=True,
                     obs_noise=200,
                     max_lattice_width=5)  

#turn network into dict to quickly retrieve geometries
edges['tup'] = list(zip(edges['A'],edges['B']))
geos_dict = dict(zip(edges['tup'],edges['geometry']))

#loop through each trace in dict unless it has already been matched
for tripid, traces in tqdm(simp_dict.items()):
    if tripid in matched_traces.keys():
        continue
    
    #for testing
    #tripid = 3271
    #traces = simp_dict[tripid]
    
    #redo sequence to match up with reduced number of points
    traces.reset_index(inplace=True)
    traces.drop(columns=['sequence'],inplace=True)
    traces.rename(columns={'index':'sequence'},inplace=True)
    
    #start recording match time
    start = time.time()
        
    #get list of coords
    gps_trace = list(zip(traces.geometry.y,traces.geometry.x))
    
    #perform matching
    states, last_matched = matcher.match(gps_trace)
    match_nodes = matcher.path_pred_onlynodes
    
    #reduce the states size with match_nodes
    reduced_states = []
    for i in range(0,len(match_nodes)-1):
        reduced_states.append((match_nodes[i],match_nodes[i+1]))
    
    #calculate the match ratio
    match_ratio = last_matched / (len(gps_trace)-1)
       
    #retreive matched edges from network
    geos_list = [geos_dict.get(id,0) for id in reduced_states]
    
    #turn into geodataframe
    matched_trip = gpd.GeoDataFrame(data={'A_B':reduced_states,'geometry':geos_list},geometry='geometry',crs='epsg:2240')
    
    #turn tuple to str
    matched_trip['A_B'] = matched_trip['A_B'].apply(lambda row: f'{row[0]}_{row[1]}')
    
    #reset index to add an edge sequence column
    matched_trip.reset_index(inplace=True)
    matched_trip.rename(columns={'index':'edge_sequence'},inplace=True)
    
    # merge the lines for nearest step
    #merged_line = ops.linemerge(geos_list)

    #only find deviance up to last matched
    matched_points = traces.iloc[0:last_matched][['sequence','geometry']]
        
    #calculate distance from matched line
    '''
    We do not know which point matched to which line, so we just find nearest line
    from each point. Will not always be correct.
    '''
    deviation = ckdnearest(matched_points, matched_trip)[['sequence','edge_sequence','A_B','dist_ft']]

    #add to dictionary
    matched_traces[tripid] = {
        'nodes':match_nodes, #list of the matched node ids
        'edges':reduced_states, #list of the matched edge ids
        'last_matched':last_matched, #last gps point reached
        'match_ratio':match_ratio, #percent of points matched
        'deviation': deviation, #how far was point to matched line (df)
        'matched_trip': matched_trip, #gdf of matched lines
        'match_time_sec': time.time() - start, #time it took to match
        'time': time.time() # record when it was last matched
        }
    
    #create png for easy examination (later)
    
    #add deviation to traces
    traces = pd.merge(traces,deviation,on='sequence',how='left')

    #turn datetime to str for exporting
    traces['datetime'] = traces['datetime'].astype(str)
    traces['time_from_start'] = traces['time_from_start'].astype(str)
    
    #export traces and matched line to gpkg for easy examination
    if match_ratio == 1:
        matched_trip.to_file(export_fp/f"matched_traces/complete/{tripid}.gpkg",layer='matched')
        traces.to_file(export_fp/f"matched_traces/complete/{tripid}.gpkg",layer='points')
    elif match_ratio > 0:
        matched_trip.to_file(export_fp/f"matched_traces/mixed/{tripid}.gpkg",layer='matched')
        traces.to_file(export_fp/f"matched_traces/mixed/{tripid}.gpkg",layer='points')
    else:
        matched_trip.to_file(export_fp/f"matched_traces/failed/{tripid}.gpkg",layer='matched')
        traces.to_file(export_fp/f"matched_traces/failed/{tripid}.gpkg",layer='points')
    

#%% Deprecated

# #%% save current matches

# with (export_fp/'matched_traces.pkl').open('wb') as fh:
#     pickle.dump(matched_traces,fh)

# #%%

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", ShapelyDeprecationWarning)
    
# #one gpkg with tripid in filename with layer that has the coordinates + the matched line for comparison
# reduced_matched_traces = {tripid:{'match_ratio':matched_traces[tripid]['match_ratio'],
#                                   'last_matched':last_matched,
#                                   'mean_deveation':matched_traces[tripid]['deviation']['dist_ft'].mean(),
#                                   'geometry':[matched_traces[tripid]['geometry']]} for tripid in matched_traces.keys()
#                           }

# for tripid, matched_trace in reduced_matched_traces.items():
#     gdf = gpd.GeoDataFrame(matched_trace,geometry='geometry',crs='epsg:2240',index=[0])
#     points = simp_dict[tripid]
    
#     #turn datetime to str
#     points['datetime'] = points['datetime'].astype(str)
#     points['time_from_start'] = points['time_from_start'].astype(str)
    
#     if matched_trace['match_ratio'] == 1:
#         gdf.to_file(export_fp/f"matched_traces/complete/{tripid}.gpkg",layer='matched')
#         points.to_file(export_fp/f"matched_traces/complete/{tripid}.gpkg",layer='points')
#     elif matched_trace['match_ratio'] > 0:
#         gdf.to_file(export_fp/f"matched_traces/mixed/{tripid}.gpkg",layer='matched')
#         points.to_file(export_fp/f"matched_traces/mixed/{tripid}.gpkg",layer='points')
#     else:
#         gdf.to_file(export_fp/f"matched_traces/failed/{tripid}.gpkg",layer='matched')
#         points.to_file(export_fp/f"matched_traces/failed/{tripid}.gpkg",layer='points')


# #%%
# #create gdf of all matches
# geos = {key:matched_traces[key]['merged_line'] for key in matched_traces.keys()}
# dev = 
# match_ratio = 
# gdf = gpd.GeoPandas()


# # export raw data
# with open('matched_traces.pickle','wb') as handle:
#     pickle.dump(matched_traces, handle)

# # print out summary statistics on match performance    
# complete_routes, complete_traces, incomplete_routes, incomplete_traces, zero_traces = matching_statistics(matched_traces,routes,traces)
# #scomplete_routes, scomplete_traces, sincomplete_routes, sincomplete_traces, szero_traces = matching_statistics(simp_matched_traces,simp_routes,simplified_traces)

# #export gdfs so you can examine in GIS
# export_gdfs(complete_routes, complete_traces, incomplete_routes, incomplete_traces, zero_traces,'raw')
# #export_gdfs(scomplete_routes, scomplete_traces, sincomplete_routes, sincomplete_traces, szero_traces,'simp')

# #%%  
#     all_geos = {}
#     for key in all_paths.keys():
#         id_list = all_paths[key]
#         geos_list = [geos_dict.get(id,0) for id in id_list]
#         if geos_list != []:
#             all_geos[key] = MultiLineString(geos_list)
#     return all_geos



# #trip list with list of nodes
# matched_traces[idx] = {
#     'nodes':match_nodes,
#     'edges':states,
#     'last_matched':last_matched,
#     'match_ratio':match_ratio,
#     'runtime_s': round(stop - start)
#     }

# #% what do we want from this?
# # 1. the routes as linestrings for ez quality checks
# #get route edges as list of dicts
# route_edges = ox.utils_graph.get_route_edge_attributes(graph_proj, match_nodes)

# #create route
# route = geometry.MultiLineString([x['geometry'] for x in route_edges])

# # merge the lines
# merged_line = ops.linemerge(route)

# #create geodataframe and append to other geodataframe
# gdf = gpd.GeoDataFrame({'tripid':idx,'geometry':[merged_line]}, crs="EPSG:3395")
# routes = routes.append(gdf)

# #define projection for routes
# routes.set_crs("EPSG:3395",inplace=True)

# #total runtime
# total_end = time.perf_counter()
    



# matched_traces, routes = map_matching(paths,'tripid',map_con,graph_proj)    
# #simp_matched_traces, simp_routes = map_matching(simplified_paths,'tripid',map_con,graph_proj) 

# #find deviation from original point to look at accuracy
# matched_traces, distance = find_deviation(matched_traces, traces)
# #simp_matched_traces, simp_distance = find_deviation(simp_matched_traces, simplified_traces)

# # export raw data
# with open('matched_traces.pickle','wb') as handle:
#     pickle.dump(matched_traces, handle)

# # print out summary statistics on match performance    
# complete_routes, complete_traces, incomplete_routes, incomplete_traces, zero_traces = matching_statistics(matched_traces,routes,traces)
# #scomplete_routes, scomplete_traces, sincomplete_routes, sincomplete_traces, szero_traces = matching_statistics(simp_matched_traces,simp_routes,simplified_traces)

# #export gdfs so you can examine in GIS
# export_gdfs(complete_routes, complete_traces, incomplete_routes, incomplete_traces, zero_traces,'raw')
# #export_gdfs(scomplete_routes, scomplete_traces, sincomplete_routes, sincomplete_traces, szero_traces,'simp')







#%% extra stuff

# from shapely.geometry import box

# with (Path.home()/"Documents/GitHub/map_matching/example_trace.pkl").open('wb') as fh:
#     pickle.dump(gps_trace,fh)

# bounds = traces.total_bounds

# bounds = box(bounds[0]-1000,bounds[1]-1000,bounds[2]+1000,bounds[3]+1000)

# example_edges = edges[edges.geometry.intersects(bounds)]
# example_nodes = nodes[nodes['N'].isin(example_edges['A'].append(example_edges['B']))]

# example_edges.to_file(Path.home()/"Documents/GitHub/map_matching/example_network.gpkg",layer='edges')
# example_nodes.to_file(Path.home()/"Documents/GitHub/map_matching/example_network.gpkg",layer='nodes')

#function that runs the map matching
# def map_matching(gps_traces,tripid,map_con,graph_proj):
    
#     #total runtime
#     total_start = time.perf_counter()
    
#     #create routes geodataframe
#     routes = gpd.GeoDataFrame()
#     #create matched traces dict
#     matched_traces = {}

#     #itterate through trips
#     for idx, row in gps_traces.items():
        
#         #start the timer
#         start = time.perf_counter()
        
#         #get coords
#         gps_trace = row

#         #perform match
#         matcher = DistanceMatcher(map_con,
#                              max_dist=250,  # meter
#                              min_prob_norm=0.001,
#                              non_emitting_length_factor=0.75,
#                              non_emitting_states=True,
#                              obs_noise=50,
#                              #obs_noise_ne=75,  # meter
#                              #dist_noise=50,  # meter
#                              max_lattice_width=5)        
            
#         states, last_matched = matcher.match(gps_trace)
#         match_nodes = matcher.path_pred_onlynodes
        
#         #print that the trace has been matched
#         print(f'{gps_traces.index.get_loc(idx)+1}/{len(gps_traces)} matched.')
        
#         #get matching time
#         stop = time.perf_counter()
        
#         #calculate the match ratio
#         match_ratio = last_matched / (len(gps_trace)-1)
        
#         #trip list with list of nodes
#         matched_traces[idx] = {
#             'nodes':match_nodes,
#             'edges':states,
#             'last_matched':last_matched,
#             'match_ratio':match_ratio,
#             'runtime_s': round(stop - start)
#             }
        
#         #% what do we want from this?
#         # 1. the routes as linestrings for ez quality checks
#         #get route edges as list of dicts
#         route_edges = ox.utils_graph.get_route_edge_attributes(graph_proj, match_nodes)

#         #create route
#         route = geometry.MultiLineString([x['geometry'] for x in route_edges])

#         # merge the lines
#         merged_line = ops.linemerge(route)

#         #create geodataframe and append to other geodataframe
#         gdf = gpd.GeoDataFrame({'tripid':idx,'geometry':[merged_line]}, crs="EPSG:3395")
#         routes = routes.append(gdf)
        
#         #define projection for routes
#         routes.set_crs("EPSG:3395",inplace=True)
        
#         #total runtime
#         total_end = time.perf_counter()
        
#     #print runtime
#     print(f'Took {round(total_end - total_start)} seconds to match {len(gps_traces)} trace(s)')
    
#     return matched_traces, routes

# # find deviation distance from matched point to nearest matched line
# def find_deviation(matched_traces, traces, links):
#     '''
#     Take in the matched_traces and original traces along with the network data,
#     and find the distance for the matches (point to line)

#     '''
#     distance = pd.DataFrame()

#     for key in matched_traces:
        
#         #get edges from network graph
#         gpd2 = ox.utils_graph.get_route_edge_attributes(graph_proj, matched_traces[key]['nodes'])
    
#         #get line geometries
#         geom = [x['geometry'] for x in gpd2]
#         edgeid = [x['osmid'] for x in gpd2]
    
#         #create geodataframe of the lines
#         gdfB = gpd.GeoDataFrame({'osmid':edgeid,'geometry':geom})
        
#         #create geodataframe of the points
#         gdfA = traces[traces['tripid']==key]
        
#         #only match up to last matched
#         gdfA = gdfA[gdfA['sequence'] <= matched_traces[key]['last_matched']]
        
#         #perfom matching
#         c = ckdnearest(gdfA, gdfB)
    
#         #add to dict
#         matched_traces[key]['mean_deviation_m'] = c.dist_m.mean()
    
#         #add to match dataframe
#         #think about just adding this to the dict
#         distance = distance.append(c)
        
#     return matched_traces, distance

# def matching_statistics(matched_traces,routes,traces):

#     #get match ratio (already calculated)
#     match_ratio = [matched_traces[x]['match_ratio'] for x in matched_traces]
#     tripid = [x for x in matched_traces ]
#     match_ratio = pd.DataFrame(data={'tripid':tripid,'match_ratio':match_ratio})
    
#     avg_match_ratio = match_ratio.match_ratio.mean()
    
#     #get complete matches
#     complete_matches = match_ratio[match_ratio['match_ratio']==1]
#     #get the routes and traces to look at in QGIS
#     complete_routes = routes[routes['tripid'].isin(complete_matches['tripid'])]
#     complete_traces = traces[traces['tripid'].isin(complete_matches['tripid'])]
    
#     #get incomplete matches
#     incomplete_matches = match_ratio[(match_ratio['match_ratio']!=1) & (match_ratio['match_ratio']!=0) ]
#     #get the routes and traces to look at in QGIS
#     incomplete_routes = routes[routes['tripid'].isin(incomplete_matches['tripid'])]
#     incomplete_traces = traces[traces['tripid'].isin(incomplete_matches['tripid'])]
    
#     #zero match
#     zero_matches = match_ratio[match_ratio['match_ratio']==0] 
#     zero_traces = traces[traces['tripid'].isin(zero_matches['tripid'])] 
    
#     print(f'{len(complete_matches)} out of {len(matched_traces)} were complete matches.')
#     print(f'{len(zero_matches)} out of {len(matched_traces)} had no matches')
    
#     print(f'Average match ratio of {round(avg_match_ratio,2)}')
    
#     #average match time
#     match_time = [matched_traces[x]['runtime_s'] for x in matched_traces]
#     tripid = [x for x in matched_traces ]
#     match_time = pd.DataFrame(data={'tripid':tripid,'match_time':match_time})
    
#     print(f'Average match time of {round(match_time.match_time.mean(),2)} seconds')
#     print(f'Total match time of {round(match_time.match_time.sum() / 60,2)} minutes')
    
#     return complete_routes, complete_traces, incomplete_routes, incomplete_traces, zero_traces

# def export_gdfs(cr,ct,ir,it,zt,filename):
#     cr.to_file(rf'matched/match_results_{filename}.gpkg',layer='complete_routes',driver='GPKG')
#     ct.to_file(rf'matched/match_results_{filename}.gpkg',layer='complete_traces',driver='GPKG')
#     ir.to_file(rf'matched/match_results_{filename}.gpkg',layer='incomplete_routes',driver='GPKG')
#     it.to_file(rf'matched/match_results_{filename}.gpkg',layer='incomplete_traces',driver='GPKG')
#     zt.to_file(rf'matched/match_results_{filename}.gpkg',layer='zero_traces',driver='GPKG')



# #create simplified traces
# #turn into linestrings
# def simplify_traces(points,tolerance_m):
    
#     #turn all trips into lines
#     lines = points.sort_values(by=['tripid','datetime']).groupby('tripid')['geometry'].apply(lambda x: LineString(x.tolist()))
    
#     #turn into gdf
#     lines = gpd.GeoDataFrame({'geometry':lines}, geometry='geometry',crs="epsg:3395")
    
#     #simplify using douglass-peucker algo 
#     lines['geometry'] = lines.simplify(tolerance_m, preserve_topology = False)
    
#     #break line strings into points
#     x = lines.apply(lambda x: [y for y in x['geometry'].coords.xy[0]], axis=1)
#     y = lines.apply(lambda x: [y for y in x['geometry'].coords.xy[1]], axis=1)
    
#     #break list of points into rows
#     x = x.explode()
#     y = y.explode()
    
#     #create dataframe
#     si = pd.DataFrame({'x':x,'y':y}).reset_index()
    
#     #create tuple for merging on
#     si['geometry_tup'] = [(xy) for xy in zip(si.x,si.y)]
#     traces['geometry_tup'] = [(xy) for xy in zip(traces.geometry.x,traces.geometry.y)]
    
#     #merge
#     simplified_traces = pd.merge(traces,si[['tripid','geometry_tup']], on=['tripid','geometry_tup'])
    
#     #drop the tuple
#     simplified_traces.drop(columns=['geometry_tup'],inplace=True)
    
#     #print avg reduction in points
#     dif = traces.tripid.value_counts() - simplified_traces.tripid.value_counts()
    
#     print(f'{dif.mean()} points removed on average.')
    
#     return simplified_traces


# uncomment to show debug info
# # #debug
# # import sys
# # import logging
# # import leuvenmapmatching
# # logger = leuvenmapmatching.logger

# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler(sys.stdout))

# def export_select_trips(tripid,routes,traces):

    
#     route = routes[routes['tripid']==tripid]
#     trace = traces[traces['tripid']==tripid]
    
#     #export them
#     route.to_file(r'matched/selected_examples/selected_trips.gpkg',layer=f'{tripid}_route',driver='GPKG')
#     trace.to_file(r'matched/selected_examples/selected_trips.gpkg',layer=f'{tripid}_trace',driver='GPKG')
    

# def make_maps(tripid,routes,traces,edges):
        
#     #subset data
#     route = routes[routes['tripid']==tripid]
#     trace = traces[traces['tripid']==tripid]
        
#     #find bounding box for traces (it'll usually be bigger)
#     xmin, ymin, xmax, ymax = trace.geometry.total_bounds
    
#     #add tolerances
#     tolerance = 100
#     xmin = xmin - tolerance
#     ymin = ymin - tolerance
#     xmax = xmax + tolerance
#     ymax = ymax + tolerance
    
#     #create figures
#     fig, ax = plt.subplots()
    
#     #set limits
#     xlim = (xmin,xmax)
#     ylim = (ymin,ymax)
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
    
#     #get start and end point
#     start = trace.iloc[[0]]
#     end = trace.iloc[[-1]]
    
#     #plot
#     start.plot(ax=ax,zorder=4,color='green')
#     end.plot(ax=ax,zorder=5,color='red')
#     route.plot(ax=ax,zorder=3,color='lightcoral')
#     trace.plot(ax=ax,zorder=2,markersize=0.5, color='black')
#     edges.plot(ax=ax,zorder=1,color="lightgrey",linewidth=0.25)
    
#     ax.set(title=f'Trip ID: {tripid}')
#     ax.set_axis_off()
#     plt.show()
#     fig.savefig(rf'matched/selected_examples/trip{tripid}.png',dpi=300)
   
# #trips_of_interest = [11166,11202,9069,10803]
# trips_of_interest = [10446]


# #run functions

# for trip in trips_of_interest:
#     export_select_trips(trip, routes, traces)
#     make_maps(trip, routes, traces, edges)
   
   
# #%%

# #plot road network, traces, and route
# complete_routes[complete_routes['tripid']==11166].plot(ax=ax0,zorder=2)
# complete_traces[complete_traces['tripid']==11166].plot(ax=ax1,zorder=3)
# edges.plot(ax=ax0,zorder=1,color="grey",alp)


# #%% export routes for examination
# #save raw outputs just in case
# with open('matched_traces.pickle','wb') as f:
#     pickle.dump(matched_traces,f)

# #define projection
# routes.set_crs("EPSG:3395",inplace=True)
# routes.to_file(r'leuven/50_matched_routes.gpkg',driver='GPKG')

# #what didn't match
# nomatch = routes[routes.geometry.is_empty]['tripid'].to_list()

# #filter gps points
# nomatch_traces = traces[traces['tripid'].isin(nomatch)]
# didmatch_traces = traces[-traces['tripid'].isin(nomatch)]

# #export
# nomatch_traces[['tripid','geometry']].to_file(r'leuven/unmatched_traces.gpkg',driver='GPKG')
# didmatch_traces[['tripid','geometry']].to_file(r'leuven/matched_traces.gpkg',driver='GPKG')

# #%% export aggregated results

# #first get rid of trip id columns that didn't match
# didmatch = routes[-routes.geometry.is_empty]['tripid'].to_list()
# mask = ['u','v','uv'] + didmatch
# result_network_clean = result_network[mask] 

# #create a sum column and add rows
# result_network_clean['total_riders'] = result_network_clean.iloc[:,3:].sum(axis = 1)

# #bring in user info to add to columns
# userinfo = pd.read_csv('sample_trip_and_user_50.csv')

# #filter out the ones that didn't match
# userinfo_clean = userinfo[userinfo['tripid'].isin(didmatch)]

# #desired attributes
# attr = ['trip_type','age','gender','income','ethnicity','cyclingfreq','rider_history','rider_type']

# #get first row
# first_row = userinfo_clean.loc[0]

# #get tripid
# first_row.tripid

# userinfo_clean.pivot(index=userinfo_clean.columns.to_list()[1:],columns='tripid')

# #transpose data
# transposed = userinfo_clean.T

# #make tripid row the column names
# transposed.columns = transposed.loc['tripid']

# #create columns
# userinfo_clean['']


# df = first_row.to_frame().reset_index()

# result_network.loc[result_network[first_row.tripid]==1,] = [1] + first_row[attr].to_list()

# #pivot so that trip id is the first row



# result_network.loc[result_network.uv.isin(states),idx] = 1


# #add as columns to result network

# result_network_clean = 

# #join to edges and export to gpkg
# edges_w_riders = pd.merge(edges,result_network_clean,on=['u','v'])

# #%% outside of function

# #desired path
# #path = paths.loc[7712]
# path = paths.loc[10315]

# matcher = DistanceMatcher(map_con,
#                              max_dist=100,  # meter
#                              min_prob_norm=0.001,
#                              non_emitting_length_factor=0.75,
#                              obs_noise=50, obs_noise_ne=75,  # meter
#                              dist_noise=50,  # meter
#                              non_emitting_states=True,
#                              max_lattice_width=10)
# states, _ = matcher.match(path)
# match_nodes = matcher.path_pred_onlynodes

# print("States\n------")
# print(states)
# print("Nodes\n------")
# print(match_nodes)
# print("")
# matcher.print_lattice_stats()


# #%% plot match
# mmviz.plot_map(map_con, matcher=matcher,
#               show_labels=True, show_matching=True, show_graph=False,
#               filename=r'C:/Users/tpassmore6/Documents/BikewaySimData/gps_traces_for_matching/output.png')


# #%% get paths

# #plots the path, not neccessary
# ox.plot_graph_route(graph_proj, match_nodes)

# #get route edges as list of dicts
# test = ox.utils_graph.get_route_edge_attributes(graph_proj, match_nodes)

# #create route
# route = geometry.MultiLineString([x['geometry'] for x in test])

# # merge the lines
# merged_line = ops.linemerge(route)

# # create geodataframe
# gdf = gpd.GeoDataFrame({'tripid':[1],'geometry':[merged_line]}, crs="EPSG:3395")

# # export as geojson
# gdf.to_file(r'C:/Users/tpassmore6/Documents/BikewaySimData/gps_traces_for_matching/one_sample.geojson',driver='GeoJSON')

# #%% export match network

# ox.save_graph_geopackage(graph_proj,r'C:/Users/tpassmore6/Documents/BikewaySimData/gps_traces_for_matching/leuven_network.gpkg')

# edges.to_file(,driver='GeoJSON')

# #%% deprecated
# # %% check work to make sure things line up

# # fix, ax = plt.subplots()

# # ax.set_aspect('equal')
# # traces[traces['tripid']==11125].plot(ax=ax, color='yellow', zorder=2)
# # edges.plot(ax=ax, color='grey', zorder = 1)


# # plt.show()

# # #%% plot example path
# # fix, ax = plt.subplots()

# # ax.set_aspect('equal')
# # example.plot(ax=ax, color='yellow', zorder=2)
# # edges.plot(ax=ax, color='grey', zorder = 1)


# # plt.show()

# #%% try this one



# #check what dropping duplicates really does
# # result_network = edges[['u','v']].drop_duplicates()

# # #turn to tuple
# # result_network['uv'] = [(xy) for xy in zip(result_network.u,result_network.v)]

# #idx = 11125

# #create trip column
# #result_network[idx] = 0

# #turn to 1 if edge present
# #result_network.loc[result_network.uv.isin(states),idx] = 1



# #%%
# # from pathlib import Path
# # import requests
# # xml_file = Path("C:/Users/tpassmore6/Documents/BikewaySimData/gps_example_for_matching") / "osm.xml"
# # url = f'http://overpass-api.de/api/map?bbox={minx},{miny},{maxx},{maxy}'
# # r = requests.get(url, stream=True)
# # with xml_file.open('wb') as ofile:
# #     for chunk in r.iter_content(chunk_size=1024):
# #         if chunk:
# #             ofile.write(chunk)
            
# # #%%
# # from leuvenmapmatching.map.inmem import InMemMap
# # import osmread

# # map_con = InMemMap("myosm", use_latlon=True, use_rtree=True, index_edges=True)
# # for entity in osmread.parse_file(str(xml_file)):
# #     if isinstance(entity, osmread.Way) and 'highway' in entity.tags:
# #         for node_a, node_b in zip(entity.nodes, entity.nodes[1:]):
# #             map_con.add_edge(node_a, node_b)
# #             # Some roads are one-way. We'll add both directions.
# #             map_con.add_edge(node_b, node_a)
# #     if isinstance(entity, osmread.Node):
# #         map_con.add_node(entity.id, (entity.lat, entity.lon))
# # map_con.purge()

# #%% study area bounding box

# # #study area boundaries
# # study_area = gpd.read_file(r'C:/Users/tpassmore6/Documents/BikewaySimData/processed_shapefiles/study_areas/study_area.geojson').to_crs("EPSG:4326")

# # #get bounding box
# # bbox = study_area.dissolve().bounds

# # maxy = bbox.maxy[0]
# # miny = bbox.miny[0]
# # maxx = bbox.maxx[0]
# # minx = bbox.minx[0]




#%% function for map matching set of trips

#gps_trace = series with gps trace latlon and tripid
#tripid = column with trip ids
#map_con = the network graph
#graph_proj = the osm data


# def map_matching_single(gps_trace,map_con,graph_proj):
#         #perform match
#         matcher = DistanceMatcher(map_con,
#                               max_dist=250,  # meter
#                               min_prob_norm=0.001,
#                               non_emitting_length_factor=0.75,
#                               non_emitting_states=True,
#                               obs_noise=50,
#                               #obs_noise_ne=75,  # meter
#                               #dist_noise=50,  # meter
#                               max_lattice_width=20)     
#         states, last_matched = matcher.match(gps_trace)
#         match_nodes = matcher.path_pred_onlynodes
        
#         return states, last_matched, match_nodes

# gps_trace = paths[10803]
# states, last_matched, match_nodes = map_matching_single(gps_trace, map_con, graph_proj)

# len(gps_trace)
# len(states)
# len(match_nodes)


# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 12:09:45 2021

@author: tpassmore6
"""

#%% import
import pandas as pd
import geopandas as gpd
import fiona
import glob
from shapely.geometry import shape, Point, LineString
import os

#%%funictions

def turn_into_linestring(points):
    #turn all trips into lines
    lines = points.sort_values(by=['tripid','datetime']).groupby('tripid')['geometry'].apply(lambda x: LineString(x.tolist()))

    #get start time
    start_time = points.groupby('tripid')['datetime'].min()

    #get end time
    end_time = points.groupby('tripid')['datetime'].max()
    
    #turn into gdf
    linestrings = gpd.GeoDataFrame({'start_time':start_time,'end_time':end_time,'geometry':lines}, geometry='geometry',crs="epsg:4326")
    
    return linestrings

#%% run

#clean the trajectories to prepare for map matching
if __name__ == '__main__':

    #set working directory
    directory = r"C:/Users/tpassmore6/Documents/BikewaySimData/gps_traces_for_matching/"
    os.chdir(directory)

    #set location of gps trajectories and list of all coord csvs
    path = r'C:/Users/tpassmore6/Documents/ridership_data/fromchris/CycleAtlanta/CycleAtlanta/9-10-16 Trip Lines and Data/raw data/'
    coords_files = glob.glob(path + "coord*.csv", recursive=True)

    # import and clean
    all_coords = pd.DataFrame()

    #bring in all the coords csvs
    # for x in coords_files:
    #     coords = pd.read_csv(x, header=None)
    #     all_coords = all_coords.append(coords)

    #or just bring in one
    all_coords = pd.read_csv(coords_files[0],header=None)
    
    #rename columns
    col_names = ['tripid','datetime','lat','lon','altitude','speed','hAccuracy','vAccuracy']
    all_coords.columns = col_names
    
    #change dates to datetime
    all_coords['datetime'] = pd.to_datetime(all_coords['datetime'])
    
    #change trip id to str
    all_coords['tripid'] = all_coords['tripid'].astype(str)
    
    #drop anything with one or fewer points
    numofpoints = all_coords.groupby('tripid').size()
    idx = numofpoints[numofpoints > 1].index.to_list()
    all_coords = all_coords[all_coords['tripid'].isin(idx)]
    
    #use datetime to create sequence column
    all_coords['sequence'] = all_coords.groupby(['tripid']).cumcount()
    
    #make total points column for reference
    tot_points = all_coords['tripid'].value_counts().rename('tot_points')
    all_coords = pd.merge(all_coords, tot_points, left_on='tripid',right_index=True)
    
    #add geometry info
    all_coords['geometry'] = gpd.points_from_xy(all_coords['lon'],all_coords['lat'])
    
    #turn into geodataframe
    all_coords = gpd.GeoDataFrame(all_coords,geometry='geometry',crs=4326)
    
    # import trip info
    trip = pd.read_csv(path+"trip.csv", header = None)
    col_names = ['tripid','userid','trip_type','description','starttime','endtime','notsure']
    trip.columns = col_names

    # these don't seem too accurate
    # #convert to datetime
    # trip['starttime'] = pd.to_datetime(trip['starttime'])
    # trip['endtime'] = pd.to_datetime(trip['endtime'])
    
    # #trip time
    # trip['triptime'] = trip['endtime'] - trip['starttime']

    #drop these
    trip.drop(columns=['description','notsure','starttime','endtime'], inplace = True)
    
    #change tripid and userid to str
    trip['tripid'] = trip['tripid'].astype(str)
    trip['userid'] = trip['userid'].astype(str)
    
    # import user info and filter columns
    user = pd.read_csv(path+"user.csv", header=None)
    user_col = ['userid','created_date','device','email','age','gender','income','ethnicity','homeZIP','schoolZip','workZip','cyclingfreq','rider_history','rider_type','app_version']
    user.columns = user_col
    user.drop(columns=['device','app_version','app_version','email'], inplace=True)
    user['userid'] = user['userid'].astype(str)

    # merge trip and users
    #join the user information with trip information
    trip_and_user = pd.merge(trip,user,on='userid')
    
    # import notes (optional)
    #note = pd.read_csv(path+"note.csv", header=None)
    
    # pre-filter for creating sample dataset
    #only get traces that cross inputed borders
    coa = gpd.read_file(r'C:/Users/tpassmore6/Documents/BikewaySimData/base_shapefiles/bikewaysim_study_area/bikewaysim_study_area.shp')
    
    #dissolve points by trip id
    all_coords_dissolved = all_coords.dissolve('tripid').reset_index()
    
    #find traces that are completely within the coa
    #trips_within = all_coords_dissolved.sjoin(coa,predicate='crosses')['tripid'] # use if just crosses
    trips_within = all_coords_dissolved.sjoin(coa,predicate='within')['tripid']
    
    #only keep original columns
    all_coords = all_coords[all_coords['tripid'].isin(trips_within)]
    
    #use all
    n = all_coords['tripid'].nunique()
    random_trips = all_coords['tripid']
    sample_coords = all_coords
    
    #export gdf
    sample_coords.to_file(rf'sample_trips/sample_coords_{n}.geojson',driver='GeoJSON')
    
    #get user table
    list_of_trips = sample_coords['tripid'].astype(str).drop_duplicates()
    
    #now get user/trip info
    #drop trips that aren't represented
    sample_trip_and_user = trip_and_user[trip_and_user['tripid'].isin(random_trips)]
    
    #total datapoints
    sample_trip_and_user['tot_points'] = pd.merge(sample_trip_and_user, tot_points, left_on='tripid',right_index=True)['tot_points']
    
    #average speed
    speed_stats = sample_coords.groupby('tripid').agg({'speed':['mean','median','max']})
    speed_stats.columns = ['_'.join(col).strip() for col in speed_stats.columns.values]
    sample_trip_and_user = pd.merge(sample_trip_and_user,speed_stats,left_on='tripid',right_index=True)

    #average distance
    lines = turn_into_linestring(sample_coords)
    
    #project
    lines.to_crs(epsg='2240',inplace=True)
    mean_dist_ft = lines.geometry.length.mean()
    median_dist = lines.geometry.length.median()
    
    #time difference
    #get min, max index
    idx_max = sample_coords.groupby('tripid')['datetime'].transform(max) == sample_coords['datetime']
    idx_min = sample_coords.groupby('tripid')['datetime'].transform(min) == sample_coords['datetime']
    
    #get min, max dfs
    coords_max = sample_coords[idx_max][['tripid','datetime']].rename(columns={'datetime':'maxtime'})
    coords_min = sample_coords[idx_min][['tripid','datetime']].rename(columns={'datetime':'mintime'})
    
    #join these
    coords_dif = pd.merge(coords_max, coords_min, on='tripid')
    
    #find diffrence
    coords_dif['duration'] = coords_dif['maxtime'] - coords_dif['mintime']
    
    #add to trip and user df
    sample_trip_and_user = pd.merge(sample_trip_and_user, coords_dif[['tripid','duration']], on='tripid')
    
    #drop geometry
    sample_coords.drop(columns=['geometry'],inplace=True)
    #export csv
    sample_coords.to_csv(rf'sample_trips/sample_coords_{n}.csv',index=False)
    
    #export user
    sample_trip_and_user.to_csv(rf'sample_trips/sample_trip_and_user_{n}.csv',index=False)



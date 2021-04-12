#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Read obs from NDBC and Co-opc
Generate csv data files
functions to write and read the data
Observations will be chosen based on the hurricane track to avoid more than neccessary download.

"""

__author__ = "Saeed Moghimi"
__copyright__ = "Copyright 2018, UCAR/NOAA"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "moghimis@gmail.com"
#
import pandas as pd
import numpy as np
#
from bs4 import BeautifulSoup
import requests
import lxml.html
import sys,os
#
from pyoos.collectors.ndbc.ndbc_sos import NdbcSos
from pyoos.collectors.coops.coops_sos import CoopsSos
from retrying import retry
import datetime
import dateparser

import cf_units
from io import BytesIO
from ioos_tools.ioos import collector2table
import pickle 
import arrow

#

sys.path.append('/disks/NASARCHIVE/saeed_moghimi/opt/python-packages/')

import geopandas as gpd

from shapely.geometry import LineString
#####################################################

if 'base_info' in sys.modules:  
    del(sys.modules["base_info"])
from base_info import *

#


##### Fucnstions #####
def url_lister(url):
    urls = []
    connection = urlopen(url)
    dom = lxml.html.fromstring(connection.read())
    for link in dom.xpath('//a/@href'):
        urls.append(link)
    return urls

#################
def download(url, path, fname):
    sys.stdout.write(fname + '\n')
    if not os.path.isfile(path):
        urlretrieve(
            url,
            filename=path,
            reporthook=progress_hook(sys.stdout)
        )
        sys.stdout.write('\n')
        sys.stdout.flush()

#################
def progress_hook(out):
    """
    Return a progress hook function, suitable for passing to
    urllib.retrieve, that writes to the file object *out*.
    """

    def it(n, bs, ts):
        got = n * bs
        if ts < 0:
            outof = ''
        else:
            # On the last block n*bs can exceed ts, so we clamp it
            # to avoid awkward questions.
            got = min(got, ts)
            outof = '/%d [%d%%]' % (ts, 100 * got // ts)
        out.write("\r  %d%s" % (got, outof))
        out.flush()
    return it


#################
def get_nhc_storm_info (year,name):
    """
    
    """

    print('Read list of hurricanes from NHC based on year')
    
    if int(year) < 2008:  
        print ('  ERROR:   GIS Data is not available for storms before 2008 ')
        sys.exit('Exiting .....')
    
     
    url = 'http://www.nhc.noaa.gov/gis/archive_wsurge.php?year='+year

    r = requests.get(url,headers=headers,verify=False)

    soup = BeautifulSoup(r.content, 'lxml')

    table = soup.find('table')
    #table = [row.get_text().strip().split(maxsplit=1) for row in table.find_all('tr')]

    tab = []
    for row in table.find_all('tr'):
        tmp = row.get_text().strip().split()
        tab.append([tmp[0],tmp[-1]])
    
    print (tab)   

    df = pd.DataFrame(
        data=tab[:],
        columns=['identifier', 'name'],
    ).set_index('name')


    ###############################

    print('  > based on specific storm go fetch gis files')
    hid = df.to_dict()['identifier'][name.upper()]
    al_code = ('{}'+year).format(hid)
    hurricane_gis_files = '{}_5day'.format(al_code)
    
    return al_code,hurricane_gis_files

#################
#@retry(stop_max_attempt_number=5, wait_fixed=3000)
def download_nhc_gis_files(hurricane_gis_files):
    """
    """
    
    base = os.path.abspath(
        os.path.join(os.path.curdir, 'data', hurricane_gis_files)
    )
    
    if len (glob(base+'/*')) < 1:
        nhc = 'http://www.nhc.noaa.gov/gis/forecast/archive/'

        # We don't need the latest file b/c that is redundant to the latest number.
        fnames = [
            fname for fname in url_lister(nhc)
            if fname.startswith(hurricane_gis_files) and 'latest' not in fname
        ]

        if not os.path.exists(base):
            os.makedirs(base)

        for fname in fnames:
            path1 = os.path.join(base, fname)
            if not os.path.exists(path1):
                url = '{}/{}'.format(nhc, fname)
                download(url, path1,fname)

    return base
    #################################
    

def read_advisory_cones_info(hurricane_gis_files,base,year,code):
    print('  >  Read cones shape file ...')

    cones, points = [], []
    for fname in sorted(glob(os.path.join(base, '{}_*.zip'.format(hurricane_gis_files)))):
        number = os.path.splitext(os.path.split(fname)[-1])[0].split('_')[-1]
        
        # read cone shapefiles
        
        if int(year) < 2014:
            #al092008.001_5day_pgn.shp
            divd =  '.'
        else:
            divd =  '-'
        
        pgn = gpd.read_file(
            ('/{}'+divd+'{}_5day_pgn.shp').format(code, number),
            vfs='zip://{}'.format(fname)
        )
        cones.append(pgn)
        
        #read points shapefiles
        pts = gpd.read_file(
            ('/{}'+divd+'{}_5day_pts.shp').format(code, number),
            vfs='zip://{}'.format(fname)
        )
        # Only the first "obsevartion."
        points.append(pts.iloc[0])
    
    return cones,points,pts

#################


#################
@retry(stop_max_attempt_number=5, wait_fixed=3000)
def get_coops(start, end, sos_name, units, bbox,datum='NAVD', verbose=True):
    """
    function to read COOPS data
    We need to retry in case of failure b/c the server cannot handle
    the high traffic during hurricane season.
    """
    collector = CoopsSos()
    collector.set_bbox(bbox)
    collector.end_time = end
    collector.start_time = start
    collector.variables = [sos_name]
    ofrs = collector.server.offerings
    title = collector.server.identification.title
    config = dict(
        units=units,
        sos_name=sos_name,
        datum = datum,            ###Saeed added     ["MLLW","MSL","MHW","STND","IGLD", "NAVD"]
    )

    data = collector2table(
        collector=collector,
        config=config,
        col='{} ({})'.format(sos_name, units.format(cf_units.UT_ISO_8859_1))
    )

    # Clean the table.
    table = dict(
        station_name = [s._metadata.get('station_name') for s in data],
        station_code = [s._metadata.get('station_code') for s in data],
        sensor       = [s._metadata.get('sensor')       for s in data],
        lon          = [s._metadata.get('lon')          for s in data],
        lat          = [s._metadata.get('lat')          for s in data],
        depth        = [s._metadata.get('depth', 'NA')  for s in data],
    )

    table = pd.DataFrame(table).set_index('station_name')
    if verbose:
        print('Collector offerings')
        print('{}: {} offerings'.format(title, len(ofrs)))
    return data, table

#################
@retry(stop_max_attempt_number=5, wait_fixed=3000)
def get_ndbc(start, end, bbox , sos_name='waves',datum='MSL', verbose=True):
    """
    function to read NBDC data
    ###################
    sos_name = waves    
    all_col = (['station_id', 'sensor_id', 'latitude (degree)', 'longitude (degree)',
           'date_time', 'sea_surface_wave_significant_height (m)',
           'sea_surface_wave_peak_period (s)', 'sea_surface_wave_mean_period (s)',
           'sea_surface_swell_wave_significant_height (m)',
           'sea_surface_swell_wave_period (s)',
           'sea_surface_wind_wave_significant_height (m)',
           'sea_surface_wind_wave_period (s)', 'sea_water_temperature (c)',
           'sea_surface_wave_to_direction (degree)',
           'sea_surface_swell_wave_to_direction (degree)',
           'sea_surface_wind_wave_to_direction (degree)',
           'number_of_frequencies (count)', 'center_frequencies (Hz)',
           'bandwidths (Hz)', 'spectral_energy (m**2/Hz)',
           'mean_wave_direction (degree)', 'principal_wave_direction (degree)',
           'polar_coordinate_r1 (1)', 'polar_coordinate_r2 (1)',
           'calculation_method', 'sampling_rate (Hz)', 'name'])
    
    sos_name = winds    

    all_col = (['station_id', 'sensor_id', 'latitude (degree)', 'longitude (degree)',
       'date_time', 'depth (m)', 'wind_from_direction (degree)',
       'wind_speed (m/s)', 'wind_speed_of_gust (m/s)',
       'upward_air_velocity (m/s)', 'name'])

    """
    #add remove from above
    if   sos_name == 'waves':
            col = ['sea_surface_wave_significant_height (m)','sea_surface_wave_peak_period (s)',
                   'sea_surface_wave_mean_period (s)','sea_water_temperature (c)',
                   'sea_surface_wave_to_direction (degree)']
    elif sos_name == 'winds':
            col = ['wind_from_direction (degree)','wind_speed (m/s)',
                   'wind_speed_of_gust (m/s)','upward_air_velocity (m/s)']
   

    #if   sos_name == 'waves':
    #        col = ['sea_surface_wave_significant_height (m)']
    #elif sos_name == 'winds':
    #        col = ['wind_speed (m/s)']


    collector = NdbcSos()
    collector.set_bbox(bbox)
    collector.start_time = start

    collector.variables = [sos_name]
    ofrs = collector.server.offerings
    title = collector.server.identification.title
    
    collector.features = None
    collector.end_time = start + datetime.timedelta(1)
    response = collector.raw(responseFormat='text/csv')
    
    
    df = pd.read_csv(BytesIO(response), parse_dates=True)
    g = df.groupby('station_id')
    df = dict()
    for station in g.groups.keys():
        df.update({station: g.get_group(station).iloc[0]})
    df = pd.DataFrame.from_dict(df).T
    
    station_dict = {}
    for offering in collector.server.offerings:
        station_dict.update({offering.name: offering.description})
    
    names = []
    for sta in df.index:
        names.append(station_dict.get(sta, sta))
    
    df['name'] = names
    
    #override short time
    collector.end_time = end
    
    data = []
    for k, row in df.iterrows():
        station_id = row['station_id'].split(':')[-1]
        collector.features = [station_id]
        response = collector.raw(responseFormat='text/csv')
        kw = dict(parse_dates=True, index_col='date_time')
        obs = pd.read_csv(BytesIO(response), **kw).reset_index()
        obs = obs.drop_duplicates(subset='date_time').set_index('date_time')
        series = obs[col]
        series._metadata = dict(
            station=row.get('station_id'),
            station_name=row.get('name'),
            station_code=str(row.get('station_id').split(':')[-1]),
            sensor=row.get('sensor_id'),
            lon=row.get('longitude (degree)'),
            lat=row.get('latitude (degree)'),
            depth=row.get('depth (m)'),
        )
    
        data.append(series)
    
    
    # Clean the table.
    table = dict(
        station_name = [s._metadata.get('station_name') for s in data],
        station_code = [s._metadata.get('station_code') for s in data],
        sensor       = [s._metadata.get('sensor')       for s in data],
        lon          = [s._metadata.get('lon')          for s in data],
        lat          = [s._metadata.get('lat')          for s in data],
        depth        = [s._metadata.get('depth', 'NA')  for s in data],
    )
    

    table = pd.DataFrame(table).set_index('station_name')
    if verbose:
        print('Collector offerings')
        print('{}: {} offerings'.format(title, len(ofrs)))
    
    return data, table


#################
def write_csv(obs_dir, name, year, table, data, label):
    """
    examples
    print('  > write csv files')
    write_csv(obs_dir, name, year, table=wnd_ocn_table, data= wnd_ocn , label='ndbc_wind' )
    write_csv(obs_dir, name, year, table=wav_ocn_table, data= wav_ocn , label='ndbc_wave' )
    write_csv(obs_dir, name, year, table=ssh_table    , data= ssh     , label='coops_ssh' )
    write_csv(obs_dir, name, year, table=wnd_obs_table, data= wnd_obs , label='coops_wind')
    
    """
    #label   = 'coops_ssh'
    #table   = ssh_table
    #data    = ssh

    outt    = os.path.join(obs_dir, name+year,label)
    outd    = os.path.join(outt,'data')  
    if not os.path.exists(outd):
        os.makedirs(outd)

    table.to_csv(os.path.join(outt,'table.csv'))
    stations = table['station_code']

    for ista in range(len(stations)):
        sta   = str(stations [ista])
        fname = os.path.join(outd,sta+'.csv')
        df = data[ista]
        try:
            #in case it is still a series like ssh
            df = df.to_frame()
        except:
            pass
                
        df.to_csv(fname)
        
        fmeta    = os.path.join(outd,sta)+'_metadata.csv'
        metadata = pd.DataFrame.from_dict( data[ista]._metadata , orient="index")
        metadata.to_csv(fmeta)
     
def read_csv(obs_dir, name, year, label):
    """
    examples
    print('  > write csv files')
    write_csv(base_dir, name, year, table=wnd_ocn_table, data= wnd_ocn , label='ndbc_wind' )
    write_csv(base_dir, name, year, table=wav_ocn_table, data= wav_ocn , label='ndbc_wave' )
    write_csv(base_dir, name, year, table=ssh_table    , data= ssh     , label='coops_ssh' )
    write_csv(base_dir, name, year, table=wnd_obs_table, data= wnd_obs , label='coops_wind')
    
    """
    outt    = os.path.join(obs_dir, name+year,label)
    outd    = os.path.join(outt,'data')  
    if not os.path.exists(outd):
       sys.exit('ERROR: check path to: ',outd )

    table = pd.read_csv(os.path.join(outt,'table.csv')).set_index('station_name')
    table['station_code'] = table['station_code'].astype('str')
    stations = table['station_code']

    data     = []
    metadata = []
    for ista in range(len(stations)):
        sta   = stations [ista]
        fname8 = os.path.join(outd,sta)+'.csv'
        df = pd.read_csv(fname8,parse_dates = ['date_time']).set_index('date_time')
        
        fmeta = os.path.join(outd,sta) + '_metadata.csv'
        meta  = pd.read_csv(fmeta, header=0, names = ['names','info']).set_index('names')
        
        meta_dict = meta.to_dict()['info']
        meta_dict['lon'] = float(meta_dict['lon'])
        meta_dict['lat'] = float(meta_dict['lat'])        
        df._metadata = meta_dict
        data.append(df)
    
    return table,data

#################
def write_csv(obs_dir, name, year, table, data, label):
    """
    examples
    print('  > write csv files')
    write_csv(obs_dir, name, year, table=wnd_ocn_table, data= wnd_ocn , label='ndbc_wind' )
    write_csv(obs_dir, name, year, table=wav_ocn_table, data= wav_ocn , label='ndbc_wave' )
    write_csv(obs_dir, name, year, table=ssh_table    , data= ssh     , label='coops_ssh' )
    write_csv(obs_dir, name, year, table=wnd_obs_table, data= wnd_obs , label='coops_wind')
    
    """
    #label   = 'coops_ssh'
    #table   = ssh_table
    #data    = ssh

    outt    = os.path.join(obs_dir, name+year,label)
    outd    = os.path.join(outt,'data')  
    if not os.path.exists(outd):
        os.makedirs(outd)

    table.to_csv(os.path.join(outt,'table.csv'))
    stations = table['station_code']

    for ista in range(len(stations)):
        sta   = str(stations [ista])
        fname = os.path.join(outd,sta+'.csv')
        df = data[ista]
        try:
            #in case it is still a series like ssh
            df = df.to_frame()
        except:
            pass
                
        df.to_csv(fname)
        
        fmeta    = os.path.join(outd,sta)+'_metadata.csv'
        metadata = pd.DataFrame.from_dict( data[ista]._metadata , orient="index")
        metadata.to_csv(fmeta)
     
def read_csv(obs_dir, name, year, label):
    """
    examples
    print('  > write csv files')
    write_csv(base_dir, name, year, table=wnd_ocn_table, data= wnd_ocn , label='ndbc_wind' )
    write_csv(base_dir, name, year, table=wav_ocn_table, data= wav_ocn , label='ndbc_wave' )
    write_csv(base_dir, name, year, table=ssh_table    , data= ssh     , label='coops_ssh' )
    write_csv(base_dir, name, year, table=wnd_obs_table, data= wnd_obs , label='coops_wind')
    
    """
    outt    = os.path.join(obs_dir, name+year,label)
    outd    = os.path.join(outt,'data')  
    if not os.path.exists(outd):
       sys.exit('ERROR: check path to: ',outd )

    table = pd.read_csv(os.path.join(outt,'table.csv')).set_index('station_name')
    table['station_code'] = table['station_code'].astype('str')
    stations = table['station_code']

    data     = []
    metadata = []
    for ista in range(len(stations)):
        sta   = stations [ista]
        fname8 = os.path.join(outd,sta)+'.csv'
        df = pd.read_csv(fname8,parse_dates = ['date_time']).set_index('date_time')
        
        fmeta = os.path.join(outd,sta) + '_metadata.csv'
        meta  = pd.read_csv(fmeta, header=0, names = ['names','info']).set_index('names')
        
        meta_dict = meta.to_dict()['info']
        meta_dict['lon'] = float(meta_dict['lon'])
        meta_dict['lat'] = float(meta_dict['lat'])        
        df._metadata = meta_dict
        data.append(df)
    
    return table,data
    
    
    
def write_high_water_marks(obs_dir, name, year):
    url = 'https://stn.wim.usgs.gov/STNServices/HWMs/FilteredHWMs.json'
    params = {'EventType': 2,  # 2 for hurricane
              'EventStatus': 0}  # 0 for completed
    default_filter = {"riverine": True,
                      "non_still_water": True}

    nameyear = (name+year).lower()
    
    out_dir = os.path.join(obs_dir,'hwm')
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    fname = os.path.join(out_dir,nameyear+'.csv')
    usgs_json_file = os.path.join(out_dir,'usgs_hwm_tmp.json')

    if not os.path.exists( usgs_json_file):
        response = requests.get(url, params=params, headers=headers,verify=False)
        response.raise_for_status()
        json_data = json.loads(response.text)
        with open(usgs_json_file, 'w') as outfile:
            json.dump(json_data, outfile )
    else:
        with open(usgs_json_file) as json_file:
            json_data = json.load(json_file)

    hwm_stations = dict()
    for data in json_data:
        if 'elev_ft' in data.keys() and name.lower() in data['eventName'].lower():
            hwm_stations[str(data['hwm_id'])] = data

    log = pd.DataFrame.from_dict(hwm_stations)

    hwm = []   
    ii = 0
    for key in log.keys():
        l0 = []
        for key0 in log[key].keys() :
            l0.append(log[key][key0])
        hwm.append(l0)
    #
    hwm = np.array(hwm)     
    df = pd.DataFrame(data=hwm, columns=log[key].keys()) 


    drop_poor = False
    if drop_poor:
        for i in range(len(df)):
            tt = df.hwmQualityName[i]
            if 'poor' in tt.lower():
                df.hwmQualityName[i] = np.nan    

        df  = df.dropna()    
    df['elev_m'] = pd.to_numeric(df['elev_ft']) *  0.3048  #in meter
    #
    df.to_csv(fname) 


def get_all_data()
    ###############################################
    ###############################################
    ############ MAIN code Starts here ############

    if False:
        # not needed. will take from the storm specific obs list from coops and ndbc
        obs_station_list_gen()
    #
    #######
    # out dir
    obs_dir = os.path.join(base_dirf,'obs')

       
    if get_usgs_hwm:
        for key in storms.keys():
            name = storms[key]['name']
            year = storms[key]['year']
            print('  > Get USGS HWM for ', name)
            try:
                write_high_water_marks(obs_dir, name, year)  
            except:
                print (' > Get USGS HWM for ', name , '   ERROR ...')


    for key in storms.keys():
        name = storms[key]['name']
        year = storms[key]['year']

        print('\n\n\n\n\n\n********************************************************')
        print(            '*****  Storm name ',name, '      Year ',  year, '    *********')
        print(            '******************************************************** \n\n\n\n\n\n')
        
        #if bbox_from_best_track:
        try:    
            
            #bbox_from_best_track = False
            code,hurricane_gis_files = get_nhc_storm_info (year,name)        
            
            ###############################################################################
            #download gis zip files
            base = download_nhc_gis_files(hurricane_gis_files)
            # get advisory cones and track points
            cones,pts_actual,points_actual = read_advisory_cones_info(hurricane_gis_files,base,year,code)
            start    = pts_actual[0] ['FLDATELBL']
            end      = pts_actual[-1]['FLDATELBL']
            #start_txt_actual = ('20' + start[:-2]).replace('/','')
            #end_txt_actual   = ('20' + end  [:-2]).replace('/','')


            #print('\n\n\n\n\n\n ********************************************************')
            #for key1 in pts_actual[0].keys():
            #    print(            '*****  pts_actual[0] [', key1, ']',pts_actual[0] [key1]   ,  '*********')
            #print(            '******************************************************** \n\n\n\n\n\n')

            start_dt = dateparser.parse(start,settings={"TO_TIMEZONE": "UTC"}).replace(tzinfo=None) - obs_xtra_days
            end_dt   = dateparser.parse(end  ,settings={"TO_TIMEZONE": "UTC"}).replace(tzinfo=None) + obs_xtra_days   
            
            #try:
            #    # bbox_from_best_track:
            #    start_txt = start_txt_best
            #    end_txt   = end_txt_best
            #    #bbox      = bbox_best
            #except:
            #    start_txt = start_txt_actual
            #    end_txt   = end_txt_actual

            #
            #start_dt = arrow.get(start_txt, 'YYYYMMDDhh').datetime - obs_xtra_days
            #end_dt   = arrow.get(end_txt  , 'YYYYMMDDhh').datetime + obs_xtra_days    

            
            #if False:
            # get bbox from actual data
            last_cone = cones[-1]['geometry'].iloc[0]
            track = LineString([point['geometry'] for point in pts_actual])
            lons_actual = track.coords.xy[0]
            lats_actual = track.coords.xy[1]
            bbox_actual = min(lons_actual)-2, min(lats_actual)-2, max(lons_actual)+2, max(lats_actual)+2
            ################################################################################

            # Find the bounding box to search the data.
            bbox_from_best_track = False
            bbox      = bbox_actual
        except:
            start_dt   = storms[key]['start']
            end_dt     = storms[key]['end'  ]
            bounds  = storms[key]['bbox' ]


        if storms[key]['bbox'] is not None:
            bbox = storms[key]['bbox']
        
        #print('\n\n\n\n  >>>>> Download and read all GIS data for Storm >',name, '      Year > ', year, '\n     **  This is an old STORM !!!!!! \n\n\n\n')

        #
        # Note that the bounding box is derived from the track and the latest prediction cone.
        strbbox = ', '.join(format(v, '.2f') for v in bbox)


        #
        # Note that the bounding box is derived from the track and the latest prediction cone.
        strbbox = ', '.join(format(v, '.2f') for v in bbox)
        print('\n\n\n\n\n\n********************************************************')
        print(            '*****  Storm name ',name, '      Year ',  year, '    *********')
        print('bbox: {}\nstart: {}\n  end: {}'.format(strbbox, start_dt, end_dt))
        print(            '******************************************************** \n\n\n\n\n\n')
        #
        #########
        
        if get_cops_wlev:
            try:
                print('  > Get water level information CO-OPS ... ')
                
                # ["MLLW","MSL","MHW","STND","IGLD", "NAVD"]
                datum =  'NAVD'
                datum =  'MSL'
                print ('datum=', datum )
                ssh, ssh_table = get_coops(
                    start=start_dt,
                    end=end_dt,
                    sos_name='water_surface_height_above_reference_datum',
                    units=cf_units.Unit('meters'),
                    datum = datum ,
                    bbox=bbox,
                    )

                write_csv(obs_dir, name, year, table=ssh_table    , data= ssh     , label='coops_ssh' )

            except:
                print('  > Get water level information  CO-OPS  >>>> ERRORRRRR')
        ######
        
        

        if get_cops_wind:
            try:
                print('  > Get wind information CO-OPS ... ')
                wnd_obs, wnd_obs_table = get_coops(
                    start=start_dt,
                    end=end_dt,
                    sos_name='wind_speed',
                    units=cf_units.Unit('m/s'),
                    bbox=bbox,
                    )

                write_csv(obs_dir, name, year, table=wnd_obs_table, data= wnd_obs , label='coops_wind')
            except:
                print('  > Get wind information CO-OPS >>> ERORRRR')
        ######
        if get_ndbc_wind:
            try:
                print('  > Get wind ocean information (ndbc) ... ')
                wnd_ocn, wnd_ocn_table = get_ndbc(
                    start=start_dt,
                    end=end_dt,
                    sos_name='winds',
                    bbox=bbox,
                    )

                write_csv(obs_dir, name, year, table=wnd_ocn_table, data= wnd_ocn , label='ndbc_wind' )
            except:
                print('  > Get wind ocean information (ndbc)  >>> ERRRORRRR')
        ######
        if get_ndbc_wave:
            try:
                print('  > Get wave ocean information (ndbc) ... ')
                wav_ocn, wav_ocn_table = get_ndbc(
                    start=start_dt,
                    end=end_dt,
                    sos_name='waves',
                    bbox=bbox,
                    )

                write_csv(obs_dir, name, year, table=wav_ocn_table, data= wav_ocn , label='ndbc_wave' )
            except:  
                print('  > Get wave ocean information (ndbc)  >>> ERRORRRR ')
        ######


    if False:
        # test reading files
        ssh_table1    , ssh1      = read_csv (obs_dir, name, year, label='coops_ssh' )
        wnd_obs_table1, wnd_obs1  = read_csv (obs_dir, name, year, label='coops_wind')
        wnd_ocn_table1, wnd_ocn1  = read_csv (obs_dir, name, year, label='ndbc_wind' )
        wav_ocn_table1, wav_ocn1 = read_csv (obs_dir, name, year, label='ndbc_wave' )

        #if False:
        #
        # back up script file
        args=sys.argv
        scr_name = args[0]
        scr_dir = os.path.join(obs_dir, name+year)
        os.system('cp -fr ' + scr_name + '    ' + scr_dir)
        #
        #with open(pick, "rb") as f:
        #    w = pickle.load(f)

        #f = open(pick, "rb")
        #w = pickle.load(f)


        #if __name__ == "__main__":
        #    main()


if __name__ == "__main__":
    get_all_data()



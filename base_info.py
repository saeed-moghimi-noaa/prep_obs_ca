from __future__ import division,print_function


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Functions for handling observations from ndbc and coops


"""

__author__ = "Saeed Moghimi"
__copyright__ = "Copyright 2018, UCAR/NOAA"
__license__ = "GPL"
__version__ = "1.0"
__email__ = "moghimis@gmail.com"



import pandas    as pd
import numpy as np
import sys,os
import datetime
from   collections import defaultdict

base_dirf = '/disks/NASARCHIVE/saeed_moghimi/post/wrk_dir/'

storms = defaultdict(dict)


if True:    
    key  = 'FLORENCE'
    storms[key]['name' ]   = key
    storms[key]['year' ]   = '2018'
    storms[key]['start']   = datetime.datetime(2018, 8, 26)
    storms[key]['end'  ]   = datetime.datetime(2018, 9, 26)
    storms[key]['bbox' ]   = [-84.40, 9.90, -16.40, 38.20]


get_cops_wlev = True
get_cops_wind = True
get_ndbc_wave = True
get_ndbc_wind = True
get_usgs_hwm  = True

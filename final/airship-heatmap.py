# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:42:16 2020

@author: aerelr
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os



# fig, ax = plt.subplots(figsize=a4_dims)


def main():
    
    # file = r'E:/winds/csv/pct_500_700__2.572_.csv'
    filepath = 'E:/winds/csv/'
    directory = os.fsencode(filepath)
    
    
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith('.csv') and filename.startswith('pct_'):
            generate_heatmap(filepath,filename)
            continue
        else:
            continue
    
    return



def generate_heatmap(directory, filename):
    
    filespec = os.path.join(directory,filename)

    df = pd.read_csv(filespec, sep=',', names=['latitude' , 'month',
                                               'pct_diversity',  'tot_count',
                                               'tot_diversity'])
    print(df.head())
    
    df_heat = df.loc[:,{'latitude', 'month', 'pct_diversity'}]
    print(df_heat.head())
    
    
    df_heat_pt = pd.pivot_table(df_heat, values='pct_diversity', 
                                index=['latitude'], columns='month')
    
    # df_heat_pt.columns = df_heat_pt.columns.map(str)
    
    airspeed_ms, ktas, flt_floor, flt_ceil = parse_filename(filename)
    chart_title = '{:4.1f}'.format(ktas) + ' KTAS (' + str(airspeed_ms) + ' m/s), FL' \
        + str(flt_floor) + '-' + str(flt_ceil)
    
    plt.figure(figsize=(16,9))
    heat = sns.heatmap(df_heat_pt, annot = True, cmap = 'BuGn' )
    heat.set(title=chart_title)
    heat.set_yticklabels(heat.get_yticklabels(),rotation=0)
    heat.invert_yaxis()
        
    return


def parse_filename(filename):
    airspeed_ms = float(filename[-8:-4].replace('_',' '))
    ktas = airspeed_ms * 1.9438
    
    flt_floor = int(filename[3:7].replace('_',' '))
    flt_ceil = int(filename[7:11].replace('_',' '))
    
    return airspeed_ms, ktas, flt_floor, flt_ceil

if __name__ == '__main__':
    main()
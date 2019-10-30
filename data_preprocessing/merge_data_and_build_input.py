# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os, sys
from os import path
import json
from time import clock
from datetime import datetime
import csv
import time
from timeit import timeit


def ReadTxt(txt_path):
    fr = open(txt_path, "r")
    all_txt = fr.read().split("\n")
    while all_txt[-1] == "":
        all_txt = all_txt[:-1]
    fr.close()
    return all_txt

def mkDir(name):
    filepath = os.path.join(os.getcwd(), name)
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    return filepath

def timer(func):
    def _timer(*args, **kwargs):
        start_time = clock()
        result = func(*args, **kwargs)
        end_time = clock()
        print("The total time of %s (function) is %.4f seconds." % (func.__name__, end_time - start_time))
        return result
    return _timer


def get_data(name, **kwargs):
    code_dir = os.getcwd()
    data_path = path.abspath(code_dir) + '/' + read_path_1
    return pd.read_csv(data_path + name, sep=',', **kwargs)


def export_data(df, name, **kwargs):
    code_dir = os.getcwd()
    data_path = path.abspath(code_dir) + '/' + save_path
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    return df.to_csv(data_path + name, index=False, **kwargs)

def calculate_dist(long_1, lat_1, long_2, lat_2):
    east_west = (long_1 - long_2) * 111 / 1.3
    north_south = (lat_1 - lat_2) * 111
    return np.sqrt(east_west ** 2 + (north_south ** 2))


def calculate_dist_and_angle(long_1, lat_1, long_2, lat_2):
    x = (long_2 - long_1) * 111 / 1.3
    y = (lat_2 - lat_1) * 111
    distance = np.sqrt(x ** 2 + y ** 2)
    angle = np.arctan2(y, x)
    if angle < 0:
        angle += np.pi * 2
    return distance, angle

def calculate_nearest(df, meo_name, dist, meo_df, meo_id):
    if type(meo_name) == str:
        meo_name = (meo_name,)
    if type(dist) == str:
        dist = (dist,)

    n = len(meo_name)
    
    for k in range(n):
        df[meo_name[k]] = np.nan
        df[dist[k]] = np.nan
    for i in range(len(df)):
        meo = [(np.inf, np.inf)] * n
        for j in range(len(meo_df)):
            cur_dist = calculate_dist(df['longitude'].iloc[i], df['latitude'].iloc[i],
                                      meo_df['longitude'].iloc[j], meo_df['latitude'].iloc[j])
            if cur_dist < meo[-1][1]:
                meo[-1] = (meo_df[meo_id].iloc[j], cur_dist)
                meo.sort(key=lambda x: x[1])
        # fill up nan
        for k in range(n):
            df.loc[df.index == i, meo_name[k]] = meo[k][0]
            df.loc[df.index == i, dist[k]] = meo[k][1]


def fill_nan(df, road_id, *args):
    station = np.array(df[road_id])
    for aqi in args:
        arr = np.array(df[aqi])
        i, j = 0, 0
        while i < len(arr):
            while i < len(arr) and not np.isnan(arr[i]):
                i += 1
            j = i
            while j < len(arr) and np.isnan(arr[j]) and station[i] == station[j]:
                j += 1
            if i <= 0 or j >= len(arr) or station[i] != station[i - 1] or station[j] != station[j - 1]:
                i = j
            else:
                n = j - i + 1
                for k in range(i, j):
                    
                    arr[k] = (k - (i - 1)) / n * arr[j] + (j - k) / n * arr[i - 1]
        df[aqi] = arr
        df[aqi].fillna(df[aqi].mean(), inplace=True)            

def datelist(beginDate, endDate):
    date_l = [datetime.strftime(x, '%Y-%m-%d') for x in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l

def weeklist(beginDate, endDate):
    date_l = [datetime.strftime(i, "%w") for i in list(pd.date_range(start=beginDate, end=endDate))]
    return date_l

def form_rnn(df, train, test, file_name, sites, forecast, parameters, prev_param):
    dictX = {}
    dictXp = {}
    dictY = {}
    dictYp = {}
    count = 0
    
    for site in sites:
        %time
        site_df = df[df['road_id'] == site].copy(deep=True)
        site_df.sort_values(by='time', inplace=True)

        one_day = np.timedelta64(1, 'D')
        start = np.datetime64(site_df['time'].min(), 'D')
        # Prepare for T - 7
        start = start + one_day * 6
        end = np.datetime64(site_df['time'].max(), 'D')
        end = (end + one_day) - one_day * ((train + test) // 24) + one_day

        rnn = pd.DataFrame({'date': np.arange(start, end, dtype='datetime64[D]')})
        rnn['id'] = site

        # Fill n hours data
        def fill_n(n, day_df, hour_df, values, mark, delta):
            for value in values:
                a = np.array(hour_df[value])
                print(count, file_name, site, mark, value, len(day_df), len(hour_df))
                for i in range(n):
                    day_df['%s#%d' % (value, i)] = \
                        a[(i + delta): (i + delta) + 24 * len(day_df): 24]
            return day_df
        count += 1
        
        X = rnn.copy(deep=True)
        X = fill_n(train, X, site_df, forecast + parameters, "X", delta=24*6)
        
        X_prev = rnn.copy(deep=True)
        X_prev = fill_n(train, X_prev, site_df, prev_param, "X_prev", delta=0)
        
        Y = rnn.copy(deep=True)
        Y = fill_n(test, Y, site_df, forecast, "Y", delta=24*6+train)
        
        Yp = rnn.copy(deep=True)
        Yp = fill_n(test, Yp, site_df, parameters, "Y_wrf", delta=24*6+train)
        
        dictX[site] = X
        dictY[site] = Y
        dictXp[site] = X_prev
        dictYp[site] = Yp

    def add_prefix(prefix, labels):
        return dict((label, prefix + label) for label in labels)
    
    def all_site_merge(dicts, prefixs):
        rnn = None
        j = 0
        for site in sites:
            j += 1
            for i in range(len(dicts)):
                d = dicts[i]
                prefix = prefixs[i]
                X = d[site].copy(deep=True)
                X.drop(['id'], axis=1, inplace=True)
                labels = list(X.columns)
                labels.remove('date')
                X.rename(columns=add_prefix(prefix + site + '_', labels), inplace=True)
                if rnn is None:
                    rnn = pd.DataFrame({'date': X['date']})
                X.drop(['date'], axis=1, inplace=True)
                for label in list(X.columns):
                    rnn[label] = X[label]
                    # print(label)
            print('[INFO] site {} has been processed.'.format(j))
        return rnn

    print('X (meo t-1, traffic t-1), X_prev (traffic t-7) and Y (meo t, traffic t) start to be generated.') 
    
    X = all_site_merge([dictX, dictXp, dictYp], ['', 'prev__', 'wrf__'])
    print('X has been generated, (meo t-1, traffic t-1, traffic t-7, meo t).shape = 14')

    Y = all_site_merge([dictY], ['future__'])
    print('Y has been generated, (traffic t).shape = 1')

    return X, Y

def generate_train_val_test(x, y, output_dir):
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # 7/10 is used for training.
    # 1/10 is used for validation.
    # 2/10 is used for test.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
        )

def main():
    input_step = 24
    timesteps_in = 24
    timesteps_out = 24
    time_tag = '_' + str(timesteps_in) + 'to' + str(timesteps_out)
    road_level = 'all_id_level_sorted'

    read_path_1 = '10_shenzhen_data_gcn/'
    save_path = "11_sz_data_merge_output_1909016/"
    
    # Read road_info (beijing_aqi)
    road_info = get_data('road_info_' + road_level + '.csv')
    print('[DATA-1] road info has been extracted!')

    # read meo grid info (beijing_grid)
    meo_grid_info = get_data("wrf_extracted_grid_xy.csv")
    print('[DATA-2] meo grid info has been extracted!')

    # Read history road and meo grid data
    # road_hist (beijing_hist_aqi_total)
    road_hist = get_data("result_sz_road_170407-190115_sorted.csv")
    print('[DATA-3] traffic data has been extracted!')

    # meo_grid_hist (beijing_hist_grid)
    meo_grid_hist = get_data("wrf_extraction_clipframe_20170407-20190115.csv")
    print('[DATA-4] meo data has been extracted!')

    road_info[(road_info.dtypes == np.float64) & (road_info < 0)] = np.nan
    road_hist[(road_hist.dtypes == np.float64) & (road_hist < 0)] = np.nan
    print('[Pre-1] null values have been replaced by np.nan!')

    ### 1st step, find nearest meo grid
    calculate_nearest(road_info, 'nearest_grid', 'dist_0',
                      meo_grid_info, 'meo_grid_id')
    print('[STEP-1] find nearest meo grid.')

    # Fill the missing data in meo
    road_hist['time'] = pd.to_datetime(road_hist['time'])
    road_hist.drop_duplicates(subset=['road_id', 'time'], inplace=True)
    meo_grid_hist['time'] = pd.to_datetime(meo_grid_hist['time'])
    meo_grid_hist.drop_duplicates(subset=['meo_grid_id', 'time'], inplace=True)
    print('[INFO] Fill the missing data in road and meo.')

    # Fill the missing time series
    def fill_time(df, sites, road_id, time):
        left = pd.DataFrame({road_id: sites})
        arr = np.array(df[time])
        right = pd.DataFrame({time: pd.to_datetime(
            np.arange(arr.min(), arr.max() + np.timedelta64(1, 'h'), dtype='datetime64[h]')
        )})
        left['on'] = 0
        right['on'] = 0
        framework = pd.merge(left, right, how="outer", on='on')
        framework.drop('on', axis=1, inplace=True)
        return pd.merge(framework, df, how='left', on=[road_id, time])
    road_hist = fill_time(road_hist, np.array(road_info['road_id']), 'road_id', 'time')
    print('[INFO] fill the missing time series.')

    # Merge the meo data into aqi data

    ### 2nd step, merge nearest grid to road
    road_hist_info_neargrid = pd.merge(road_hist, road_info, on='road_id')
    print('[STEP-2] merge nearest grid to road.')

    ### 3rd step, merge nearest meo info accordding to nearest grid
    road_meo_hist = pd.merge(road_hist_info_neargrid, meo_grid_hist.drop(['latitude', 'longitude'], axis=1), how='left',
                            left_on=['time', 'nearest_grid'],
                            right_on=['time', 'meo_grid_id'])
    print('[STEP-3] merge nearest meo info accordding to nearest grid.')

    # Sort the data
    road_meo_hist.sort_values(by=['road_id', 'time'], inplace=True)

    # Fill the missing data in all air pollutants
    fill_nan(road_meo_hist, 'road_id', 'distance', 'travel_time', 'road_speed')

    print('[INFO] sorted and fill nan.')

    # Calculate distance(eta), angle(theta, c1->c2 for theta(c1,c2))
    assert len(road_info['road_id']) == road_info.shape[0]

    # Prepare to form rnn input
    road_meo_hist_selected = road_meo_hist[['road_id', 'time',
                                'road_speed', 'temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed',
                                 'hourly_precipitation'
                                ]].copy(deep=True)

    print('[END] the data frame for forming rnn has been built.')

    ### 1st step, find nearest meo grid
    ### 2nd step, merge nearest station to aqi table
    ### 3rd step, merge nearest meo info accordding to nearest station
    
    
    ## generate data for DCRNN
    FORECAST = ['road_speed']
    PARAMETERS = ['temperature', 'pressure', 'humidity', 'wind_direction', 'wind_speed', 'hourly_precipitation']

    X, Y = form_rnn(road_meo_hist_selected, train=timesteps_in, test=timesteps_out, file_name='all', 
                         sites=road_info['road_id'],
                         forecast=FORECAST,
                         parameters=PARAMETERS,
                         prev_param=FORECAST)

    print('[INFO] starting to generate rnn-form data.')

    ## output intermediate data
    X.to_csv('{folder}/{name}.csv'.format(folder=save_path, name='X14'), index=False, float_format='%.6f')
    Y.to_csv('{folder}/{name}.csv'.format(folder=save_path, name='Y1'), index=False, float_format='%.6f')
    
    INPUT_NUM = 642  ##days or samples
    OUTPUT_NUM = 642
    TIMESTEPS_IN_SLICE = 24
    TIMESTEPS_OUT_SLICE = 24
    SITES = 1378

    X_DIM = len(PARAMETERS) *2 + 2
    Y_DIM = 1
    
    df_X = X.copy(deep=True)
    df_Y = Y.copy(deep=True)

    df_X = df_X.drop(['date'], axis=1).values
    df_X = df_X.reshape((INPUT_NUM, SITES, X_DIM, TIMESTEPS_IN_SLICE), order='C')
    # Current shape: num_samples, num_nodes, input_dim, input_length
    df_X = df_X.swapaxes(1, 3)
    # Current shape: num_samples, input_length, input_dim, num_nodes
    df_X = df_X.swapaxes(2, 3)
    # Current shape: num_sample, input_length, num_nodes, input_dim
    print(df_X.shape)
    
    ## concat zero matrix to match the shape of Y to the shape of X 
    df_0 = np.zeros(shape=(X.shape[0], X.shape[1] - Y.shape[1]))
    print(df_0.shape)

    df_Y = df_Y.drop(['date'], axis=1).values
    df_Y = np.concatenate((df_Y, df_0), axis=1)
    print(df_Y.shape)

    df_Y = df_Y.reshape((OUTPUT_NUM, X_DIM, SITES, TIMESTEPS_OUT_SLICE), order='C')
    # df_Y = df_Y.reshape((OUTPUT_NUM, SITES, X_DIM, TIMESTEPS_OUT_SLICE), order='C')
    # Current shape: date, param, site, seq(slice)
    df_Y = df_Y.swapaxes(1, 3)
    # Current shape: num_samples, input_length, num_nodes, input_dim

    PATH = os.path.abspath('sz_data_merge_output_1909016')
    generate_train_val_test(df_X, df_Y, PATH)
    
    date = X[['date']]
    date_num = len(date_X)

    num_test = round(date_num * 0.2)
    num_train = round(date_num * 0.7)
    num_val = date_num - num_test - num_train

    date_train = date[:num_train]
    date_val = date[num_train: num_train + num_val]
    date_test = date[-num_test:]
    print(date_test)

    ## output train, val, test dataset
    date_train.to_csv('{folder}/{filename}.csv'.format(folder=PATH, filename='date_train'), index=False)
    date_val.to_csv('{folder}/{filename}.csv'.format(folder=PATH, filename='date_val'), index=False)
    date_test.to_csv('{folder}/{filename}.csv'.format(folder=PATH, filename='date_test'), index=False)

    print('[INFO] rnn-form data has been built.')

    print("[INFO] Merging End!")
    
    
if __name__ == "__main__":
    print("[INFO] Merging Begin")
    main()
    
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import datetime
from wrf import getvar, to_np, ALL_TIMES
import csv


def get_wrf(path, row_slice_begin, row_slice_end, col_slice_begin, col_slice_end):
    # Option 2, provided by Dongsheng
    dataset = Dataset(path, 'r', format = 'NETCDF4')
    times_ori = np.array(dataset.variables['Times'])
    times_unicode = np.char.decode(times_ori)
    times_unicode = list(map(lambda x: ''.join(x), times_unicode))
    times = np.array(list(map(lambda x: pd.to_datetime(x, format='%Y-%m-%d_%H:%M:%S'), times_unicode)))

    # extract lon and lat
    lon = np.array(dataset.variables['XLONG'])
    lat = np.array(dataset.variables['XLAT'])
    lon_long = np.reshape(lon[0, row_slice_begin:row_slice_end, col_slice_begin:col_slice_end], (-1, 1), order='F')
    lat_long = np.reshape(lat[0, row_slice_begin:row_slice_end, col_slice_begin:col_slice_end], (-1, 1), order='F')
    times_repeat = np.reshape(np.repeat(times, len(lon_long)), (-1,1), order = 'F')


    # revise lat and lon, loop 24 times
    lon_long_repeat = np.reshape(np.tile(lon_long, 24), (-1, 1), order='F')
    lat_long_repeat = np.reshape(np.tile(lat_long, 24), (-1, 1), order='F')

    # extract x-wind component at 10 M
    x_wind_10_ori = np.array(dataset.variables['U10'])
    # x_wind_10_lst = x_wind_10_ori.tolist()
    x_wind_10_2dim = np.swapaxes(
                        list(map(lambda x: np.reshape(x, (-1,), order='F'),
                                 x_wind_10_ori[:, row_slice_begin:row_slice_end, col_slice_begin:col_slice_end])), 1, 0)
    x_wind_10_1dim = np.reshape(x_wind_10_2dim, (-1, 1), order='F')

    # extract y-wind component at 10 M
    y_wind_10_ori = np.array(dataset.variables['V10'])
    y_wind_10_2dim = np.swapaxes(
                        list(map(lambda x: np.reshape(x, (-1,), order='F'),
                                 y_wind_10_ori[:, row_slice_begin:row_slice_end, col_slice_begin:col_slice_end])), 1, 0)
    y_wind_10_1dim = np.reshape(y_wind_10_2dim, (-1, 1), order='F')


    # extract TEMP at 2 M
    temp_ori = np.array(dataset.variables['T2'])
    temp_2dim = np.swapaxes(
                        list(map(lambda x: np.reshape(x, (-1,), order='F'),
                                 temp_ori[:, row_slice_begin:row_slice_end, col_slice_begin:col_slice_end])), 1, 0)
    temp_1dim = np.reshape(temp_2dim, (-1, 1), order='F')
    # print(dataset.variables['T2'])


    # extract Surface PRESSURE
    press_ori = np.array(dataset.variables['PSFC'])
    press_2dim = np.swapaxes(
                        list(map(lambda x: np.reshape(x, (-1,), order='F'),
                                 press_ori[:, row_slice_begin:row_slice_end, col_slice_begin:col_slice_end])), 1, 0)
    press_1dim = np.reshape(press_2dim, (-1, 1), order='F')
    # print(dataset.variables['PSFC'])

    # extract humidty
    # https://wrf-python.readthedocs.io/en/latest/user_api/generated/wrf.rh.html?highlight=rh
    rh_ori = to_np(getvar(dataset, 'rh', timeidx=ALL_TIMES))
    rh_surf = rh_ori[:, 0, :, :]
    rh_surf_2dim = np.swapaxes(
        list(map(lambda x: np.reshape(x, (-1,), order='F'),
                 rh_surf[:, row_slice_begin:row_slice_end, col_slice_begin:col_slice_end])), 1, 0)
    humi_1dim = np.reshape(rh_surf_2dim, (-1, 1), order='F')

    # extract wind direction and speed
    # https://wrf-python.readthedocs.io/en/latest/internal_api/generated/wrf.g_uvmet.get_uvmet10_wspd_wdir.html?highlight=wspd_wdir10
    wspd_wdir10 = to_np(getvar(dataset, 'wspd_wdir', timeidx=ALL_TIMES))
    wspd10 = wspd_wdir10[0, :, 0, :, :]
    wdir10 = wspd_wdir10[1, :, 0, :, :]

    wspd10_2dim = np.swapaxes(
        list(map(lambda x: np.reshape(x, (-1,), order='F'),
                 wspd10[:, row_slice_begin:row_slice_end, col_slice_begin:col_slice_end])), 1, 0)
    wspd10_1dim = np.reshape(wspd10_2dim, (-1, 1), order='F')

    wdir10_2dim = np.swapaxes(
        list(map(lambda x: np.reshape(x, (-1,), order='F'),
                 wdir10[:, row_slice_begin:row_slice_end, col_slice_begin:col_slice_end])), 1, 0)
    wdir10_1dim = np.reshape(wdir10_2dim, (-1, 1), order='F')

    # generate id for grids
    grid_id = np.reshape(['meo_grid_' + str(i) for i in range(lon_long.shape[0])], (-1, 1))
    grid_id_repeat = np.reshape(np.tile(grid_id, 24), (-1, 1), order='F')

    # concat and output
    all_meo = np.concatenate((grid_id_repeat, lon_long_repeat, lat_long_repeat, times_repeat,
                              x_wind_10_1dim, y_wind_10_1dim, temp_1dim, press_1dim, humi_1dim,
                              wspd10_1dim, wdir10_1dim), axis=1)
    return (all_meo)

def get_daily_rain(path, row_slice_begin, row_slice_end, col_slice_begin, col_slice_end):
    dataset = Dataset(path, 'r', format='NETCDF4')
    # extract ACCUMULAED TOTAL CUMULUS PRECIPITATION
    preci_ori_cum = np.array(dataset.variables['RAINC'])
    # extract ACCUMULATED TOTAL GRID SCALE PRECIPITATION
    preci_ori_grid = np.array(dataset.variables['RAINNC'])
    preci_ori = preci_ori_cum + preci_ori_grid
    preci_ori = preci_ori[:, row_slice_begin:row_slice_end, col_slice_begin:col_slice_end]
    return (preci_ori)

def main():
    # current shape = (24, 20, 40) in Shenzhen

    # col min = 0, col max = 19，
    # col min = 0, col max = 39，
    row_slice_begin = 1
    row_slice_end = 19
    col_slice_begin = 1
    col_slice_end = 39
    read_path = '~/external_hdd2/shenzhen/'
    save_path_1 = 'wrf_extraction_clipframe_20170407-20190115.csv'

    # creat blank .csv and write the title
    with open(save_path_1, 'w') as csvfile:
        clo_name = ['meo_grid_id', 'longitude', 'latitude', 'time', 'wind_x', 'wind_y', 'temperature', 'pressure',
                    'humidity', 'wind_speed', 'wind_direction', 'hourly_precipitation']
        writer = csv.DictWriter(csvfile, fieldnames=clo_name)
        writer.writeheader()
        csvfile.close()

    # settging the begin and end of date
    begin = datetime.date(2017, 4, 7)
    end = datetime.date(2019, 1, 15)

    preci_24th_hr_3dim = np.zeros((1, row_slice_end - row_slice_begin, col_slice_end - col_slice_begin))
    for i in range((end - begin).days + 1):
        day_now = begin + datetime.timedelta(days=i)
        meo_name_now = "{year:0>4}new/wrfout_d01_{year:0>4}-{month:0>2}-{day:0>2}_00:00:00".format(
                year=day_now.year, month=day_now.month, day=day_now.day)
        print(meo_name_now)
        # get precipitation
        if os.path.isfile(read_path + meo_name_now):
            
            preci_ori = get_daily_rain(read_path + meo_name_now, row_slice_begin, row_slice_end, col_slice_begin, col_slice_end)
            # judge if it is the firtst day and 20160512. 20160512 is a breakpoint if you would like to generate data in 2016.
            preci_daily = []
            for t in range(preci_ori.shape[0]):
    
                if (t == 0 and meo_name_now == "{year:0>4}new/wrfout_d01_{year:0>4}-{month:0>2}-{day:0>2}_00:00:00".format(
                        year=begin.year, month=begin.month, day=begin.day)) or (t == 0 and meo_name_now == "2016/wrfout_d01_2016-05-13_00:00:00"):
                    
                    for x in range(col_slice_end - col_slice_begin):
                        for y in range(row_slice_end - row_slice_begin):
                            preci = preci_ori[t, y, x]
                            preci_daily.append(preci)
                # the first hour，initial value is: 1st-hour accumulated preci - last 24th-hour accumulated preci
                elif t == 0 and meo_name_now != "{year:0>4}new/wrfout_d01_{year:0>4}-{month:0>2}-{day:0>2}_00:00:00".format(
                        year=begin.year, month=begin.month, day=begin.day):
                    # for x in range(col_slice_begin, col_slice_end):
                    for x in range(col_slice_end - col_slice_begin):
                        for y in range(row_slice_end - row_slice_begin):
                            # tmp为1st-hour accumulated preci - last 24th-hour accumulated preci
                            tmp = preci_ori[t, y, x] - preci_24th_hr_3dim[0, y, x]
                            # tmp = preci_ori[t, y, x] - preci_24th_hr_3dim[0, y, x - col_slice_begin]
                            if tmp < 0:
                                preci = preci_ori[t, y, x]
                                preci_daily.append(preci)
                            else:
                                preci = tmp
                                preci_daily.append(preci)
                else:
                    
                    for x in range(col_slice_end - col_slice_begin):
                        for y in range(row_slice_end - row_slice_begin):
                            preci_t = preci_ori[t, y, x]
                            preci_t1 = preci_ori[t - 1, y, x]
                            tmp = preci_t - preci_t1
                            if tmp < 0:
                                preci = preci_ori[t, y, x]
                                preci_daily.append(preci)
                            else:
                                preci = tmp
                                preci_daily.append(preci)

            # store the value of last hour in last data and use it as the t-1 value in next day
            preci_24th_hr = preci_ori[23, :, :]
            
            preci_24th_hr_3dim = np.reshape(preci_24th_hr, (1, preci_ori.shape[1], -1), order='F')
            print("got hourly-precipitation from " + meo_name_now + "!")

            # concat and output
            preci_daily = np.reshape(preci_daily, (-1, 1))
            meo_info = get_wrf(read_path + meo_name_now, row_slice_begin, row_slice_end, col_slice_begin, col_slice_end)
            meo_all = np.concatenate((meo_info, preci_daily), axis=1)
            predict_output = pd.DataFrame(data=meo_all)
            
            predict_output = predict_output.apply(pd.to_numeric, errors='ignore')
            # print(predict_output.dtypes)
            
            predict_output.to_csv(save_path_1, mode='a', header=False, float_format='%.6f', index=False)
            print("got data from " + meo_name_now + "!")
        else:
            continue

    print("Finish extracting meo data from netcdf!")

if __name__ == "__main__":
    main()



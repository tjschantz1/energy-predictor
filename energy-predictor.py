# =============================================================================
# DATA-59000, Fall II 2019
# Thomas J Schantz
# Data Science Final Project
# =============================================================================
''' 
Data Source:
https://www.kaggle.com/c/ashrae-energy-prediction/data
'''

#%% Import libraries

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Home-grown
import myplotters as mp # plotting functions
import myfuncs as mf # data manipulation functions
import mymodels as mdl # ML model object


# Stats
import time
import math
import statistics
import pandas as pd
pd.options.mode.chained_assignment = None  #suppress SettingWithCopyWarning
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import numpy as np
import matplotlib.pyplot as plt
import collections
from datetime import datetime
from dateutil.parser import parse


# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error # RMSLE


#%% Initiate
print('\n=== Program Initiated ===')
start_time = time.time()
path = mf.get_path(subdir='ashrae-energy-prediction')
save_time, log_y, save_plots = 'False', 'False', 'False' #converted to bool blw
try: save_time = save_time != input('Save on runtime (True/False)? ')
except: 
    save_time = False
    print('!!! Input of wrong type; running save_time as False !!!')
try: log_y = log_y != input('Log transform y (True/False)? ')
except: 
    log_y = False
    print('!!! Input of wrong type; running log_transoform as False !!!')
try: save_plots = save_plots != input('Export plots to auto_exports (True/False)? ')
except: 
    save_plots = False
    print('!!! Input of wrong type; running save_plots as False !!!')

    
#%% Import data & establish definitions
   
print('\n[>>> Importing data...]')

rand_samp = 0
building = mf.import_data(path=path, file_name='building_metadata.csv',
                               datetimeCol='timestamp', rand_samp=rand_samp)
train = mf.import_data(path=path, file_name='train.csv',
                            datetimeCol='timestamp', rand_samp=rand_samp)
train_weather = mf.import_data(path=path, file_name='weather_train.csv',
                                    datetimeCol='timestamp', rand_samp=rand_samp)

test = mf.import_data(path=path, file_name='test.csv',
                           datetimeCol='timestamp', rand_samp=rand_samp)
test_weather = mf.import_data(path=path, file_name='weather_test.csv',
                               datetimeCol='timestamp', rand_samp=rand_samp)
    
# Handle meter type 0 error found here
# https://www.kaggle.com/.../discussion/119261#latest-687297
def kBTU_to_kWh(dfs):
    for df in dfs: df.loc[df['meter']==0, ['meter_reading']] *= 0.2931
kBTU_to_kWh([train])

# Establish mappers for encoding categorical
target = 'meter_reading'
predictors_categor = ['meter', 'site_id']
lab_rename = 'NonTraditionalUsage'
def map_dict():
    
    # Existing category fields
    map_meter= {0:'electricity', 1:'chilledwater', 2:'steam', 3:'hotwater'}
    map_primary_use = {i : sorted(building.primary_use.unique())[i] \
                       for i in range(0, len(building.primary_use.unique()))}
    map_primary_use.update({99:lab_rename})
    
    # New category fields
    map_season = {'Spring':[3,4,5], 'Summer':[6,7,8],
                  'Fall':[9,10,11], 'Winter':[12,1,2]}
    
    # Time zones
    site_gmt_offsets = [-5, 0, -7, -5, -8, 0, -5, -5, \
                        -5, -6, -7, -5, 0, -6, -5, -5]
    map_gmt_offset = {site: offset for site, offset in \
                       enumerate(site_gmt_offsets)}
    
    # All together
    mappers = {'map_meter':map_meter, 'map_primary_use':map_primary_use,
                'map_season':map_season, 'map_gmt_offset':map_gmt_offset}
    
    return mappers
mappers = map_dict()

# Define functions for quick encoding
def cat_encode(encode_to_int, cat_list, df_list):
    for df in df_list: #[building, train, test]
        for col in set(cat_list).intersection(df): #check if col is in df
            mf.map_field(df=df, map_col=col, 
                            dict_map=mappers['map_'+col],
                            reverse=encode_to_int)
    
# Define function to check if encoding is needed
def change_encoding(df_list, cat_list, encode_to_int):
    for df in df_list:
        for col in cat_list:
            try: # check if col exists in df
                if df[col].dtype!='object' and encode_to_int==False:
                    # encoded, and requested convert to txt
                    cat_encode(encode_to_int=encode_to_int,
                               cat_list=[col],
                               df_list=[df])
                elif df[col].dtype=='object' and encode_to_int==True:
                    # txt, and requested encode
                    cat_encode(encode_to_int=encode_to_int,
                               cat_list=[col],
                               df_list=[df])
            except: next
            
# Unencode categorical fields for data exploration
change_encoding(df_list=[building, train, test],
               cat_list=predictors_categor, encode_to_int=False)

# Identify observations for train/test split
np.random.seed(3)
train_val_dict = {'train':0, 'val':1} # for ref later
train['train_val_split'] = np.random.choice([0,1],
                            size=train.shape[0], p=[.8,.2])

print('\n=== Data Import Complete ===')
print('--- Runtime =', mf.timer(start_time, time.time()),'---')
print('--- Running =', mf.timer(start_time, time.time()),'---')
new_time = time.time()
   
#%% Exploratory Analysis

def explore_data(new_time):
    
    print('\n[>>> Exploring data...]')
    
    # Quick stats
    print('\nNumber of zero-value meter readings: {}'.format( \
          train.meter[train.meter_reading==0].value_counts()))
    print('\nNumber of buildings in data: {}'.format( \
          len(train.building_id.unique())))
    mf.determine_largest_target_variation(df=train, target_col=target,
                                          by_col='building_id', n=5)
    '''
    Conclusion: may want to remove ID:stddev 778: 116709.890625, 1099: 4834778.5
    '''
    
    meter_types = sorted(train.meter.unique().tolist())
    max_by_meter = [train[target].loc[train.meter==x].max() \
                    for x in meter_types]
    print('\nMax by meter type: {}'.format( \
          dict(zip(meter_types, max_by_meter))))
    std_by_meter = [train[target].loc[train.meter==x].std() \
                    for x in meter_types]
    print('\nStd dev by meter type: {}'.format( \
          dict(zip(meter_types, std_by_meter))))
    
    dfs = [building, train, test, train_weather, test_weather]
    
    # Print shape & missing
    def describe_shape_missing(dfs):
        print('\n---Data Shape Analysis---\n')
        for df in dfs:
            print('Size of {} data: {}'.\
                  format([x for x in globals() if globals()[x] is df][0],
                         df.shape))
        print('\n---Missing Data Analysis---\n')
        for df in dfs:
            print('Missing {} data:\n{}\n'.\
                  format([x for x in globals() if globals()[x] is df][0],
                         mf.tbl_missing(df)))
    describe_shape_missing(dfs=dfs)
    
    # Floor Count Assessment
    def quick_print(df):
        df = mf.assess_by_cat(df=building,
                              by_cat_col='primary_use', 
                              agg_col='floor_count',
                              save=save_plots)
        return df
    quick_print(building)
    def predict_naive(dfi, target, by_cat_col, agg_col):
        df = dfi.copy()
        
        # Split data for training
        np.random.seed(3)
        # {'train':0, 'val':1} 
        df['train_val_split'] = np.random.choice([0,1],
                                    size=df.shape[0], p=[.8,.2])
            
        # "Train" naive predictor by creating piv tbl for using in naive pred
        trainer = mf.assess_by_cat(df=df[df.train_val_split==0],
                                   by_cat_col=by_cat_col, 
                                   agg_col=agg_col,
                                   print_tbls=True,
                                   return_df=True,
                                   save=save_plots)
        
        # Test naive predictor
        df.dropna(inplace=True)
        df['prediction'] = df[by_cat_col].map(trainer['mode']) #index-match
        df['eval'] = df.prediction==df[agg_col]
        accuracy = df['eval'].sum() / df['eval'].count()
        return 'Naive Accuracy: {:.1%}'.format(accuracy)
        
    predict_naive(dfi=building, target='floor_count', by_cat_col='primary_use',
                  agg_col='floor_count') #mode naive accuracy = 17.7%
    '''
    !!!Conclusion: no good way found to impute/replace missing floor_count data
    '''
    
    # Analysis zero-value readings
    print('\nZero-value meter readings in train set:')
    vol_zero = train.meter_reading[train.meter_reading==0].count()
    perc_zero = vol_zero / train.shape[0]
    print('  {:,} ({:.1%})'.format(vol_zero, perc_zero))
    print('\nZero-value meter readings by meter type:')
    for val in list(train.meter.unique()):
        vol_zero = train.meter_reading[(train.meter_reading==0) & \
                                       (train.meter==val)].count()
        perc_zero = vol_zero / train.meter_reading[train.meter==val].count()
        print('  {}: {:,} ({:.1%})'.format(val, vol_zero, perc_zero))
    
    # Time series plot of weather features
    mp.matrix_series(df=train_weather, by_cols=list(train_weather.iloc[:,2:]),
                     time_col='timestamp', save=save_plots)
    for site in list(train.site_id.unique()):
        mp.matrix_series(df=train_weather,
                         by_cols=list(train_weather.iloc[:,2:]),
                         time_col='timestamp', filter_col='site_id',
                         filter_val=site, save=save_plots)
    
    for col in list(train_weather.iloc[:,2:]):
        mp.matrix_series(df=train_weather, by_cols=[col],
                         time_col='timestamp', by_vals=True,
                         filter_col='site_id', save=save_plots)
    
    # Quick plot of target values
    plt.figure(figsize = (15,5))
    train[target].plot()
    
    # Histogram of Meter Reading in log scale
    mp.myHist(dfi=train, xColN=target, ibins=20,
              pTitle='Meter Reading Histogram (Log-Scaled)', 
              xColLab='Meter Reading', 
              log_scale=True, save=save_plots)
    
    # Bar chart of 10^x interval counts for meter reading
    interval_df = mf.count_intervals(train[target],
                                     [10**x for x in np.arange(1,10)])
    interval_df = collections.OrderedDict(sorted(interval_df.items()))
    interval_df = pd.DataFrame(list(interval_df.items()),
                               columns=['interval', 'volume'])
    interval_df.interval = ['{:,}'.format(x) for x in interval_df.interval]
    mp.myBar(dfi=interval_df, 
             xColN='interval', xColLab='Interval',
             yCol1N='volume', yColLab='Count',
             pTitle='Count of Meter Reading Values by Intervals 10^x',
             data_labels=True, save=save_plots)
    
    # Field value comparison
    def compare_field_values(df1, df2, by_col):
        if df1[by_col].dtype == '<M8[ns]' or \
        df1[by_col].dtype == 'datetime64[ns]': # datetime
            comp1 = list(df1[by_col].dt.date.unique())
            comp2 = list(df2[by_col].dt.date.unique())
            print('\nSame dates betweem {} & {} == {}'.\
                  format([x for x in globals() if globals()[x] is df1][0],
                         [x for x in globals() if globals()[x] is df2][0],
                         comp1==comp2))
            if comp1!=comp2: 
                print('   {} missing dates'.format(len(set(comp1)-set(comp2))))
    compare_field_values(train, test, 'timestamp')
    compare_field_values(train_weather, test_weather, 'timestamp')
    
    print('\n=== Exploratory Analysis Complete ===')
    print('--- Runtime =', mf.timer(new_time, time.time()),'---')
    print('--- Running =', mf.timer(start_time, time.time()),'---')
    new_time = time.time()
 
#explore_data(new_time) # comment-out  when not needed

#%% Data Cleaning
def clean_data(new_time, building, train, test, train_weather, test_weather):
    '''
    impute/interpolate missing values, change nominal attributes into a 
    set of dummy indicators, detect and remove abnormal data
    '''
    print('\n[>>> Cleaning data...]')
    
    dfs = [building, train, test, train_weather, test_weather]
            
    pre_clean_length = mf.catalog_length(dfs=dfs)

    # Ensure mirrored data sets remain mirrored after feature removal
    mf.match_df_cols(train_weather, test_weather)
    
    # Offset GMT by site timezone mapper
    '''
    Weather is tracked in GMT, readings are tracked in local standard time,
    and actual building usage will be affected by local daylight savings time.
    '''
    def norm_gmt(df):
        df.timestamp = df.timestamp + pd.to_timedelta(df.site_id.map( \
                                        mappers['map_gmt_offset']), unit='h')
        return df
    train_weather = norm_gmt(train_weather)
    test_weather = norm_gmt(test_weather)
    
    # Handle missing weather data through interpolation
    def interpolate_by_val(df, by_val, sort_cols, interp_cols):
        start_time = time.time()
        new_time = time.time()
        interpolated_df = pd.DataFrame(columns=list(df.columns)) # empty df
        df.sort_values(by=sort_cols, inplace=True) # ensure order
        #df.set_index(sort_cols,inplace=True) # set index
        for val in list(df[by_val].unique()):
            print('\n---Interpolating on {} == {}---'.format(by_val, val))
            dfc = df[df[by_val]==val].copy() # filter by value
            dfc[interp_cols] = dfc[interp_cols].interpolate(method='linear')
            
            # Fill any remaining miss vals failed to interpolate
            for col in interp_cols:
                mf.back_fwd_filler(df=dfc, col=col) # custom filler
                
            # Append to master df
            interpolated_df = interpolated_df.append(dfc, ignore_index=True)
            print('  elapsed time = {}'.format( \
                  mf.timer(new_time, time.time())))
            new_time = time.time()
        print('Total elapsed time = {}'.format( \
              mf.timer(start_time, time.time())))
        return interpolated_df
    by_cols=['site_id', 'timestamp']
    print('\n...interpolating missing data in train_weather...')
    train_weather = interpolate_by_val(df=train_weather, by_val='site_id',
                                       sort_cols=by_cols,
                                       interp_cols= \
                                       list(train_weather.iloc[:,2:]))
    
    test_weather = interpolate_by_val(df=test_weather, by_val='site_id',
                                       sort_cols=by_cols,
                                       interp_cols= \
                                           list(test_weather.iloc[:,2:]))
    print('\n---Missing Values Following Linear Interpolation---')
    print('\n{}\n{}'.format('train_weather', mf.tbl_missing(train_weather)))
        
    '''
    XXX 191207 - Conclusion: there are sites who had entire year's worth of
    data not captures for certain temp features; will remove precip_depth_1_hr
    & sea_level_pressure for this reason
    '''
    
    # Identify & delete cols with nan & remove
    cols_with_nan = train_weather.columns[train_weather.isnull().any()] \
                    .tolist()
    print('\nWeather features to be removed due to lack of yearly data:\n{}'\
          .format(cols_with_nan))  
    train_weather.drop(cols_with_nan, axis=1, inplace=True)
    test_weather.drop(cols_with_nan, axis=1, inplace=True)
    
    # Fill in missing hourly time stamps and interpolate missing vals
    def fill_in_time(df, time_col, stack_col):
        start, stop = df[time_col].min(), df[time_col].max()
        df_byhr = pd.DataFrame(columns=[stack_col, time_col]) # empty df
        for val in list(df[stack_col].unique()): # multiply length by num vals
            df_byhr_byval = pd.DataFrame({stack_col:val, \
                                          time_col:pd.date_range(start, stop,
                                                                 freq='1H',
                                                                 closed=None)})
            df_byhr = df_byhr.append(df_byhr_byval, sort=False) # stack
        by_cols=[stack_col, time_col]
        df = pd.merge(df, df_byhr, how='right', left_on=by_cols,
                      right_on=by_cols)
        
        # Final interpolation to fill in the missing date values
        df = interpolate_by_val(df=df, by_val=stack_col,
                                sort_cols=by_cols,
                                interp_cols = list(df.iloc[:,2:]))
        return df
    
    train_weather = fill_in_time(df=train_weather, time_col='timestamp',
                                 stack_col='site_id')
    if save_time == False:
        test_weather = fill_in_time(df=test_weather, time_col='timestamp',
                                    stack_col='site_id')
    
    def handle_target_outliers(df):
        
        pre_removal_count = df.shape[0]
        print('\nMeter Reading by Meter Pre-Outlier Removal\n{}'
              .format(mf.describe_stats_by_val(df, 'meter', 'meter_reading')))
        
        # Identify/remove target outliers by meter type in training set
        print('\n...removing extreme-valued meter readings by meter type...')
        pre_count = df.shape[0]
        mf.detect_outliers(df=df, outlier_col=target, 
                           new_col='outliers_target_meter',
                           detect_type = 'MAD',
                           by_col='meter',
                           by_vals = list(df.meter.unique()))
        df = df.loc[df.outliers_target_meter.str.contains( 
                               'Normal', na=True)] # more dynamic; keeps na's
        removed = pre_count - df.shape[0]
        print('    Data points removed: {:,} ({:.1%})'.format( 
                  removed, removed / pre_count))

        print('\nMeter Reading by Meter Post-Outlier Removal\n{}'
              .format(mf.describe_stats_by_val(df, 'meter', 'meter_reading')))
        
        # Identify/remove target outliers by building ID in training set
        n=2 # number of highest variations to remove outliers by
        half1 = '\n...removing extreme-valued meter readings by top {}' \
                .format(n)
        half2 = ' highest variations by building ID...'
        print(half1+half2)
        pre_count = df.shape[0]
        
        # Use main data set to determine largest variation
        top_2 = mf.determine_largest_target_variation(df=df,
                                                      target_col=target,
                                                      by_col='building_id',
                                                      n=n,
                                                      return_top_n=True)
        top_2 = [k for k,v in top_2.items()] # get building id's from dict
        try:
            mf.detect_outliers(df=df, outlier_col=target, 
                               new_col='outliers_target_bid',
                               detect_type = 'MAD',
                               by_col='building_id',
                               by_vals = top_2)
            df = df.loc[df.outliers_target_bid.str.contains( 
                                   'Normal', na=True)]
            removed = pre_count - df.shape[0]
            print('    Data points removed: {:,} ({:.1%})'.format( 
                      removed, removed / pre_count))
        except: print('\nOutliers for building ID already identified')
        
        print('\nTarget value outliers removed = {:,} ({:.1%})\n'.
              format(pre_removal_count - df.shape[0], 
                      1-(df.shape[0]/pre_removal_count)))
        
        return df
    
    train = handle_target_outliers(df=train)
    
    dfs = [building, train, test, train_weather, test_weather]
    post_clean_length = mf.catalog_length(dfs=dfs)
    
    # Merge datasets
    '''
    inner on train removes missing year_built & floor_count rows
    length of train/test remains the same
    '''
    train = pd.merge(train, building, how='inner',
                     left_on='building_id', right_on='building_id')
    test = pd.merge(test, building, how='left',
                    left_on='building_id', right_on='building_id')
    
    # Density Plot by Primary Use
    if save_time == False:
        mp.myDens(dfi=train,
                  xColN=target,
                  pTitle='Density Plot by Primary Use', 
                  cat_col='primary_use', xColLab='Meter Reading', 
                  legendLab='Primary Use', save=save_plots)
    
        # Display distribution of data by Primary Use
        mf.tbl_vol_perc(df=train, col='primary_use')
    
    # Drop columns with missing features which can't be imputed & which are
    # not highly correlated to the target variable
    train.drop(['year_built', 'floor_count'], axis=1, inplace=True)
    test.drop(['year_built', 'floor_count'], axis=1, inplace=True)
    
    # Correlation Analysis Pre-FE
    change_encoding(df_list=[train],
                   cat_list=predictors_categor, encode_to_int=True)
    if save_time == False:
        mf.correlate(dfi=train, target_col=target, save=save_plots,
                     exclude_cols=['building_id', 'site_id', 'train_val_split', 
                                   'timestamp', 'outliers_target_meter',
                                   'outliers_target_bid'], #, 'outliers_target_sid'],
                     calc_corr_sigs=False) # 191206 True threw error
    
    print('\nTrain dataset length pre-clean: {}'.format(pre_clean_length))
    print('\nTrain dataset length post-clean: {}'.format(post_clean_length))
    print('\n=== Data Cleaning Complete ===')
    print('--- Runtime =', mf.timer(new_time, time.time()),'---')
    print('--- Running =', mf.timer(start_time, time.time()),'---')
    new_time = time.time()
    return building, train, test, train_weather, test_weather
    
building, train, test, train_weather, test_weather = \
    clean_data(new_time=new_time,
               building=building,
               train=train, test=test,
               train_weather=train_weather,
               test_weather=test_weather)

#%% Feature Engineering
def engineer_features(df, df_weath, new_time, to_combine=None):
    df_name = [x for x in globals() if globals()[x] is df][0] # df name to str
    print('\n[>>> Engineering {} features...]'.format(df_name)) 
    
    # Separate timestamps by date & time
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['time'] = df['timestamp'].dt.time
    df['hr'] = df['timestamp'].dt.hour
    ts_col = df.columns.get_loc('timestamp')
    d_col = df.columns.get_loc('date')
    cols = df.columns.tolist() # create list of cols
    cols = cols[:ts_col+1] + cols[d_col:] + cols[ts_col+1:d_col]
    df = df[cols] # reorder cols: place date/time by timestamp
    
    # Merge weather data to df set 
    by_cols=['site_id', 'timestamp']
    df = pd.merge(df, df_weath, how='left',
                  left_on=by_cols, right_on=by_cols)
    
    # Plot Meter Reading vs. Square Feet by Primary Use
    if target in list(df): # for train set only
        change_encoding(df_list=[df],
                        cat_list=['primary_use'], encode_to_int=False)
    
    # Handle irregular phenomena within meter readings
    if target in list(df): # for train set only
        if save_time == False:
            for meter_type in list(df.meter.unique()):
                mf.assess_by_cat(df=df[(df.meter_reading==0) & \
                                       (df.meter==meter_type)],
                                 by_cat_col='site_id', 
                                 agg_col='month', save=save_plots,
                                 main_title= \
            'Freq. of Zero-Value Readings by Month on Site ID (meter = {})'\
                                              .format(meter_type))
        
        # Remove zero-valued readings for site_id 0 for Jan-May days
        print('\n...removing irregular meter readings from site_id 0...')
        pre_count = df.shape[0]
        df = df[(df.site_id!=0) | (df.meter_reading!=0) | \
                (df.month>=6)]
        removed = pre_count - df.shape[0]
        print('    Data points removed: {:,} ({:.1%})'.format( \
                  removed, removed / pre_count))
        
        # Remove all zero-valued readings for electricity meter type
        print('\n...removing zero-value readings for eletricity meter...')
        if is_numeric_dtype(df.meter): # meter col encoded as int
            lbl = [k for k,v in mappers['map_meter'].items() \
                   if v=='electricity'][0]
        else: # primary_use col encoded as string
            lbl = [v for k,v in mappers['map_meter'].items() \
                   if v=='electricity'][0]
        pre_count = df.shape[0]
        df = df[(df.meter_reading!=0) | (df.meter!=lbl)]
        removed = pre_count - df.shape[0]
        print('    Data points removed: {:,} ({:.1%})'.format( \
                  removed, removed / pre_count))

    print('\n[>>> Transforming {} features...]'.format(df_name))
    
    # Create list of numeric features used in prediction  
    sf_col = df.columns.get_loc('square_feet')
    predictors_numeric = list(df.iloc[:, sf_col:]) #dynamic based on train/test

    # Create list of categorical & binary features used in prediction
    df['day_of_week'] = df.timestamp.dt.dayofweek
    df['is_workday'] = np.where((df['day_of_week'] == 5) | \
                                  (df['day_of_week'] == 6), \
                                  0, 1) # 0 = weekend, 1 = weekday
    holidays16 = [parse('1/1/16'), parse('1/18/16'), parse('2/15/16'),
                  parse('5/30/16'), parse('7/4/16'), parse('9/5/16'),
                  parse('10/10/16'), parse('11/11/16'), parse('11/24/16'),
                  parse('11/25/16'), parse('12/23/16'), parse('12/26/16')]
    df['is_holiday'] = df.timestamp.dt.date.astype( \
                      'datetime64').isin(holidays16).astype(int)
    
    TOD = [4,4,4,4,4,4,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3] # morn,af,eve,ni
    hr_to_TOD = dict(zip(range(1,25), TOD)) # morning,afternoon,evening,night
    df['time_of_day'] = df.timestamp.dt.hour.map(hr_to_TOD) #00:00:00 == NaN
    df['time_of_day'].fillna(value=4, inplace=True) # midnight vals from above
    bus_hrs = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0]
    hr_to_bus = dict(zip(range(1,25), bus_hrs))
    df['is_business_hr'] = df.timestamp.dt.hour.map(hr_to_bus)
    df['is_business_hr'].fillna(value=0, inplace=True) # midnight vals
    df['is_business_hr'] = np.where((df.is_business_hr==1) & \
                                    ((df.is_workday==0) | \
                                     (df.is_holiday==1)), 0, df.is_business_hr)
    
    # Seasonal encoding (spring=2 | summer=3 | fall=4 | winter=1)
    print('\n...adding season encodings...')
    seasons = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1] 
    month_to_season = dict(zip(range(1,13), seasons))
    df['season'] = df.timestamp.dt.month.map(month_to_season)
    
    # Remove all zero-valued readings logged consecutively by day by season
    if target in list(df): # for train set only
        def handle_target_outliers_by_season(df):
            
            # Address zero-valued readings in summer for steam/hotwater
            ''' 
            Normal for steam/hotwater to be turned off in summer;
            not normal for winter months, unless on wknds for fair weath sites
            Note: assumes all sites have fairly warm winters 
            '''
            season_code = 1 # winter
            meter_types = [2,3] #['steam', 'hotwater'] might expect off on wknd
            n_consec = 3 # concecutive acceptance threshold < n_consec (wknd)
            pre_count = df.shape[0]
            print('\n...removing {} consec-day 0-value readings for winter...'
                  .format(n_consec))
            for meter_type in meter_types: # remove consec 0val days >=thresh
                df = mf.remove_n_consec_zero(df=df,
                                             by_val_col='building_id',
                                             filter_col1='meter', 
                                             filter_val1=meter_type,
                                             filter_col2='season', 
                                             filter_val2=season_code,
                                             grouper_col='date',
                                             agg_col='meter_reading',
                                             n_consec=n_consec)

            removed = pre_count - df.shape[0]
            print('    Data points removed: {:,} ({:.1%})'.format( 
                      removed, removed / pre_count))
            
            # Address zero-valued readings in winter for chilled water
            ''' 
            Normal for chilledwater to be turned off in winter;
            not normal for summer months, unless on wknds for fair weath sites
            Note: assumes all sites have fairly cool summers 
            '''
            season_code = 3 # summer
            meter_types = [1] # ['chilledwater'] might expect off on wknd
            n_consec = 3 # concecutive acceptance threshold < n_consec (wknd)
            pre_count = df.shape[0]
            print('\n...removing {} consec-day 0-value readings for winter...'
                  .format(n_consec))
            for meter_type in meter_types: # remove consec 0val days >=thresh
                df = mf.remove_n_consec_zero(df=df,
                                             by_val_col='building_id',
                                             filter_col1='meter', 
                                             filter_val1=meter_type,
                                             filter_col2='season', 
                                             filter_val2=season_code,
                                             grouper_col='date',
                                             agg_col='meter_reading',
                                             n_consec=n_consec)

            removed = pre_count - df.shape[0]
            print('    Data points removed: {:,} ({:.1%})'.format( 
                      removed, removed / pre_count))
                
            # New Histogram of target w/o outliers post-clean
            if save_time == False:
                try:
                    mp.myHist(dfi=df,
                              xColN=target, ibins=20,
                              pTitle='Meter Reading Histogram',
                              pSubTitle='Outliers Removed Post Seasonal Clean',
                              xColLab='Meter Reading', save=save_plots)
                except: next
            
            return df
        
        df = handle_target_outliers_by_season(df)
        
    # Visualize target variance by site_id (post-outlier removal)
    if target in list(df) and save_time == False: # for train set only
        '''
        191207 - ran a pre-outlier removal separately by selective runs to get
        merged data set as-is and saved to exports for comparison
        '''
        print('...visualizing target variance by site_id...')
        def visualize_target_variance_by_site(dfi):
            df = dfi[['timestamp', 'site_id', target]].groupby( \
                 ['timestamp', 'site_id']).agg('median').reset_index()
            df = df.rename({target:'median_'+target}, axis=1)
            mp.matrix_series(df=df, by_cols=['median_'+target],
                             time_col='timestamp', by_vals=True,
                             filter_col='site_id',
                             main_title='Time series (post-outlier removal)',
                             save=save_plots)
        visualize_target_variance_by_site(df)
        
        def visualize_target_variance_by_meter(dfi):
            df = dfi[['timestamp', 'meter', target]].groupby( \
                 ['timestamp', 'meter']).agg('median').reset_index()
            df = df.rename({target:'median_'+target}, axis=1)
            mp.matrix_series(df=df, by_cols=['median_'+target],
                             time_col='timestamp', by_vals=True,
                             filter_col='meter',
                             main_title='Time series',
                             save=save_plots)
        visualize_target_variance_by_meter(df)
    
    # Transform numeric predictor data
    print('\Desc. Stats for Numeric Predictor Fields Pre-Transform:\n{}'
          .format(mf.describe_stats_by_col(df, predictors_numeric)))
    
    mf.log_transform_numeric(df, predictors_numeric)
        
    print('\Desc. Stats for Numeric Predictor Fields Post-Transform:\n{}'
          .format(mf.describe_stats_by_col(df, ['log_'+col 
                                        for col in predictors_numeric])))
        
    # Transform target col to fix skew 
    def plot_target_dist_logged():
        mp.my_distribution(df[target], save=save_plots,
                           pTitle='Distribution of Meter Reading in kWh')
        mp.my_distribution(np.log1p(df[target]), save=save_plots,
                           pTitle='Distribution of Logged Meter Reading')
            
    if target in list(df): # for train set only
        plot_target_dist_logged()


    # Combine primary_use types in broader categories by energy usage
    '''
    !!! 191209 - there are some primary uses categories which do show a dip
    in avg meter readings on week nums 5 & 6 (weekends), but will let model
    learn from is_weekend feature before adding addition features below
    '''
    if target in list(df) and save_time == False: # for train set only
        for meter_type in list(df.meter.unique()):
            mf.assess_by_cat(df.loc[df.meter==meter_type],
                  by_cat_col='primary_use', 
                  agg_col='day_of_week',
                  main_title= \
        'Meter Reading Avg by Day of Week on Primary Use (meter type={})'\
                              .format(meter_type),
                  agg_type='avg', agg_target=target, save=save_plots)

    print('\n...combining primary_use by time-based usage type...')
    weekend_light = ['Technology/science', 'Utility', 'Retail']
    even_distrib = list(set(list(train.primary_use.unique())) - \
                   set(weekend_light))
    primary_use_lists = [weekend_light] + [even_distrib]
    primary_use_labels = [0, 1]
    primary_use_mapper = {}
    def map_primary_use(lab, lis, primary_use_mapper):
        primary_use_mapper.update({i:lab for i in lis})
        return primary_use_mapper
    for lab, lis in zip(primary_use_labels, primary_use_lists):
        primary_use_mapper = map_primary_use(lab, lis, primary_use_mapper)
    df['use_categ'] = df.primary_use.map(primary_use_mapper)
    
    # Uniquely append predictors_categor list
    for i in ['season']:
        predictors_categor.append(i) if i not in predictors_categor \
                                     else predictors_categor

    print('\n[>>> Encoding {} features...]'.format(df_name))
    
    # One-hot encode categorical predictors
    change_encoding(df_list=[df],
                   cat_list=predictors_categor, encode_to_int=True)
    for cat in predictors_categor:
        df = pd.concat([df, pd.get_dummies(df[cat], prefix=cat)], axis=1)
            
    # Correlation Analysis on Numeric Predictors Post-FE
    if target in list(df) and save_time == False: # for train set only
        include_cols = [target]+[f for f in list(df.columns) if 'log_' in f]
        mf.correlate(dfi=df, target_col=target, save=save_plots,
                          exclude_cols=[feat for feat in \
                                        list(df.columns) \
                                        if feat not in include_cols],
                          calc_corr_sigs=False) #191206 True throws error
        '''
        XXX Note: no real change in findings from original correlation analysis
        run in data cleaning phase aside from wind_direction / wind_speed & 
        air_temp / dew_temp. Could remove 1 of each.
        '''
    return df, predictors_numeric

# Run feature engineering
train, predictors_numeric = engineer_features(df=train, 
                                              df_weath=train_weather, 
                                              new_time=new_time) 
if save_time == False:
    test, _  = engineer_features(df=test, 
                                 df_weath=test_weather,
                                 new_time=new_time) 

print('\n=== Feature Engineering Complete ===')
print('--- Runtime =', mf.timer(new_time, time.time()),'---')
print('--- Running =', mf.timer(start_time, time.time()),'---')
new_time = time.time()

#%% Model Data Preprocessing

print('\n[>>> Preprocessing data for model training...]')

print('\nDescriptive Stats for Target Variable:\n{}'
      .format(mf.describe_stats_by_col(train, [target])))

# Sample if train set is still too large
def preprocess_for_training(dfi, reduce_to=1000000, log_y=False, 
                            filter_by_col=None, filter_by_val=None):
    
    if dfi.shape[0] > reduce_to:
        dfi = dfi.sample(n=reduce_to, random_state=5)
    
    # Train/Validation Split
    if filter_by_col != None: # used in for loop by meter type
        X_train = dfi.loc[(dfi[filter_by_col]==filter_by_val) & \
                          (dfi.train_val_split==train_val_dict['train'])] \
                          .drop(target, axis=1)
        X_val = dfi.loc[(dfi[filter_by_col]==filter_by_val) & \
                          (dfi.train_val_split==train_val_dict['val'])] \
                          .drop(target, axis=1)
        y_train = dfi[target].loc[(dfi[filter_by_col]==filter_by_val) & \
                            (dfi.train_val_split==train_val_dict['train'])]
        y_val = dfi[target].loc[(dfi[filter_by_col]==filter_by_val) & \
                            (dfi.train_val_split==train_val_dict['val'])]
    else:
        X_train = dfi.loc[dfi.train_val_split==train_val_dict['train']] \
                    .drop(target, axis=1)
        X_val = dfi.loc[dfi.train_val_split==train_val_dict['val']] \
                    .drop(target, axis=1)
        y_train = dfi[target].loc[dfi.train_val_split==train_val_dict['train']]
        y_val = dfi[target].loc[dfi.train_val_split==train_val_dict['val']]
        
    # Trainsform y if called
    if log_y == True:
        y_train, y_val = np.log1p(y_train), np.log1p(y_val)
    
    print('\nTraining on {:,} observations. Validating on {:,}.'.format( \
      X_train.shape[0], X_val.shape[0]))
    return X_train, X_val, y_train, y_val

reduce_to = 2000000 # ceiling
X_train, X_val, y_train, y_val = \
        preprocess_for_training(dfi=train,
                                reduce_to=reduce_to,
                                log_y=log_y)

# Establish baseline performance measure using naive prediction
def establish_baseline(s_train, s_val):
    
#    # Reverse y transformation, if called
    if log_y == True:
        s_train = np.expm1(s_train) 
        s_val = np.expm1(s_val)
    
    baseline_guess = np.median(s_train)
    baseline_perf_val = np.sqrt(mean_squared_log_error(s_val, 
                                [baseline_guess]*s_val.shape[0]))
    print('\nBaseline RMSLE using median, naive approach = {:.5}'. 
          format(baseline_perf_val))
establish_baseline(s_train=y_train, s_val=y_val)

# Final feature selection
def select_features():
    keep = list(
            X_train.iloc[:, X_train.columns.get_loc( 
                            'log_square_feet'):].columns)
    remove = ['log_wind_direction', 'log_wind_speed',
              'log_sea_level_pressure', 'log_precip_depth_1_hr',
              'log_cloud_coverage']
    ''' 
    some may have already been removed above
    may choose to keep log_dew_temperature later
    '''
    
    keep = [feat for feat in keep if feat not in remove]
    keep = keep + ['is_workday', 'is_holiday', 'is_business_hr']
    print('\nTraining on {} features: '.format(len(keep)))
    for f in keep: print('  {}'.format(f))
    print('\nSample:Feature ratio = {:,}:1'.\
          format(int(len(X_train)/len(keep))))
    return keep
trainable = select_features()

# Final handling of missing data #!!! this is not great logic
print('\nMissing Values Check:\n train\n{}\n test\n{}'.format( \
      list(train.columns[train.isnull().any()]), \
      list(test.columns[test.isnull().any()])))
try:
    test[trainable] = test[trainable].fillna(0)
except:
    print('!!! Unable to replace missing values in test set !!!')

# Compare train vs. test
print('\nFeatures found in train but not in test:\n{}'.format( \
      list(set(train)-set(test))))
print('\nFeatures found in test but not in train:\n{}'.format( \
      list(set(test)-set(train))))

# Feature scaling
def scale_features():
    features_to_scale = [feat for feat in trainable if 'log_'  in feat]
    scaler = MinMaxScaler(feature_range=(0, 1)) # instanstiate
    scaler.fit(X_train[features_to_scale]) # fit on training data
    
    # Scale train/test sets
    X_train[features_to_scale] = scaler.transform(X_train[features_to_scale]) 
    X_val[features_to_scale] = scaler.transform(X_val[features_to_scale])
scale_features()

# Set up sampled sets for rapid prototyping with 
def random_sample(df, s_size=.2, gen_rule=False, features=[], r_state=5):
    if gen_rule==True and len(features)!=0:
        s_size = len(features)*10 / df.shape[0] # >=10:1 ratio, obs:variable
    df = df.sample(frac=s_size, random_state=r_state)
    return df

if rand_samp == 0: # full data sets were imported at start, so sample
    s_size = 0.001
    X_train_rp, X_val_rp, y_train_rp, y_val_rp = \
        random_sample(X_train, s_size=s_size), \
        random_sample(X_val, s_size=s_size), \
        random_sample(y_train, s_size=s_size), \
        random_sample(y_val, s_size=s_size)
    mlrp = mdl.MyModel(trainable, X_train_rp, X_val_rp, 
                       test, y_train_rp, y_val_rp)

# Instantiate machine learning model object
ml = mdl.MyModel(trainable, X_train, X_val, test, y_train, y_val, log_y=log_y)

print('\n=== Model Data Preprocessing Complete ===')
print('--- Runtime =', mf.timer(new_time, time.time()),'---')
print('--- Running =', mf.timer(start_time, time.time()),'---')
new_time = time.time()

#%% Model Training 

## Top performer
ml_name='gbr'
ml.gbr()
if not np.isscalar(ml.predictions[ml_name][0]): #keras
    predictions = pd.Series([x[0] for x in ml.predictions[ml_name]]) #flatten
else: #scikit
    predictions = ml.predictions[ml_name]
mp.plot_pred_vs_true(y_true=ml.y_val, y_pred=predictions, 
                     pTitle='Predicted vs. True', save=save_plots)

#%% Sub-Model Training Attempt

# Train on each subset of data
def train_by_meter(df, trainable, reduce_to, ml_idx, export=False):
    train_start_time = time.time()
    ml_name = ml_names[ml_idx]
    mls = None # remove ref to obj for next interation
    mls = [mdl.MyModel() for i in range(len(df.meter.unique()))]
    ml_dict = {}
    rmsle_dict = {}
    all_pred = np.array([])
    all_true = np.array([])
    all_pred_dict = {}
    all_true_dict = {}
    
    for meter, ml_tmp in zip(sorted(list(df.meter.unique())), mls):
        
        print('\n[>>> Training {} models by meter type {}...]\n'.format( 
              ml_name, meter))
        new_time = time.time()
        
        # Define train/test split by meter type
        X_train, X_val, y_train, y_val = \
            preprocess_for_training(dfi=df,
                                    reduce_to=reduce_to,
                                    log_y=log_y,
                                    filter_by_col='meter', filter_by_val=meter)
        ml_tmp.X_trainable = trainable
        ml_tmp.X_test = test
        ml_tmp.X_train = X_train[trainable] #override ml() which initiates w/[]
        ml_tmp.X_val = X_val[trainable] #override ml() which initiates w/[]
        ml_tmp.y_train = y_train
        ml_tmp.y_val = y_val
        
        # Train model
        if export!=False:
            suff = str(len(trainable))+'_o'+str(df.shape[0])+ \
                    '_meter_'+str(meter)
            file_name = ml_name + '_f'+suff
        else: file_name=None # pass default arg
        
        if ml_name == 'lr':
            ml_tmp.lr(file_name=file_name)
        elif ml_name == 'rf':
            ml_tmp.rf(file_name=file_name)
        elif ml_name == 'gbr':
            ml_tmp.gbr(file_name=file_name)
        elif ml_name == 'mlp':
            ml_tmp.mlp(file_name=file_name)
        elif ml_name == 'xgbr':
            ml_tmp.xgbr(file_name=file_name)
        elif ml_name == 'svm':
            ml_tmp.svm(file_name=file_name)
        else:
            print('!!! Model selected not incorporated in loop !!!')
            break
        
        # Capture predictions for plotting
        if not np.isscalar(ml_tmp.predictions[ml_name][0]): # flatten array
            predictions = pd.Series([x[0] for x in \
                                     ml_tmp.predictions[ml_name]])
        else: #scikit
            predictions = ml_tmp.predictions[ml_name]
        
        # Catalog model, rmsle, & true/pred y
        ml_dict.update({ml_name+'_'+str(meter): ml_tmp})
        rmsle_dict.update({'meter_'+str(meter): ml_tmp.performance[ml_name]})
        all_pred = np.append(all_pred, predictions)
        all_true = np.append(all_true, ml_tmp.y_val)
        all_pred_dict.update({'meter_'+str(meter): predictions})
        all_true_dict.update({'meter_'+str(meter): ml_tmp.y_val})
        
        # Plot predicted vs. true value
        mp.plot_pred_vs_true(y_true=ml_tmp.y_val, y_pred=predictions, 
                             pTitle='Predicted vs. True for Meter '+str(meter),
                             save=save_plots)
        print('Meter {} RMSLE: {:.4}'.format( \
              meter, ml_tmp.performance[ml_name]))
        
        ml_tmp = None # remove ref to obj for next interation
        
        print('--- Runtime =', mf.timer(new_time, time.time()),'---')
        new_time = time.time()
    
    # Evaluate final results
    rmsle_avg = statistics.mean([v for k,v in rmsle_dict.items()])
    print('\nRMSLE average for all models: {:.4}'.format(rmsle_avg))
    rmsle_all = mls[0].rmsle_eval(y_true=all_true, y_pred=all_pred)
    print('\nRMSLE for all models: {:.4}'.format(rmsle_all))
        
    # Plot overall predicted vs. true value
    mp.plot_pred_vs_true(y_true=all_true, y_pred=all_pred, save=save_plots,
                         pTitle='Predicted vs. True All {} Models' \
                         .format(ml_name))
    
    print('--- Total Training Runtime =', \
          mf.timer(train_start_time, time.time()),'---')
    
    return ml_dict, all_true, all_true_dict, all_pred, all_pred_dict

ml_names = ['lr', 'rf', 'gbr', 'mlp', 'xgbr', 'svm'] # used for ml_idx arg below
ml_dict, all_true, all_true_dict, all_pred, all_pred_dict = \
                        train_by_meter(df=train, trainable=trainable,
                                       reduce_to=reduce_to,
                                       ml_idx=2) #, export=True)

#%% Model 01: Linear Regression
#ml.lr(file_name='linear_regression_f'+str(len(trainable)))
#mlrp.lr() # rapid prototyper
#print('--- Runtime =', mf.timer(new_time, time.time()),'---')
#new_time = time.time()

#%% Model 02: K-Nearest Neighbors
#ml.knn(file_name='k_nearest_neighbor_f'+str(len(trainable)))
#mlrp.knn() # rapid prototyper
#print('--- Runtime =', mf.timer(new_time, time.time()),'---')
#new_time = time.time()

#%% Model 03: Random Forest
#ml.rf(file_name='random_forest_f'+str(len(trainable)))
#mlrp.rf() # rapid prototyper
#print('--- Runtime =', mf.timer(new_time, time.time()),'---')
#new_time = time.time()

#%% Model 04: Gradient Boosted Regression
#ml.gbr(file_name='gradient_boosted_f'+str(len(trainable)))
#mlrp.gbr() # rapid prototyper
#print('--- Runtime =', mf.timer(new_time, time.time()),'---')
#new_time = time.time()

#%% Model 05: Support Vector Machine
#ml.svm(file_name='support_vector_machine_f'+str(len(trainable)))
#mlrp.svm() # rapid prototyper
#print('--- Runtime =', mf.timer(new_time, time.time()),'---')
#new_time = time.time()

#%% Model 06: Multi-Layer Perceptron Neural Network
#ml.mlp(file_name='mlp_f'+str(len(trainable)))
#mlrp.mlp(file_name='mlp_rp_f'+str(len(trainable))) # rapid prototyper
#print('--- Runtime =', mf.timer(new_time, time.time()),'---')
#new_time = time.time()

#%% Model 07: XGBoost
#ml.xgb(file_name='XGBoost_f'+str(len(trainable)))
#mlrp.xgb() # rapid prototyper
#print('--- Runtime =', mf.timer(new_time, time.time()),'---')
#new_time = time.time()

print('\n=== Model Training Complete ===')
print('--- Runtime =', mf.timer(new_time, time.time()),'---')
new_time = time.time()


#%% Apply final model to test set
X_test = test[trainable].copy()

## Make predictions on test data
def predict_on_test(model_name, ml_obj=ml, file_name=None):
    if file_name == None: # from in-memory model (untested)
        ml_obj.predict(ml_obj.models[model_name], test[trainable])
    else: # from load
        ml = mdl.MyModel(X_trainable=trainable, X_test=test)
        final_model = ml.load_model(model_name=model_name, file_name=file_name)
        ml.predict(final_model, ml.X_test[trainable])
        
predict_on_test(model_name=ml_name) # use if in-memory

# Export for submission
submission = pd.concat([test.row_id,
                        pd.DataFrame(ml.predictions[ml_name],
                                     columns=[target])],
                        axis=1, sort=False)
                        
# Check for missing: https://www.kaggle.com/c/ashrae-energy-prediction/submit
print('\nMissing predictions: {:,}'.format(abs((submission.shape[0]-41697600))))
mf.export_df(df=submission, path=path, 
             df_name='submission_ThomasShantz')
print('\nTest set predictions exported to *.csv')

print('\n--- Total Runtime =', mf.timer(start_time, time.time()),'---')
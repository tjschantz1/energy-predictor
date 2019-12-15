from collections import defaultdict
from collections import OrderedDict
from bisect import bisect_left
import pandas as pd
import numpy as np
import os
import random
import myplotters as mp #home-grown
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from more_itertools import unique_everseen
import matplotlib.pyplot as plt
import warnings
import operator
import math
import time

fsize_x = 10
fsize_y = 8

    
#%% Path Getter
def get_path(subdir=None):
    curdir = os.getcwd() # get path to current directory
    if subdir!=None:
        if os.name == 'nt': #Windows Machine
            subdir = '\\'+subdir+'\\'
        elif os.name == 'posix': #MacBook Pro
            subdir = '/'+subdir+'/'
    path = curdir+subdir
    return path

#%% Run-Time Timer
def timer(start, end):
   hours, rem = divmod(end-start, 3600)
   minutes, seconds = divmod(rem, 60)
   return('{:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds))

#%% Datetime Converter
def dt_convert(df, colList): # should convert any string input for datetime
    for c in colList: # float values & blanks are cast to NaT
        df[c] = pd.to_datetime(df[c], errors='coerce') 
        
#%% csv to DataFrame Importer
def import_data(path, file_name, datetimeCol=None, rand_samp=0):
    print('\n   Importing {}...'.format(file_name))
    if rand_samp>0: # import a seeded random sample % of the data length
        random.seed(3)
        df = pd.read_csv(path+file_name, header=0, skiprows=\
                         lambda i: i>0 and random.random() > rand_samp)
    else: # import entire length of data
        df = pd.read_csv(path+file_name)
    
    # Convert object type to datetime   
    try: dt_convert(df,[datetimeCol])
    except: None
    
    # Reduce Memory Usage
    print('   {} imported successfully. Reducing memory usage...'\
          .format(file_name))
    df = reduce_mem_usage(df)
    df.name = file_name.replace('.csv','')
    #!!!df.name is easilly overwritten in normal pd operations
    
    return df

#%% Field Encoder/Unencoder
def map_field(df, map_col=None, dict_map={}, reverse=False, encode_col=None):
    
    # For creating dictionary maps & encoding cols at once
    if encode_col != None:
        encoder = LabelEncoder()
        original_label = list(unique_everseen(df[encode_col])) #uniqueunordered
        encoder.fit(df[encode_col]) # encode encode_col
        df[encode_col] = encoder.transform(df[encode_col])
        encoded_label = list(unique_everseen(df[encode_col])) #uniqueunordered
        label_dict = dict(zip(encoded_label, original_label))
        
    # For already created dictionary maps (ability to encode & unencode)
    if dict_map != {}:
        if reverse == True: # reverse key-value relationship 
            dict_map = {v:k for k,v in dict_map.items()}
        df[map_col] = df[map_col].map(dict_map)
        label_dict = None
    
    return label_dict

#%% Create missing data tables
def tbl_missing(df):
    volume = df.isnull().sum()
    percent = volume/df.isnull().count()
    missing_tbl = pd.concat([volume, percent], axis=1, 
                     keys=['volume', 'percent']).sort_values(ascending=False,
                          by=['volume'])
    missing_tbl.volume = ['{:,}'.format(x) for x in missing_tbl.volume]
    missing_tbl.percent = ['{:.1%}'.format(x) for x in missing_tbl.percent]
    return missing_tbl

#%% Create aggregate count & percentage table by value in df col
def tbl_vol_perc(df, col):
    volume = df[col].value_counts()
    percent = df[col].value_counts(normalize=True)
    vol_perc_tbl = pd.concat([volume, percent], axis=1, 
                             keys=['volume', 'percent']).sort_values(\
                                  ascending=False,
                                  by=['volume'])
    vol_perc_tbl['cumulative_perc'] = vol_perc_tbl.percent.cumsum(skipna=True)
    vol_perc_tbl.volume = ['{:,}'.format(x) for x in vol_perc_tbl.volume]
    vol_perc_tbl.percent = ['{:.1%}'.format(x) for x in vol_perc_tbl.percent]
    vol_perc_tbl['cumulative_perc'] = ['{:.1%}'.format(x) \
                 for x in vol_perc_tbl['cumulative_perc']]
    return vol_perc_tbl

#%% Catalog length of df
def catalog_length(dfs):
    catalog = {}
    counter = 0
    for df in dfs:
        counter+=1
        try: df_name = [x for x in globals() if globals()[x] is df][0]
        except: df_name = 'df'+str(counter)
        catalog.update({df_name:df.shape[0]})
    return catalog

#%% Identify category label with largest variation with target value
def determine_largest_target_variation(df, target_col, by_col, n=1, 
                                       return_top_n=False):
    variation_dict = {}
    for i in list(df[by_col].unique()):
        std_dev = df[target_col][df[by_col]==i].std()
        variation_dict.update({i:std_dev})
    top_n = dict(sorted(variation_dict.items(), \
                        key=operator.itemgetter(1), \
                        reverse=True)[:n])
    print('\nTop {} highest variations found in:\n{}'.format( \
          n, top_n))
    if return_top_n==True:
        return top_n

#%% Remove columns with sparse data
def remove_sparse_features(dfs, thresh=0.5):
    for df in dfs:
        tbl = tbl_missing(df)
        remove_cols = list(tbl.index[tbl.percent\
                                     .str.replace(r'%','')\
                                     .astype(float)/100>thresh])
        df.drop(remove_cols, axis=1, inplace=True)
        
# Ensure mirrored data sets remain mirrored after column removal
def match_df_cols(df1, df2):
    all_cols = list(df1.columns) + list(df2.columns)
    unique_cols = list(OrderedDict.fromkeys(all_cols))
    try: df1.drop(set(unique_cols) ^ set(df2.columns), axis=1, inplace=True)
    except: next
    try: df2.drop(set(unique_cols) ^ set(df1.columns), axis=1, inplace=True)
    except: next
        
#%% Remove category labels with sparse data
def remove_sparse_labels(df, cat_col, thresh=0.995, print_pre=False, 
                         ignore_col=None, ignore_val=None):
    tbl = tbl_vol_perc(df=df, col=cat_col)
    if print_pre == True:
        print('\nDistribution of Data by {}:\n{}'.format(
                cat_col, tbl))
    remove_labels = list(tbl.index[tbl['cumulative_perc'] \
                                   .str.replace(r'%','') \
                                   .astype(float)/100>=thresh])
    if ignore_col!=None:
        df = df[~df[cat_col].isin(remove_labels)]
    else: # remove sparse while ignoring certain rows (e.g. validation rows)
        df = df[(df[ignore_col]==ignore_val) | \
                (~df[cat_col].isin(remove_labels))]
    print('\nRemoved category labels:\n{}'.format(remove_labels))
    return df

#%% Combine category labels with sparse data
def identify_sparse_labels(df, cat_col, lab_rename, thresh=0.005, 
                           print_pre=False):
    tbl = tbl_vol_perc(df=df, col=cat_col)
    if print_pre == True:
        print('\nDistribution of Data by {}:\n{}'.format(
                cat_col, tbl))
    combine_labels = list(tbl.index[tbl['percent'] \
                                   .str.replace(r'%','') \
                                   .astype(float)/100<=thresh])
    return combine_labels

def combine_sparse_labels(df, cat_col, lab_rename, combine_labels):
    for label in combine_labels:
        df[cat_col] = np.where(df[cat_col]==label, lab_rename, df[cat_col])
    print('\nCombined category labels for {}:\n{}'.format( \
          cat_col, combine_labels))
    return df

#%% Matrix Plot of Freq vs aggregation column by value of category column
def assess_by_cat(df, by_cat_col, agg_col, main_title='', print_tbls=False,
                  return_df=False, agg_type='freq', agg_target=None,
                  save=False):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    # Sort by_cat_col vals alphabetically for prettier printing
    by_cat_vals = sorted(list(df[by_cat_col].unique()))
    
    # Stats: by_cat_col vs. agg_col
    from scipy.stats import mode
    piv = pd.pivot_table(df, values=agg_col, index=[by_cat_col],
                         aggfunc=[len, min, np.mean, np.std, 
                                  lambda x:mode(x).mode[0], max], 
                                  dropna=False)
    piv.columns = ['count', 'min', 'mean', 'std', 'mode', 'max']
    piv.fillna(0, inplace=True)
    piv = np.round(piv, 2) #piv.round(2) # not working
    piv = piv.astype({i:np.int32 for i in ['count', 'min', 'mode', 'max']})
    piv.sort_index(inplace=True)
    piv.index.name = None # flatten column headers
    
    na_dict = {} # for capturing vol nan in for loop below
    plt.rcParams.update({'font.size': 6})
    num_vals = len(by_cat_vals)
    def round_up(n, decimals=0):
        import math
        multiplier = 10 ** decimals 
        return int(math.ceil(n * multiplier) / multiplier)
    cols = round_up(np.sqrt(num_vals),0)
    rows = round_up((num_vals/cols),0)
    fig = plt.figure(figsize=(fsize_x,fsize_y))
    gs = plt.GridSpec(rows, cols)
    for val, n in zip(by_cat_vals, range(num_vals)):
        
        # Count nan by val
        num_na = df[df[by_cat_col] == val][agg_col].isnull().sum()
        try: perc_na = num_na / df[df[by_cat_col] == val].shape[0]
        except: perc_na = 0
        na_dict.update({val:[num_na, perc_na]})
        
        # Create matrix of subplots & add
        i = n%cols
        j = n//cols
        plt.subplot(gs[j,i])
        if agg_type == 'freq':
            agg_by_col = df[df[by_cat_col] == val][agg_col].value_counts()
            plt.bar(agg_by_col.index, agg_by_col)
        elif agg_type == 'avg':
            agg_by_col = pd.pivot_table(df[df[by_cat_col] == val],
                                        values=agg_target, 
                                        index=[agg_col], aggfunc=np.mean)
            plt.bar(agg_by_col.index, agg_by_col.iloc[:,0])
        plt.yticks(fontsize=8) #overrides params.update
        
        '''Below doesn't work for all cases (when there are blank spots in 
                                             beginning or end of plot'''
#        if j+1 == cols: # apply xtick labels to last row of matrix
#            x_labs = sorted(list(df[agg_col].unique())) #agg_by_col.index 
#            plt.xticks(range(len(x_labs)), x_labs, fontsize=8)
#        else: # remove xtick label for all others
#            plt.tick_params(labelbottom=False)
        plt.title(val, fontsize=10) #fontweight='semibold')
    
    # Print final plot to screen
    plt.tight_layout()
    fig.subplots_adjust(top=0.88) #create space for main title
    if main_title=='':
        main_title = '{} vs. {} by {}'.format(agg_type, agg_col, by_cat_col)
    fig.suptitle(main_title, fontsize=11)
    if save==True: mp.save_high_res(prefix='matrix_countby_')
    plt.show()
    
    # Print missing table by val to screen
    if print_tbls==True:
        missing = pd.DataFrame.from_dict(na_dict, orient='index',
                                         columns=['volume', 'perc_row_total'])
        try: missing['perc_col_total'] = missing.iloc[:,0] / \
                                            missing.iloc[:,0].sum()
        except: missing['perc_col_total'] = 0
        missing.volume = ['{:,}'.format(x) for x in missing.volume]
        missing.perc_row_total = ['{:.1%}'.format(x) \
                                 for x in missing.perc_row_total]
        missing.perc_col_total = ['{:.1%}'.format(x) \
                                     for x in missing.perc_col_total]
        missing.sort_index(inplace=True)
        print('\nMissing {} data by {}\n'.format(agg_col, by_cat_col), missing)
        print('\nStats on {} by {}\n'.format(agg_col, by_cat_col), piv)
    if return_df==True:
        return piv

#%% Remove missing values from multiple df's
def remove_missing(dfs, subset=[]):
    for df in dfs:
        if subset==[]: df.dropna(inplace=True)
        else: df.dropna(subset = subset, inplace=True)
        
#%% Custom back filler
def back_fwd_filler(df, col):
    
    # Get list of indices within df which contain valid vals after NaN'
    filler_indx = df[col].index.get_indexer(df[col].index[~df[col].isnull()])
    
    # Iterrate over rows & fill
    for i,(index,row) in enumerate(df.iterrows()): # also handles indx start >0
        if math.isnan(df[col].values[i]): # found NaN
            if i == 1: # NaN found in 1st row; use 1st indx (back)
                indx = 0
            elif i == len(df) - 1: # NaN found in last row; use last indx (fwd)
                indx = len(filler_indx) - 1
            else: # NaN found in middle rows; use backwards logic 
                try: # won't work if i>filler_indx.amx (shouldn't happen)
                    indx = next(x[0] for x in enumerate(filler_indx) if x[1] > i)
                except: break # exit if filler_indx=[]
            df[col].iloc[i] = df[col].iloc[filler_indx[indx]]
            
#%% Removing consecutive rows which match a condition
def remove_n_consec_zero(df, by_val_col, grouper_col, agg_col, 
                         filter_col1, filter_val1, \
                         filter_col2=None, filter_val2=None, \
                         n_consec=2):
    '''
    by_val_col = col to iterate through (e.g. building_id)
    filter_col = col to filter by (e.g. meter)
    filter_val = value to filter by within filter_col (e.g. 0)
    grouper_col col to group by (e.g. date)
    agg_col = col to aggregate by (e.g. meter_reading)
    '''
    
    start_time = time.time()
    new_time = time.time()
    to_remove = set() # for capturing indices to be removed from master df

    # Define function for determining n-consectutive days
    def f(col, threshold):
        mask = col.groupby((col != col.shift()).cumsum()).transform( \
                          'count').lt(threshold)
        mask &= col.eq(0)
        col.update(col.loc[mask].replace(0,1))
        return col

    # Iterate over vals in by_val_col and identify n-conesc days of 0vals
    vals = list(df[by_val_col].unique())
    iter_count = 0
    for val in vals:
        iter_count+=1
        
        # Create filtered df to remove vals from once indices are established
        if filter_col2==None:
            df_filt = df.loc[(df[by_val_col]==val) & \
                                 (df[filter_col1]==filter_val1)]
        else:
            df_filt = df.loc[(df[by_val_col]==val) & \
                                 (df[filter_col1]==filter_val1) & \
                                 (df[filter_col2]==filter_val2)]
        
        # Create grouped df to idenfity indices where there are n-consec 0's
        df_by = df_filt.groupby([grouper_col])[agg_col].agg(['sum'])

        
        # Update indices to remove
        df_by.apply(f, threshold=n_consec)  # replace "allowable" 0 values
        to_keep = df_by['sum'].nonzero() # return arr of non0 indices
        to_keep = df_by.iloc[to_keep].index.values.tolist() #dates
        to_remove.update(df_filt[~df_filt[grouper_col].isin( \
                                 to_keep)].index.values)
        
        if iter_count%100 == 0:
            print('[====> {} of {} complete in {} (cum remove count = {})]' \
                    .format(iter_count, len(vals),
                            timer(new_time, time.time()),
                            len(to_remove)))
        new_time = time.time()
    
    df_final = df.drop(to_remove, axis=0)
        
    print('Total elapsed time = {}'.format(timer(start_time, time.time())))
    print('Total removed = {}\n'.format(len(to_remove)))
    
    return df_final
            

#%% Descriptive Statistic Generators
def describe_stats_by_col(df, cols, index_name=None):
    min_by_col = df[cols].min()
    mean_by_col = df[cols].mean()
    std_by_col = df[cols].std()
    median_by_col = df[cols].median()
    mad_by_col = df[cols].mad()
    max_by_col = df[cols].max()
    stats_tbl = pd.concat([min_by_col, mean_by_col, std_by_col,
                            median_by_col, mad_by_col, max_by_col], axis=1, 
                            keys=['min','mean','std dev','median','mad','max'])
    stats_tbl = stats_tbl.rename_axis(index=index_name)
    stats_tbl = stats_tbl.round(3) # set sig digits  
    return stats_tbl

def describe_stats_by_val(df, by_val_col, agg_col, index_name=None):
       
    # Stats: by_val_col on agg_col
    from scipy.stats import median_absolute_deviation as mad
    piv = pd.pivot_table(df, values=agg_col, index=[by_val_col],
                         aggfunc=[len, min, np.mean, np.std, 
                                  np.median,
                                  lambda x: mad(x),
                                  max], 
                         dropna=False)
    piv.columns = ['count', 'min', 'mean', 'std', 'median', 'mad', 'max']
    piv.fillna(0, inplace=True)
    piv = np.round(piv, 2) #piv.round(2) # not working
#    piv = piv.round({'mean':2, 'std':2}) # not working
    piv = piv.astype({i:np.int32 for i in ['count']})
    piv.sort_index(inplace=True)
    piv.index.name = None # flattent column headers
    
    return piv

#%% Interval Counter
def count_intervals(sequence, intervals):
    count = defaultdict(int)
    intervals.sort()
    for item in sequence:
        pos = bisect_left(intervals, item)
        if pos == len(intervals):
            count[None] += 1
        else:
            count[intervals[pos]] += 1
    return count

    #Example:
    #data = [4,4,1,18,2,15,6,14,2,16,2,17,12,3,12,4,15,5,17]
    #print(count_intervals(data, [10, 20]))
    
#%% Reduce DF memory size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and \
                    c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and \
                    c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and \
                     c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and \
                    c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and \
                    c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and \
                    c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('   Memory usage decreased to {:5.2f} Mb ({:.1f}%)'\
                      .format(end_mem, 100*(start_mem - end_mem) / start_mem))
    return df

#%% Correlate features
def correlate(dfi, target_col, cat_cols=[], exclude_cols=[], 
              calc_corr_sigs=False, sig_lvl=0.05, save=False):
    
    df = dfi.copy()
    
    # Remove columns not meant to be correlated
    df = df.drop(exclude_cols, axis=1)
    
    # Print target correlation calcs
    correlations = df.corr()[target_col].sort_values()
    print('\n---Correlation Anlaysis---')
    corr_var = 5 # num of significant corr's to display
    print('\nPearson Correlation Matrix:\n', df.corr().round(3))
    print('\n{} Most Positive Correlated Features w/ Target:\n{}'\
          .format(corr_var, correlations.tail(corr_var)))
    print('\n{} Most Negative Correlated Features w/ Target:\n{}'\
          .format(corr_var, correlations.head(corr_var)))
        
    # Plot correlation matrix
    mp.myCorrelation2(df, save=save)
    
    # Calc Correlation Significance Levels 
    if calc_corr_sigs == True: 
        print('\n >>>Calculating sig of feature correlations w/ 2-tailed ...')
        corr_cols = [x for x in list(df.columns) \
                     if (x not in [target_col])] # exclude target
        sig_matrix = pd.DataFrame(index=corr_cols, columns=corr_cols)
        sigs = [] # for filling rows of matrix 1 by 1 in loop
        var_track = [] # for tracking sets of significant vars
        seen = set() # blank set for creating list of unique sets of vars later
        sig = sig_lvl # significance level
        sig_count = 0
        for var1 in corr_cols:
            for var2 in corr_cols:
                sig_val = pearsonr(df[var1],df[var2])[1] #(x,y)=(r,p-val)
                if sig_val <= sig and sig_val != 0:
                    sig_count += 1
                    var_track.append((var1,var2))
                sigs.append(sig_val)
            sig_matrix[var1] = sigs
            sigs = []
            print('[====> {} of {} complete]'\
                  .format(corr_cols.index(var1)+1,
                  len(corr_cols)))
        sig_perc = (sig_count/2.0) / ((len(sig_matrix)**2 - \
                                       len(sig_matrix))/2.0)
        unique_pairs = [t for t in var_track \
                        if tuple(sorted(t)) not in seen \
                        and not seen.add(tuple(sorted(t)))]
        print('\nSignificance Matrix:\n', sig_matrix)
        print('\nCorrelations sig at a {} sig level = {:0.0f} ({:.0%})'\
              .format(sig_lvl, sig_count/2.0, sig_perc))
        print('Significant correlation pairs:')
        for pair in unique_pairs: print('   {}'.format(pair))
    return

#%% Outlier Detector
def detect_outliers(df, outlier_col, new_col, detect_type, by_col=None,
                    by_vals=[]):
    
    # Suppress false+ SettingWithCopyWarning (default='warn')
    pd.options.mode.chained_assignment = None  
    
    modZscore = 3 # changed from 3.5 for more sensitivity
    
    if detect_type == 'MAD':
        if by_col == None: # detect outliers against entire column
            
            Ymedian = df[outlier_col].median()
            MAD = df[outlier_col].agg('mad') 
            df[new_col] = df[outlier_col].apply( \
                          lambda Y: 'Outlier' \
                          if modZscore < abs(0.6745 * (Y - Ymedian) / MAD) \
                          else 'Normal')
            
        else: # detect outliers by subset of data
            for val in by_vals:
                if new_col in df: # code prev run, so don't overwrite old vals
                    df['tmp'] = df[new_col] # make copy of data in new col
                    
                Ymedian = df[outlier_col][df[by_col]==val].median()
                MAD = df[outlier_col][df[by_col]==val].agg('mad') 
                df['tmp'] = df[outlier_col][df[by_col]==val].apply( 
                            lambda Y: 'Outlier_'+by_col+'-'+str(val) 
                            if modZscore < abs(0.6745 * (Y - Ymedian) / MAD) 
                            else 'Normal_'+by_col+'-'+str(val))
                
                if new_col in df:
                    df[new_col] = df[new_col].fillna(df['tmp']) # fill new
                else: # this is first run
                    df[new_col] = df['tmp'] 
                df.drop(['tmp'], axis=1, inplace=True)
        
    if detect_type == 'Quartile': # detect outliers against entire column
        if by_col == None: # detect outliers against entire column
            
            first_quartile = df[outlier_col].describe()['25%'] #1 st qtl
            third_quartile = df[outlier_col].describe()['75%'] # 3rd qtl
            iqr = third_quartile - first_quartile # interquartile range
            df[new_col] = df[outlier_col].apply( \
              lambda Y: 'Normal' \
              if (Y > (first_quartile - modZscore * iqr)) & \
                  (Y < (third_quartile + modZscore * iqr)) \
              else 'Outlier')
            
        else: # detect outliers by subset of data
            for val in by_vals:
                if new_col in df: # code prev run, so don't overwrite old vals
                    df['tmp'] = df[new_col] # make copy of data in new col
                    
                first_quartile = df[outlier_col][df[by_col]==val].describe() \
                                ['25%'] #1 st qtl
                third_quartile = df[outlier_col][df[by_col]==val].describe() \
                                ['75%'] # 3rd qtl
                iqr = third_quartile - first_quartile # interquartile range
                df[new_col] = df[outlier_col][df[by_col]==val].apply( \
                  lambda Y: 'Normal_'+by_col+'-'+str(val)  \
                  if (Y > (first_quartile - modZscore * iqr)) & \
                      (Y < (third_quartile + modZscore * iqr)) \
                  else 'Outlier_'+by_col+'-'+str(val) )
                
                if new_col in df:
                    df[new_col] = df[new_col].fillna(df['tmp']) # fill new
                else: # this is first run
                    df[new_col] = df['tmp'] 
                df.drop(['tmp'], axis=1, inplace=True)
        
#%% Aggregate numeric data by fields
def aggregate_by_field(dfi, by_cols, agg_cols, agg_type):
    df = dfi.groupby(by_cols)[agg_cols].agg(agg_type).reset_index()
    df.columns = [''.join(x) if x[1] == '' 
                  else '_'.join(x) for x in df.columns.ravel()]
    return df

#%% Natural Log Field Transformer
def log_transform_numeric(df, numeric_cols, prefix=True):
    for col in numeric_cols:
        if df[numeric_cols].min().min() <= 0: #avoid neg values passed to log
            # Bring min val of all cols undergoing transform to 1 for ln()
            if prefix==True:
                df['log_'+col] = df[col] + abs(df[numeric_cols].min().min())+1
            else: # reset min>0 in place
                df[col] = df[col] + abs(df[numeric_cols].min().min())+1
        
        # Transform using natural log
        if prefix==True:
            df['log_'+col] = np.log(df['log_'+col])
        else: # transform in-place
            df[col] = np.log(df[col])

#%% Export Dataframe to csv
def export_df(df, path, df_name, suffix=''):
    df.to_csv(path + df_name + suffix + '.csv', index=False)
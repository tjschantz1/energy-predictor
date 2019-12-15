import myfuncs as mf #home-grown

import time
from datetime import datetime
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cbook import boxplot_stats
from mpldatacursor import datacursor

# Set default size of figures
fsize_x = 10
fsize_y = 8

# Set default size of fonts
small_font = 10
medium_font = 12
large_font = 14

plt.rc('font', size=small_font) # controls default text sizes
plt.rc('axes', titlesize=small_font) # fontsize of the axes title
plt.rc('axes', labelsize=medium_font) # fontsize of the x and y labels
plt.rc('xtick', labelsize=small_font) # fontsize of the tick labels
plt.rc('ytick', labelsize=small_font) # fontsize of the tick labels
plt.rc('legend', fontsize=small_font) # legend fontsize
plt.rc('figure', titlesize=large_font) # fontsize of the figure title

#%%Timer Function
def timer(start,end):
   hours, rem = divmod(end-start, 3600)
   minutes, seconds = divmod(rem, 60)
   return('{:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds))

#%% Human format number generator
def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

#%% Save high-res file
def save_high_res(prefix):
    export_path  = mf.get_path('auto_exports')
    suff = datetime.now().strftime("%y%m%d.%H.%M.%S")
    plt.savefig(export_path + prefix + suff + '.png', dpi=300)
#    print('\nPlot saved to {}'.format(export_path))

#%% Bar Plot
def myBar(dfi, xColN, yCol1N, pTitle=None, pSubTitle=None, xColLab=None, 
          yCol2N=None, yColLab=None, legendL=None, barWidth=1, 
          stackedBool=None, y2ColN=None, y2ColLab=None, y2ColLeg=None, 
          data_labels=None, save=False):
    fig, ax = plt.subplots(figsize=(fsize_x,fsize_y))
    xVar = np.arange(len(dfi[xColN])) # for stacked vs. side-by-side below
    plt.bar(xVar, dfi[yCol1N], color='#63646a', edgecolor='white', 
            width=barWidth) #63646a dark-gray
    if yCol2N != None:
        if stackedBool == False:
            xVar = xVar + barWidth # allows side-by-side bar
        plt.bar(xVar, dfi[yCol2N], color='#d9d9d9', edgecolor='white',
                width=barWidth) #d9d9d9 light-gray
    
    # Add Line to 2nd Axis
    if y2ColN is not None:
        ax2 = plt.twinx()
        ax2.plot(dfi[xColN], dfi[y2ColN], color='#9e2e62', 
                 label=y2ColLab) #9e2e62 dark purple
        ax2.set_ylabel(y2ColLab, fontsize=16, rotation=270, labelpad=15)
        ax2.legend([y2ColLeg], loc=2) 

    # Add data labels
    if data_labels != None:
        for x,y in zip(xVar, dfi[yCol1N]):
            label = human_format(y)
            plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center') # horizontal align can be left, right or center

    plt.xlabel(xColLab, fontsize=16)
    plt.ylabel(yColLab, fontsize=16)
    plt.suptitle(pTitle, y=.96, horizontalalignment='center', fontsize=20, 
                 fontweight='bold')
    plt.title(pSubTitle, horizontalalignment='center', fontsize=15, 
              style='italic')
    plt.xticks(range(dfi[xColN].count()), dfi[xColN]) # show all x-value labels
    if legendL!=None: plt.legend(legendL, loc=1)
    plt.xlim([-0.5, dfi[xColN].size - 0.5]) # remove white space on left/right
    fig.tight_layout(rect=[0, 0, 1, 0.95]) # keeps from pushing text offscreen
    if save==True: save_high_res(prefix='bar_')
    return plt.show()

#%% Matrix Series Plot by val of categorical column
def matrix_series(df, by_cols, time_col, filter_col=None, filter_val=None,
                  by_vals=False, main_title='Time series', save=False):
    
    if filter_val!=None: # filter df before plotting
        df = df.loc[df[filter_col]==filter_val]
    
    if by_vals==True and filter_col!=None: # plot unstackeds data by filter_col
        new_df = pd.DataFrame(df[time_col].unique(), 
                              columns=[time_col]) #create df w/ timestamp asref
        vals = sorted(list(df[filter_col].unique()))
        for val in vals:
            add_df = df[[time_col, filter_col, by_cols[0]]]. \
                     loc[df[filter_col]==val]
            add_df.drop([filter_col], axis=1, inplace=True)
            new_df = pd.merge(new_df, add_df, how='left', left_on=time_col,
                              right_on=time_col)
            if is_numeric_dtype(val): suffix = str(int(val))
            new_df.columns.values[vals.index(val)+1] = \
                by_cols[0]+'_'+filter_col+'_'+suffix
        by_cols = list(new_df.iloc[:,1:])
        df = new_df.copy()
            
    
    # Sort by_cat_col vals alphabetically for prettier printing
    if by_vals==False: by_cols = sorted(list(by_cols))
    
    na_dict = {} # for capturing vol nan in for loop below
    plt.rcParams.update({'font.size': 6})
    num_vals = len(by_cols)
    def round_up(n, decimals=0):
        import math
        multiplier = 10 ** decimals 
        return int(math.ceil(n * multiplier) / multiplier)
    cols = round_up(np.sqrt(num_vals),0)
    rows = round_up((num_vals/cols),0)
    fig = plt.figure(figsize=(fsize_x,fsize_y))
    gs = plt.GridSpec(rows, cols)
    for col, n in zip(by_cols, range(num_vals)):
        
        # Count nan by val
        num_na = df[col].isnull().sum()
        perc_na = num_na / df[col].shape[0]
        na_dict.update({col:[num_na, perc_na]})
        
        # Create matrix of subplots & add
        i = n%cols # return mod
        j = n//cols # return int
        plt.subplot(gs[j,i])
        plt.plot(df[time_col], df[col])
        plt.yticks(fontsize=8) # #overrides params.update
        if j+1 == cols: # apply xtick labels to last row of matrix
            plt.xticks(rotation=45, fontsize=8) #, fontweight='semibold')
        else: # remove xtick label for all others
            plt.tick_params(labelbottom=False)
        plt.title(col, fontsize=10) #fontweight='semibold')
    
    # Print final plot to screen
    plt.tight_layout()
    fig.subplots_adjust(top=0.88) #create space for main title
    if filter_val!=None and main_title=='Time series':
        main_title = main_title+' (by '+str(filter_val)+')'
    fig.suptitle(main_title, fontsize=16)
    if save==True: save_high_res(prefix='matrix_series_')
    plt.show()
    
    # Print missing table by val to screen
    missing = pd.DataFrame.from_dict(na_dict, orient='index',
                                     columns=['volume', 'perc_row_total'])
    missing['perc_col_total'] = missing.iloc[:,0] / missing.iloc[:,0].sum()
    missing.volume = ['{:,}'.format(x) for x in missing.volume]
    missing.perc_row_total = ['{:.1%}'.format(x) \
                             for x in missing.perc_row_total]
    missing.perc_col_total = ['{:.1%}'.format(x) \
                                 for x in missing.perc_col_total]
    missing.sort_index(inplace=True)
    print('\nMissing data:\n', missing)

#%% Line Charts
def myLine(dfi, pTitle, xColN, xColLab, yColN, yColLab, goalL, trendLbool,
           save=False):
    df = dfi.dropna(subset = [xColN, yColN])
    with sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(fsize_x,fsize_y))
        ax = sns.lineplot(x=xColN, y=yColN, data=df, color="g", label=yColLab)
        plt.title(pTitle, fontsize=20, fontweight='bold')
        ax.set(xlabel=xColLab, ylabel=yColLab)
        fig.tight_layout(rect=[0, 0, 1, 0.95]) # keeps from pushing txt offscrn
    
    # Calc & plot linear trend line for greater context
    if trendLbool == True:
        z = np.polyfit(df[xColN], df[yColN], 1)
        p = np.poly1d(z)
        plt.plot(df[xColN],p(df[xColN]),"k", linewidth=.15)
    
    #Add goal line, if available
    if goalL is not None:
        plt.axhline(y=goalL, linewidth=1.5, linestyle='dashed', color='r')
        ax.text(-.5, goalL, 'Goal<'+str(goalL), bbox=dict(facecolor='red', 
                alpha=0.5), weight='bold', size='small', color='w')
    # ax.tick_params(labelsize='small' ,color=fColor)
    plt.xticks(df[xColN]) # show all x-value labels on x-axis
    fig.tight_layout(rect=[0, 0, 1, 0.95]) 
    
    if save==True: save_high_res(prefix='line_')
    return plt.show()

def mySeries(dfi, xColN, yColN, pTitle=None, xColLab=None, yColLab=None,
             save=False):
    df = dfi.dropna(subset = [xColN, yColN])
    fig = plt.figure(figsize=(fsize_x,fsize_y))
    # ax = plt.plot(dfi[yColN])
    ax = sns.lineplot(x=range(len(df[xColN])), y=yColN, data=df)
    plt.title(pTitle, fontsize=20, fontweight='bold')
    plt.xlabel(xColLab, fontweight='bold')
    plt.ylabel(yColLab, fontweight='bold')
    plt.xticks(range(len(df[xColN])), df[xColN]) # allows non-numeric x-ax rng
    ax.xaxis.grid(color='grey', linestyle=':', linewidth=.5) # vertical lines
    fig.tight_layout(rect=[0, 0, 1, 0.95]) # keeps from pushing text offscreen
        #rect[] allows for suptitle
    if save==True: save_high_res(prefix='series_')
    return plt.show()

#%% Scatter Plot
def myScatt(dfi, xColN, yColN, pTitle=None, pSubTitle=None, xColLab=None,
            yColLab=None, grouper=None, goalL=None, extraTxt=None, 
            outliers_col=None, trendLbool=False, save=False):
    
    start_time = time.time()
    
    df = dfi.dropna(subset = [xColN, yColN])
    
    if outliers_col != None:
        df = df[df[outliers_col] != 'Outlier']  # remove extreme outliers
    
    with sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(fsize_x,fsize_y))
        ax = sns.scatterplot(x=xColN, y=yColN, data=df, hue=grouper, 
                             label=yColLab)
        # plt.title(pTitle, fontsize=20, fontweight='bold')
        plt.suptitle(pTitle, y=.96, horizontalalignment='center', fontsize=25, 
                     fontweight='bold')
        plt.title(pSubTitle, horizontalalignment='center', fontsize=20, 
                  style='italic')
        ax.set(xlabel=xColLab, ylabel=yColLab)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    xlocs, xlabels = plt.xticks()
    ylocs, ylabels = plt.yticks()
    
    #Add goal line, if available
    if goalL is not None:
        plt.axhline(y=goalL, linewidth=1.5, linestyle='dashed', color='r')
        if xlocs[0] == 0:
            txtBoxPlacement = abs(xlocs[1])* -.25 
        else:
            txtBoxPlacement = abs(xlocs[0])* -.25
        ax.text(txtBoxPlacement, goalL, 'Goal<'+str(goalL), 
                bbox=dict(facecolor='red', alpha=0.5), weight='bold', 
                size='small', color='w')
    # ax.tick_params(labelsize='small' ,color=fColor)
    
    #Add extra text, if available
    if extraTxt is not None:
        ax.text(xlocs[-2], 0, extraTxt, fontsize=12, 
                horizontalalignment='right')
    
    # Calc & plot linear trend line for greater context
    if trendLbool == True:
        
        # Poly Fit 
        from scipy.optimize import curve_fit
        try:
            x_new = np.linspace(df[xColN].values.min(), 
                                df[xColN].values.max(), 50)
            model = lambda x, A, x0, sigma, offset:  \
                                offset+A*np.exp(-((x-x0)/sigma)**2)
            popt, pcov = curve_fit(model, df[xColN].values, 
                                df[yColN].values, p0=[1,0,2,0])
            plt.plot(x_new,model(x_new,*popt), 'k', label='Fit') # k = black
        except:
            pass
            
        plt.legend()
    
    if save==True: save_high_res(prefix='scatt_')
    
    return plt.show(), print('\nRuntime =', timer(start_time, time.time()))


#%% # Sorted Bar Plot
def mySortedBar(df, byColN, save=False): 
    if save==True: save_high_res(prefix='sortedbar_')
    return df.sort_values(byColN, 
                          ascending=False)[[byColN]].plot.bar(stacked=False) 
#mySortedBar(dfAggImg, 'ImgOrder-to-Finalizedmedian')
    
#%% Histogram
def myHist(dfi, xColN, ibins='auto', pTitle=None, pSubTitle=None, xColLab=None, 
           log_scale=False, extraTxt=None, outliers_col=None, save=False):

    df = dfi.dropna(subset = [xColN]) 
    
    if outliers_col != None:
#        df = df[df[outliers_col] != 'Outlier'] # more dynamic option below
        df = df.loc[df[outliers_col].str.contains('Normal', na=True)]
        
    
    fig = plt.figure(figsize=(fsize_x,fsize_y))
    
    n, bins, patches = plt.hist(x=df[xColN], bins=ibins, color='#9e2e62', 
                                edgecolor='white', log=log_scale)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(xColLab, fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.suptitle(pTitle, y=.96, horizontalalignment='center', fontsize=23, 
                 fontweight='bold')
    plt.title(pSubTitle, horizontalalignment='center', fontsize=18, 
              style='italic')
    plt.text(23, 45, r'$\mu=15, b=3$')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    xlocs, xlabels = plt.xticks()
    ylocs, ylabels = plt.yticks()
    #Add extra text, if available
    if extraTxt is not None:
        plt.text(xlocs[-2], ylocs[-2], extraTxt, fontsize=12, 
                 horizontalalignment='right')   
    
    if save==True: save_high_res(prefix='histo_')
    return plt.show()

def my_distribution(data, ibins=None, pTitle='', save=False):
    fig = plt.figure(figsize=(fsize_x,fsize_y))
    ax = sns.distplot(data, bins=ibins)
    plt.title(pTitle)
    if save==True: save_high_res(prefix='distrib_')
    return plt.show
    
#%% Density Plot
def myDens(dfi, xColN, pTitle=None, cat_col=None, cat_labels=[], xColLab=None, 
           legendLab=None, extraTxt=None, outliers_col=None, save=False):
    '''
    cat_labels is list of categorical variables to plot; cat_col is where to  
    find those categs in df; user can choose to plot none, some, or all below
    '''
    fig = plt.figure(figsize=(fsize_x,fsize_y))
    orig_palette = sns.color_palette() # capture for returning to later
    try: #not to leave color palette default changed on failure
        sns.set_palette('husl') #(sns.cubehelix_palette(8, start=.5, rot=-.75))
        
        if outliers_col != None:
            df = dfi[dfi['OutlierDetect'] != 'Outlier'] 
        else: df = dfi.copy()
        
        # Iterate through the list & create new df for each density plot
        if cat_col != None and cat_labels == []: \
                        cat_labels = df[cat_col].unique()
        if cat_labels != []:
            for lab in cat_labels:
                dfSub = df[[cat_col, xColN]][df[cat_col] == lab]
            
                # Draw the density plot
                sns.distplot(dfSub[xColN], hist = False, kde = True,
                            kde_kws = {'linewidth': 3},
                            label = lab)
        else:
            # Draw the density plot
            sns.distplot(dfSub[xColN], hist = False, kde = True,
                        kde_kws = {'linewidth': 3},
                        label = lab)
        
        # Plot formatting
        plt.legend(prop={'size': 12}, title = legendLab)
        plt.title(pTitle, horizontalalignment='center')
        plt.xlabel(xColLab)
        plt.ylabel('Density')
        
        xlocs, xlabels = plt.xticks()
        ylocs, ylabels = plt.yticks()
        #Add extra text, if available
        if extraTxt is not None:
            plt.text(xlocs[2], ylocs[-2], extraTxt, fontsize=12, 
                     horizontalalignment='right')   
        
        fig.tight_layout(rect=[0, 0, 1, 0.95]) # prevents pushing txt offscreen
    except: next
    sns.set_palette(orig_palette)
    if save==True: save_high_res(prefix='dens_')
    return plt.show()

#%% Correlation Matrix
def myCorrelation(dfi, save=False):
    fig = plt.figure(figsize=(fsize_x,fsize_y))
    plt.matshow(dfi.corr(), fignum=fig.number, aspect='equal')
    plt.xticks(range(dfi.shape[1]), dfi.columns, fontsize=8, rotation=45)
    plt.yticks(range(dfi.shape[1]), dfi.columns, fontsize=8, rotation=45)
    plt.gca().xaxis.tick_bottom() # put x-axis ticks at bottom
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=8)
    plt.title('Correlation Matrix', fontsize=16, loc='center')
    if save==True: save_high_res(prefix='corr_')
    return plt.show()

def myCorrelation2(df, save=False):
#    sns.set(style="white")
    corr = df.corr() # compute the correlation matrix
    
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    
    # Add the middle diagonol to the mask
    mask_shape = len(mask[0]) # always nxn
    for i, j in zip(range(mask_shape), range(mask_shape)):
        if i == j:
            mask[i][j] = not mask[i][j]
    
    fig, ax = plt.subplots(figsize=(fsize_x, fsize_y))
    
    # Generate a custom diverging colormap
#    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    cmap="YlGnBu"
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={'shrink': .9},
                annot=True) #, annot_kws={'size': 'medium'})
    
    plt.xticks(range(df.shape[1]+1), df.columns, fontsize=10)#, rotation=45)
    plt.yticks(range(df.shape[1]+1), df.columns, fontsize=10)#, rotation=45)
    if save==True: save_high_res(prefix='corr_')
    plt.title('Correlation Matrix', fontsize=16, loc='center')
    plt.tight_layout()
#    fig.subplots_adjust(bottom=0.05) #create space for main title
    return plt.show()

#%% Plot neural network model loss
def plot_loss(training_history, loss_type, save=False):
    fig = plt.figure(figsize=(fsize_x,fsize_y))
    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'], 'g--')
    plt.title('Model Loss')
    plt.ylabel(loss_type)
    plt.xlabel('Epoch')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')
    print('  >> Loss after final iteration: ', 
          training_history.history['val_loss'][-1])
    if 'val_accuracy' in training_history.history.keys(): #classification only
        print('  >> Accuracy after final iteration: {:.2%}'.
              format(training_history.history['val_accuracy'][-1]))
    plt.show()

# Plot gradient boosted regression loss 
def plot_gbr_loss(train_loss, val_loss, n_estimators, loss_type='Error', 
                  plot_title='Model Loss', save=False):
    fig = plt.figure(figsize=(fsize_x,fsize_y))
    plt.plot(np.arange(n_estimators) + 1, train_loss, '-b', 
             label='training_loss')
    plt.plot(np.arange(n_estimators) + 1, val_loss, 'green', 
             linestyle='dashed',
             label='val_loss')
    plt.title(plot_title)
    plt.ylabel(loss_type)
    plt.xlabel('n_estimators')
    plt.legend(loc='upper right')
    
    if save==True: save_high_res(prefix='gbr_loss_')
    return plt.show()

#%% Plot predicted vs. true
def plot_pred_vs_true(y_true, y_pred, pTitle='', save=False):
    
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(figsize=(fsize_x, fsize_y))
        ax = sns.scatterplot(x=y_true, y=y_pred)
        plt.title(pTitle, fontsize=16) #, fontweight='bold')
        
        # Plot diagonol
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             'k--', lw=1)
    xlocs, xlabels = plt.xticks(fontsize=9)
    ylocs, ylabels = plt.yticks(fontsize=9)
    
    ax.set_xlabel('True', fontsize=11)
    ax.set_ylabel('Predicted', fontsize=11)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88) #create space for main title
    
    if save==True: save_high_res(prefix='predvstrue_')
    plt.show()

#%% Plot Learning Curve
from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5, n_jobs=None, 
                        train_sizes=np.linspace(.1, 1.0, 5), save=False):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    
    # RMSLE
    plt.ylabel('RMSLE')
    from sklearn.metrics import mean_squared_log_error # RMSLE
    from sklearn.metrics.scorer import make_scorer
    def rmsle(y_true, y_pred):
        y_pred[y_pred<0]=0 # move values<0 to 0
        return np.sqrt(mean_squared_log_error(y_true, y_pred))
    my_scorer = make_scorer(rmsle, greater_is_better=True) #from mymodels
    
    '''
    Warning: learning_curve() will call the fit() function for each of the 5
    x-ticks set in arg "train_sizes=np.linspace(.1, 1.0, 5)" for each model
    objective (i.e. estimator) passed x2 (1 for train & 1 for cv line), so 
    a total of 10 fits() called on different subsections of data,
    so if there are training iterations within the model passed (e.g. 
    GradientBoostingRegressor(max_depth=100)), it'll be 100x10 training
    operations called which can be EXTREMELY time intensive.
    '''
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=my_scorer) #not RMSLE, closer than R2 default
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    if save==True: save_high_res(prefix='learncurve_')
    plt.show()
    

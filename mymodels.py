import myfuncs as mf #home-grown
import myplotters as mp #home-grown

import numpy as np
import pandas as pd
import time

# Machine Learning Modeling
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, \
                             GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #disbles AVX/FMA Tensorflow warn for CPU

# Model export/import
import pickle # for saving/loading models
from tensorflow.keras.models import model_from_json # for loading MLP models

# Model Performance Measuring
from sklearn.metrics import mean_squared_log_error # RMSLE
from sklearn.metrics import classification_report, \
                            accuracy_score, confusion_matrix
                            

# Hyperparameter tuning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dropout, Dense, Activation, \
                         MaxPooling2D, BatchNormalization, Input, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import model_from_json, Model
from tensorflow.keras.optimizers import Adam

#%% Model Object
class MyModel:
        
    def __init__(self, X_trainable=[], X_train=pd.DataFrame(),
                 X_val=pd.DataFrame(), X_test=pd.DataFrame(),
                 y_train=pd.Series(), y_val=pd.Series(), log_y=False):
        self.X_trainable = X_trainable # training feature field identification
        self.X_train = X_train[X_trainable] # training feature data
        self.X_val = X_val[X_trainable] # validation feature data
        self.X_test = X_test # for calling model on test data
        self.y_train = y_train.ravel() # training target data
        self.y_val = y_val.ravel() # validation/test target data
        self.log_y = log_y # default=False to keep target val as-is
        self.model_export_path = mf.get_path('models')
        self.models = {} # for cataloging trained models
        self.predictions = {} # for cataloging model predictions
        self.performance = {} # for cataloging model performance
        self.manual_params = {} # for cataloging manually called parameters
        
    # Return string of model object's name for catalog indexing    
    def get_name(self, model):
        model_name = [k for k,v in self.models.items() if v == model][0]
        return model_name
    
    # Print model parameters
    def print_params(self, model, print_all_params=False):
        if print_all_params==True:
            print('\nAll model parameters passed, including defaults:')
            print('  {}\n'.format(model)) # prints all args, defaults included
        else: # print only user-called args, as defined below
            print('\nModel parameters passed manually:')
            params = self.manual_params[self.get_name(model)] #call from catalog
            for k,v in params.items():
                if k!='': print('  {} = {}'.format(k,v))
                else: print('  None. All defaults employed.\n')
    
    # Fit model & evaluate
    def fit(self, model, file_name=None, epochs=0, batch_size=0):
        print('\n[>>>Training {} model...]'.format(file_name))
        start_time = time.time()
        X_train, X_val = self.X_train, \
                         self.X_val
        y_train, y_val = self.y_train, self.y_val
        if epochs==0: # not a MLP model
            model.fit(X_train, y_train)
            training_history = None
        else: # MLP model
            training_history = model.fit(X_train, y_train,
                                         batch_size=batch_size,
                                         epochs=epochs,
                                         validation_data=(X_val, y_val),
                                         use_multiprocessing=False)
        print('--- Training Complete. Runtime =', \
              mf.timer(start_time, time.time()),'---\n')
        return training_history # used in MLP plot of epoch history
    
    # Predict target & catalog
    def predict(self, model, X_set):
        predictions = model.predict(X_set) # predict
        
        # Reverse the log transform, if called
        if self.log_y == True:
            predictions = np.expm1(predictions)
        
        # Check for negatives, format, & catalog for use later
        neg_predictions = np.sum(np.array(predictions) < 0, axis=0)
        perc_neg_predictions = neg_predictions / float(len(predictions))
        if not np.isscalar(neg_predictions): # Keras results in array
            neg_predictions=neg_predictions[0] # set to float val
            perc_neg_predictions=perc_neg_predictions[0] # set to float val
        print('Moving {:,} ({:.1%}) predictions to 0'.format(neg_predictions,
                                                    perc_neg_predictions))
        predictions[predictions<0]=0 # move values<0 to 0
        predictions = np.around(predictions, decimals=4) #per Kaggle instruct
        if isinstance(predictions[0], list): # check dimension of list
            predictions = pd.Series([x[0] for x in predictions]) #ensure series
        self.predictions.update({self.get_name(model) : predictions}) # catalog
    
    # Function for printing tensorflow objects, when needed
    def get_tf_val(self, tf_obj):
        #initialize the variable
        init_op = tf.global_variables_initializer()
    
        #obtain var wihtin tf object
        with tf.Session() as sess:
            sess.run(init_op) #execute init_op
            return (sess.run(tf_obj))
    
    # Establish performance metric function for final eval
    def rmsle_eval(self, y_true, y_pred): # using tensorflow backend
        ''''
        The below calculation using scikitlearn's mean_squared_log_error
        method should work universally across all model types, but will not
        work as a custom loss function inside tensorflow's backend. 
        See rmsle_loss() for solution.
        '''
        # Reverse the log transform, if called
        if self.log_y == True:
            y_true = np.expm1(y_true) 
            y_pred = np.expm1(y_pred)

        return np.sqrt(mean_squared_log_error(y_true, y_pred))

    
    # Establish performance metric loss function (used in loss for MLP)
    def rmsle_loss(self, y_true, y_pred): # using tensorflow backend
        '''
        Designed to calc custom loss function inside keras model fit
        '''
        
        # Prep for rmsle loss value calc
        y_pred = tf.cast(y_pred>0, y_pred.dtype)*y_pred+1 # move values<0 to 1
        y_pred = tf.cast(y_pred, tf.float64)
        y_true = tf.cast(y_true, tf.float64)
        y_pred = tf.nn.relu(y_pred)
        
        # Reverse the log transform, if called
        if self.log_y == True:
            y_pred = tf.math.expm1(y_pred)
            y_true = tf.math.expm1(y_pred)
        
        return tf.sqrt(tf.reduce_mean( \
                       tf.math.squared_difference(tf.math.log1p(y_pred),
                                                  tf.math.log1p(y_true))))
    
    # Evaluate predictions & catalog
    def eval(self, model):
        rmsle_val = self.rmsle_eval(y_true=self.y_val,
                                    y_pred=self.predictions[ \
                                                        self.get_name(model)])
        print('Validation RMSLE: {:.3f}'.format(rmsle_val))
        self.performance.update({self.get_name(model) : rmsle_val}) # catalog
    
    # Save model for use later
    def export_model(self, model, model_type, file_name):
        
        # Non-MLP models
        if model_type != 'mlp':
            with open(self.model_export_path + file_name + '.dat', 'wb') as f:
                pickle.dump(model, f)
        
        # MLP model
        if model_type == 'mlp':
            
            # Serialize model to JSON
            model_json = model.to_json()
            with open(self.model_export_path + file_name + '.json', 'w') \
                as json_file:
                json_file.write(model_json)
            
            # Serialize weights to HDF5
            model.save_weights(self.model_export_path + file_name + '.h5')
    
    # Load pre-trained model for testing
    def load_model(self, model_type, file_name):

        # Non-MLP models
        if model_type != 'mlp':
            with open(self.model_export_path + file_name + '.dat', 'rb') as f:
                loaded_model = pickle.load(f)
        
        # MLP model
        elif model_type == 'mlp':
            f_path = self.model_export_path + file_name
            
            # Create model from json & load in weights
            with open(f_path + '.json', 'r') as f:
                loaded_model = model_from_json(f.read())
                loaded_model.load_weights(f_path + '.h5')
        
        # Catalog loaded model (note: this will override k,v if dupe)
        self.models.update({model_type:loaded_model})
            
        return loaded_model

    #%% Model 01: Linear Regression
    def lr(self, file_name=None, print_all_params=False):
        
        # Set model parameters
        p = {'':''}
        self.manual_params.update({'lr':p})
        
        # Instantiate & train model
        lr = LinearRegression()
        self.fit(lr, file_name)
        self.models.update({'lr':lr}) # for calling later
        '''
        Warning: Time to complete below is ~10x time to complete 1 fit based on
        params above. See notes in myplotters for more details.
        '''
        mp.plot_learning_curve(lr, 'Learning Rate',
                               X=self.X_train,
                               y=self.y_train)
        self.predict(lr, self.X_val)
        self.eval(lr)
        if file_name!=None: self.export_model(lr, 'lr', file_name)
        self.print_params(lr, print_all_params)

    #%% Model 02: K-Nearest Neighbors
    def knn(self, file_name=None, print_all_params=False):
        
        # Set model parameters
        p = {'n_neighbors':10}
        self.manual_params.update({'knn':p})
        
        # Instantiate & train model
        knn = KNeighborsRegressor(n_neighbors=p['n_neighbors'])
        self.fit(knn, file_name)
        self.models.update({'knn':knn}) # for calling later
        self.predict(knn, self.X_val)
        self.eval(knn)
        if file_name!=None: self.export_model(knn, 'knn', file_name)
        self.print_params(knn, print_all_params)
        
    #%% Model 03: Random Forest
    def rf(self, file_name=None, print_all_params=False):
        
        # Set model parameters
        p = {'n_estimators':50, 'max_depth':10, 'random_state':6}
        self.manual_params.update({'rf':p}) #100 estimators prfmd worse
        
        # Instantiate & train model
        rf = RandomForestRegressor(n_estimators=p['n_estimators'],
                                   max_depth=p['max_depth'],
                                   random_state=p['random_state'],
                                   verbose=1)
                                    #max_features=np.sqrt(len(self.X_trainable)),
        self.fit(rf, file_name)
        self.models.update({'rf':rf}) # for calling later
        self.predict(rf, self.X_val)
        self.eval(rf)
        if file_name!=None: self.export_model(rf, 'rf', file_name)
        self.print_params(rf, print_all_params)
        
    #%% Model 04: Gradient Boosted Regression
    def gbr(self, file_name=None, print_all_params=False):
        
        # Set model parameters (LAD=Least Absolute Deviation=L1)
        p = {'loss':'lad', 'max_depth':10, 'max_features':None,
             'min_samples_leaf':6, 'min_samples_split':6,
             'n_estimators':100, 'tol':0.01}
        self.manual_params.update({'gbr':p})
        
        # Instantiate & train model
        gbr = GradientBoostingRegressor(loss=p['loss'],
                                        max_depth = p['max_depth'],
                                        max_features=p['max_features'],
                                        min_samples_leaf = p['min_samples_leaf'],
                                        min_samples_split= \
                                                    p['min_samples_split'],
                                        n_estimators=p['n_estimators'],
                                        tol=p['tol'],
                                        verbose=1) #instanstiate
        self.fit(gbr, file_name)
        self.models.update({'gbr':gbr}) # for calling later
        if file_name!=None: self.export_model(gbr, 'gbr', file_name)      
        self.predict(gbr, self.X_val)
        
        # Obtain validation loss for plotting
        val_score = np.zeros((p['n_estimators'],), dtype=np.float64)
        for i, pred in enumerate(gbr.staged_predict(self.X_val)):
            val_score[i] = gbr.loss_(self.y_val, pred.reshape(-1, 1))
        
        # Plot loss
        mp.plot_gbr_loss(gbr.train_score_, val_score, p['n_estimators'], 
                         loss_type=p['loss'], save=True) #, plot_title=
        self.eval(gbr)
        self.print_params(gbr, print_all_params)
        
    #%% Model 05: Support Vector Machine
    def svm(self, file_name=None, print_all_params=False):
        
        # Set model parameters
        p = {'C':100, 'gamma':0.1}
        self.manual_params.update({'svm':p})
        
        # Instantiate & train model
        svm = SVR(C = p['C'], gamma = p['gamma'], verbose=1)
        self.fit(svm, file_name)
        self.models.update({'svm':svm}) # for calling later
        if file_name!=None: self.export_model(svm, 'svm', file_name)
        self.predict(svm, self.X_val)
        self.eval(svm)
        self.print_params(svm, print_all_params)
        
    #%% Model 06: Multi_Layer Perceptron Neural Network
    def mlp(self, file_name=None, print_all_params=False):
        
        # Define multi-input MLP network
        def create_mlp(dim):
            model = Sequential()
            model.add(Dense(8, input_dim=dim, activation='relu')) #8
            model.add(Dense(4, activation='relu')) #4
            model.add(Dense(1, activation='linear'))
            return model

        # Set model parameters
        p = {'Adam_lr':1e-3, 'Adam_decay':1e-3 / 200, 'optimizer':'adam', 
             'epochs':10, 'batch_size':10} # batch_size:8
        self.manual_params.update({'mlp':p})
        
        # Instantiate & compile model & train model
        mlp = create_mlp(self.X_train.shape[1])
        mlp.compile(loss=self.rmsle_loss, optimizer=p['optimizer'])
        print(mlp.summary())
        
        # Train model
        training_history = self.fit(mlp, file_name, epochs=p['epochs'], 
                                    batch_size=p['batch_size'])
        self.models.update({'mlp':mlp}) # for calling later
        if file_name!=None: self.export_model(mlp, 'mlp', file_name)
        mp.plot_loss(training_history, 'RMLSE')
        self.predict(mlp, self.X_val)
        self.eval(mlp)
        self.print_params(mlp, print_all_params)

    #%% Model 07: XGBoost
    def xgbr(self, file_name=None, print_all_params=False):

        # Set model parameters
        p = {'objective':'reg:squarederror', 'colsample_bytree':0.3,
             'learning_rate':0.1, 'max_depth':10, 'alpha':10,
             'n_estimators':100}
        self.manual_params.update({'xgbr':p})
        
        # Instantiate & train model
        xgbr = xgb.XGBRegressor(objective=p['objective'],
                                   colsample_bytree=p['colsample_bytree'],
                                   learning_rate=p['learning_rate'],
                                   max_depth=p['max_depth'],
                                   alpha=p['alpha'],
                                   n_estimators=p['n_estimators'],
                                   verbosity=1) #2=info; 1=warnings; 0=silent
        self.fit(xgbr, file_name)
        self.models.update({'xgbr':xgbr}) # for calling later
        if file_name!=None: self.export_model(xgbr, 'xgbr', file_name)

        self.predict(xgbr, self.X_val)
        self.eval(xgbr)
        self.print_params(xgbr, print_all_params)
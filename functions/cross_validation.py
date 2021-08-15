import  datetime
import pandas as pd
import numpy  as np

from sklearn.metrics import mean_absolute_error, mean_squared_error


def mean_absolute_percentage_error( y, yhat ):
    return np.mean( np.abs( y - yhat ) / y )
    
def ml_error( model_name, y, yhat ):
    mae  = mean_absolute_error( y, yhat )
    mape = mean_absolute_percentage_error( y, yhat )
    rmse = mean_squared_error( y, yhat, squared=False )
    
    return pd.DataFrame( {'Model Name': model_name, 
                          'MAE': mae,
                          'MAPE': mape,
                          'RMSE': rmse }, index=[0] )


def cross_validation( x_training, kfold, model_name, model, verbose=False ):
    mae_list = []
    mape_list = []
    rmse_list = []

    for k in reversed( range( kfold+1 ) ):
        if verbose:
            print(f'Kfold Number: {k}\n')

        # Start and end Date validation
        valid_start_date = x_training['date'].max() - datetime.timedelta( days=k*6*7 )
        valid_end_date = x_training['date'].max() - datetime.timedelta( days=(k - 1)*6*7)

        # Filtering Dataset
        training = x_training[x_training['date'] < valid_start_date]
        validation = x_training[(x_training['date'] >= valid_start_date) & (x_training['date'] <= valid_end_date)]

        # Training and Validation dataset
        xtrain = training.drop( ['sales', 'date'], axis=1 )
        ytrain = training['sales']

        xvalid = validation.drop(['sales', 'date'], axis=1)
        yvalid = validation['sales']

        # Model
        n = model.fit( xtrain, ytrain )

        # Prediction
        yhat = n.predict( xvalid )

        # Performace
        n_result = ml_error( model_name, np.expm1( yvalid ), np.expm1( yhat ) )

        # Storing the Performace of each Kfold iteration
        mae_list.append( n_result['MAE'] )
        mape_list.append( n_result['MAPE'] )
        rmse_list.append( n_result['RMSE'] )

    return pd.DataFrame( { 'Model Name': model_name,
                           '(CV) MAE Mean': np.round( np.mean( mae_list ), 2 ).astype('str') + ' +/- ' + np.round( np.std( mae_list ), 2 ).astype('str'), 
                           '(CV) MAPE Mean': np.round( np.mean( mape_list ), 2 ).astype('str') + ' +/- ' + np.round( np.std( mape_list ), 2 ).astype('str'), 
                           '(CV) RMSE Mean': np.round( np.mean( rmse_list ), 2 ).astype('str') + ' +/- ' + np.round( np.std( rmse_list ), 2 ).astype('str') }, index=[0] )
#import pickle
import joblib
import pandas as pd
from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann

# Loading Model
model = joblib.load(open( '/Users/T-Gamer/.AArep/Rossman-Store-Sales/saved_pickle/xgb_tunned.joblib', 'rb' ))
#model = pickle.load(open( '/Users/T-Gamer/.AArep/Rossman-Store-Sales/saved_pickle/xgb_tunned.pickle', encoding='utf-8', errors='ignore' ))


# Init API
app = Flask( __name__ )

@app.route( '/rossmann/predict', methods=['POST'] )
def rossmann_predict():
    test_json = request.get_json()
    
    if test_json: #Have Data?
        if isinstance( test_json, dict ): # Unique Data
            test_raw = pd.DataFrame( test_json, index=[0] )
            
        else: # Multiple Data
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
            
        # Instanciate rossmann class
        pipeline = Rossmann()
        
        # Data Cleaning
        df1 = pipeline.data_cleaning( test_raw )
        
        # Feature Engineering
        df2 = pipeline.feature_engineering( df1 )
        
        # Data Preparation
        df3 = pipeline.data_preparation( df2 )
        
        # Prediction
        df_res = pipeline.get_prediction( model, test_raw, df3 )
        
        return df_res
        
    else:
        return Response( '{}', status=200, mimetype='application/json' )

if __name__ == '__main__':
    app.run( '192.168.2.5' )
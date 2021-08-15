import math
import pickle
import datetime
import inflection
import pandas as pd
import numpy as np

class Rossmann( object ):
    def __init__( self ):
        self.home_path = '/Users/T-Gamer/.AArep/Rossman-Store-Sales/'
        self.competition_distance_scaler   = pickle.load( open( self.home_path + '/saved_pickle/competition_distance_scaler.pkl', 'rb' ) )
        self.competition_time_month_scaler = pickle.load( open( self.home_path + '/saved_pickle/competition_time_month_scaler.pkl', 'rb' ) )
        self.promo_time_week               = pickle.load( open( self.home_path + '/saved_pickle/promo_time_week_scaler.pkl', 'rb' ) )
        self.year_scaler                   = pickle.load( open( self.home_path + '/saved_pickle/year_scaler.pkl', 'rb' ) )
        self.store_type_scaler             = pickle.load( open( self.home_path + '/saved_pickle/store_type_scaler.pkl', 'rb' ) )

        
    def data_cleaning( self, df1 ):
        
        # Rename Columns
        old_cols = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval']

        snake_case = lambda x: inflection.underscore( x )

        new_cols = list( map( snake_case, old_cols ) )

        df1.columns = new_cols

        # Change Types
        df1['date'] = pd.to_datetime(df1['date'])

        df1['assortment'] = df1['assortment'].apply( lambda x: 'basic' if x == 'a' else 
                                           'extra' if x == 'b' else 'extended')

        df1['state_holiday'] = df1['state_holiday'].apply( lambda x: 'public_holiday' if x == 'a' else 
                                                  'easter_holiday' if x == 'b' else
                                                  'christmas' if x == 'c' else 'regular_day')

        # Fillout Na

        #competition_distance
        df1['competition_distance'] = df1['competition_distance'].apply( lambda x: 200000.0 if math.isnan( x ) else x )

        #competition_open_since_month
        df1['competition_open_since_month'] = df1.apply( lambda x: x['date'].month if math.isnan( x['competition_open_since_month'] )
                                                                    else x['competition_open_since_month'], axis=1 )

        #competition_open_since_year 
        df1['competition_open_since_year'] = df1.apply( lambda x: x['date'].year if math.isnan( x['competition_open_since_year'] )
                                                                   else x['competition_open_since_year'], axis=1 )

        #promo2_since_week
        df1['promo2_since_week'] = df1.apply( lambda x: x['date'].week if math.isnan( x['promo2_since_week'] )
                                                                   else x['promo2_since_week'], axis=1 )

        #promo2_since_year
        df1['promo2_since_year'] = df1.apply( lambda x: x['date'].year if math.isnan( x['promo2_since_year'] )
                                                                   else x['promo2_since_year'], axis=1 )

        #promo_interval  
        month_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

        df1['month_map'] = df1['date'].dt.month.map( month_map )

        df1['promo_interval'].fillna(0, inplace=True)

        df1['is_promo'] = df1[['promo_interval', 'month_map']].apply( lambda x: 0 if x['promo_interval'] == 0 
                                                                    else 1 if x['month_map'] in x['promo_interval'].split(',') 
                                                                    else 0, axis=1 )


        df1['competition_open_since_year']  = df1['competition_open_since_year'].astype('int64')
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype('int64')

        df1['promo2_since_week'] = df1['promo2_since_week'].astype('int64')
        df1['promo2_since_year'] = df1['promo2_since_year'].astype('int64')

        return df1
    
    
    def feature_engineering( self, df2 ):
        
        # Year
        df2['year'] = df2['date'].dt.year

        # Month
        df2['month'] = df2['date'].dt.month

        # Day
        df2['day'] = df2['date'].dt.day

        # Year Week
        df2['week_of_year'] = df2['date'].dt.isocalendar().week

        # Week of Year
        df2['year_week'] = df2['date'].dt.strftime( '%Y-%W' )

        # Competition Since
        df2['competition_since'] = df2.apply( lambda x: datetime.datetime( year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1 ), axis=1 )
        df2['competition_time_month'] = ( ( df2['date'] - df2['competition_since'] ) / 30 ).apply( lambda x: x.days ).astype('int64')

        # Promo Since
        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str)
        df2['promo_since'] = df2['promo_since'].apply( lambda x: datetime.datetime.strptime( x + '-1', '%Y-%W-%w') - datetime.timedelta( days= 7 ) )
        df2['promo_since_week'] = ( ( df2['date'] - df2['promo_since'] ) / 7 ).apply( lambda x: x.days ).astype('int64')

        df2 = df2[df2['open'] != 0]

        df2 = df2.drop( ['promo_interval', 'open', 'month_map'], axis=1)

        return df2

    def data_preparation( self, df5  ):

        # 6.2. Rescaling
        # Competition Distance
        df5['competition_distance'] = self.competition_distance_scaler.fit_transform( df5[['competition_distance']].values )

        # Year
        df5['year'] = self.year_scaler.fit_transform( df5[['year']].values )

        # Competition time month
        df5['competition_time_month'] = self.competition_time_month_scaler.fit_transform( df5[['competition_time_month']].values )

        # Promo time week
        df5['promo_since_week'] = self.promo_time_week.fit_transform( df5[['promo_since_week']].values )
    
        # State Holiday --> One  Hot Encoding
        df5['state_holiday'] = pd.get_dummies( df5, prefix=['state_holiday'], columns=['state_holiday'] )

        # Assortment --> Label Encoding
        assortment_dict = {'basic': 1, 'extra': 2, 'extended': 3}
        df5['assortment'] = df5['assortment'].map( assortment_dict )

        # Store Type --> Ordinal Encoding
        df5['store_type'] = self.store_type_scaler.fit_transform( df5['store_type'] )

        ### 6.3.3. Nature Transformation 
        # Month
        df5['month_sin'] = df5['month'].apply( lambda x: np.sin( x * ( 2. * np.pi / 12 ) ) )
        df5['month_cos'] = df5['month'].apply( lambda x: np.cos( x * ( 2. * np.pi / 12 ) ) )

        # Day
        df5['day_sin'] = df5['day'].apply( lambda x: np.sin( x * ( 2. * np.pi/30 ) ) )
        df5['day_cos'] = df5['day'].apply( lambda x: np.cos( x * ( 2. * np.pi/30 ) ) )

        # Day of Week
        df5['day_of_week_sin'] = df5['day_of_week'].apply( lambda x: np.sin( x * ( 2. * np.pi/7 ) ) )
        df5['day_of_week_cos'] = df5['day_of_week'].apply( lambda x: np.cos( x * ( 2. * np.pi/7 ) ) )

        # Week of Year
        df5['week_of_year_sin'] = df5['week_of_year'].apply( lambda x: np.sin( x * ( 2. * np.pi/52 ) ) )
        df5['week_of_year_cos'] = df5['week_of_year'].apply( lambda x: np.cos( x * ( 2. * np.pi/52 ) ) )

        cols_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month', 'competition_open_since_year', 'promo2',  'promo2_since_week', 'promo2_since_year', 'competition_time_month', 'promo_since_week', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'week_of_year_sin', 'week_of_year_cos']

        return df5[ cols_selected ]
    
    def get_prediction( self, model, original_data, test_date ):
        # Prediction
        pred = model.predict( test_date )
    
        # Join pred at original data
        original_data['prediction'] = np.expm1( pred )
        
        return original_data.to_json( orient='records', date_format='iso' )
import datetime
import numpy    as np
import pandas   as pd
import seaborn  as sns
import statsmodels.api         as sm
import statsmodels.formula.api as api
import matplotlib.gridspec     as gridspec

from scipy            import stats
from IPython.display  import Image
from matplotlib       import pyplot as plt
from sklearn import preprocessing as pp

rs = pp.RobustScaler()
mms = pp.MinMaxScaler()

# Test New Features Here

class metricsAndPlots():
    def metrics( self, df1, stats=False ):
        num_att = df1.select_dtypes( include=["int64", "float64"] )
        cat_att = df1.select_dtypes( include=["object"] )

        if stats == True:
            # Central Tendency mean, median
            c1 = pd.DataFrame( num_att.apply( np.mean ) ).T
            c2 = pd.DataFrame( num_att.apply( np.median ) ).T

            # Dispersion Means min, max, range, std, skew, kurtosis
            d1 = pd.DataFrame( num_att.apply( min ) ).T
            d2 = pd.DataFrame( num_att.apply( max ) ).T
            d3 = pd.DataFrame( num_att.apply( np.std ) ).T
            d4 = pd.DataFrame( num_att.apply( lambda x: x.max() - x.min() ) ).T
            d5 = pd.DataFrame( num_att.apply( lambda x: x.skew() ) ).T
            d6 = pd.DataFrame( num_att.apply( lambda x: x.kurtosis() ) ).T

            m = pd.concat( [d1, d2, d4, c1, c2, d3, d5, d6], axis=0 ).T.reset_index()
            m.columns = ["att", "min", "max", "range", "mean", "median", "std", "skew", "kurtosis"]
            return m
        else:
            return num_att, cat_att
        
    def mean_std(self, *list ):
        mean = np.round( np.mean( *list ), 3)
        std = np.round( np.std( *list ), 3)
        return mean, std
        
    def cramer_v( x, y ):
        cm = pd.crosstab( x, y ).values

        chi2 = stats.chi2_contingency( cm )[0]
        n    = cm.sum()
        r, k = cm.shape

        chi2corr = max( 0, chi2 - (r-1)*(k-1)/(n-1) )
        kcorr = k - (k-1)**2 / (n-1)
        rcorr = r - (r-1)**2 / (n-1)

        return np.sqrt( (chi2corr/n) / min(kcorr-1, rcorr-1) )
    
    def mean_absolute_error( self, y, yhat ):
        return np.mean( np.abs( y - yhat ) )

    def mean_absolute_percentage_error( self, y, yhat ):
        return np.mean( np.abs( (y - yhat) / y ) )

    def root_mean_squared_error( self, y, yhat ):
        return np.sqrt( np.mean( (y - yhat)**2 ) )
    
    def mean_percentage_error( self, y, yhat ):
        return np.sum( np.mean( y - yhat ) ) / 100

    def ml_error( self, model_name, y, yhat ):
        mae  = self.mean_absolute_error( y, yhat )
        mape = self.mean_absolute_percentage_error( y, yhat )
        rmse = self.root_mean_squared_error( y, yhat )
        return pd.DataFrame( {'Model Name': model_name, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse }, index=[0] )

    def anova_stats( self, df, format_, typ=2 ):
        return sm.stats.anova_lm(api.ols(format_, df4).fit(), typ=typ, test='F')

    def anova_scipy( self, *data ):
        f_stats, p_valor = stats.f_oneway( *data )
        return print(f'P-Valor: {p_valor}\nEstatística F: {f_stats}')

    def args( self, bins=np.arange(0, 2, 1), hstep='step', lwidth=2, c='r', label='Sales', normed=False):
        return {'bins': bins, 'histtype': hstep, 'linewidth': lwidth, 'color': c, 'density': normed, 'label': label}

    def args_b( self, edgecolor=['r', 'g', 'b'], lwidth=2, c=(1, 1, 1, 0)):
        return {'edgecolor': edgecolor, 'linewidth': lwidth, 'color': c}
    
    def train_test_prep( self, X_train, X_test, rs=rs, mms=mms ):
        for i in ['assortment', 'state_holiday', 'store_type']:
            f = X_train.groupby(i).size() / len(X_train)
            X_train[i] = X_train[i].apply( lambda x: f[x] )
            X_test[i] = X_test[i].apply( lambda x: f[x] )

        # Day Sim Bug
        # New Columns Here
        X_train['day_sin'] = X_train['day'].apply( lambda x: np.sin( x * ( 2. * ( 2. * np.pi / 30 ) ) ) )
        X_train['day_cos'] = X_train['day'].apply( lambda x: np.cos( x * ( 2. * ( 2. * np.pi / 30 ) ) ) )
        X_test['day_sin'] = X_test['day'].apply( lambda x: np.sin( x * ( 2. * ( 2. * np.pi / 30 ) ) ) )
        X_test['day_cos'] = X_test['day'].apply( lambda x: np.cos( x * ( 2. * ( 2. * np.pi / 30 ) ) ) )

        for i in zip( ['year', 'promo_time_week'], ['competition_time_month', 'competition_distance'] ):
            X_train[i[1]] = rs.fit_transform( X_train[[i[1]]].values )
            X_test[i[1]] = rs.fit_transform( X_test[[i[1]]].values )
            X_train[i[0]] = mms.fit_transform( X_train[[i[0]]].values )
            X_test[i[0]] = mms.fit_transform( X_test[[i[0]]].values )

        for i in zip( ['month', 'day_of_week', 'week_of_year'], [12, 7, 52] ):
            X_train[i[0]+'_sin'] = X_train[i[0]].apply( lambda x: np.sin( x * ( 2. * np.pi / i[1] ) ) )
            X_train[i[0]+'_cos'] = X_train[i[0]].apply( lambda x: np.cos( x * ( 2. * np.pi / i[1] ) ) )
            X_test[i[0]+'_sin'] = X_test[i[0]].apply( lambda x: np.sin( x * ( 2. * np.pi / i[1] ) ) )
            X_test[i[0]+'_cos'] = X_test[i[0]].apply( lambda x: np.cos( x * ( 2. * np.pi / i[1] ) ) )

        return X_train, X_test
    
    def cross_val( self, x_training, kfold, model_name, model, verbose=False ):
        mae_list, mape_list, rmse_list = [], [], []

        for k in reversed( range( 1, kfold+1 ) ):
            if verbose:
                print(f'KFold Number: {k}')
            validation_start_date = x_training['date'].max() - datetime.timedelta(days=k * 7 * 6)
            validation_end_date = x_training['date'].max() - datetime.timedelta(days=(k-1) * 7 * 6)

            # filters dataset
            training = x_training[x_training['date'] < validation_start_date]
            validation = x_training[(x_training['date'] >= validation_start_date) & (x_training['date'] <= validation_end_date)]

            # traning and validation dataset
            xtraining = training.drop(['date','sales'], axis=1)
            ytraining = np.log1p( training['sales'] )

            # validation
            xvalidation = validation.drop(['date','sales'], axis=1)
            yvalidation = np.log1p( validation['sales'] )

            # Model Fit
            m_ = model.fit( xtraining, ytraining )

            # Predict
            yhat = m_.predict( xvalidation )

            # Performace
            result = self.ml_error( model_name, np.expm1( yvalidation ), np.expm1( yhat ) )

            mae_list.append( result['MAE'] )
            mape_list.append( result['MAPE'] )
            rmse_list.append( result['RMSE'] )

        return pd.DataFrame({"Model Name": model_name,
                             "MAE Cv": str( self.mean_std( mae_list )[0] ) + ' +/- ' + str( self.mean_std( mae_list )[1] ),
                             "MAPE Cv": str( self.mean_std( mape_list )[0] ) + ' +/- ' + str( self.mean_std( mape_list )[1] ),
                             "RMSE Cv": str( self.mean_std( rmse_list )[0] ) + ' +/- ' + str( self.mean_std( rmse_list )[1] ) }, index=[0] )

    def create_scenarios( self, df, yhat, mae ):
        df['worst_scenario'] = df[yhat] - df[mae]
        df['best_scenario']  = df[yhat] + df[mae]
            
        if df[yhat]:
            for x in ['worst_scenario', 'best_scenario', 'prediction']:
                df[x] = np.around( df[x].tolist(), 2 )
                    
        else:
                print('ERROR') # Use Try catch to Get Error no IF statement

        return df

    def plot_pred_sales( self, x, y, yhat ):
        sns.lineplot( x, y, color='r' )
        sns.lineplot( x, yhat, color='k' )
        plt.xticks( rotation=20 )
        return None

    def plot_error_rate( self, x, y ):
        sns.lineplot( x, y, color='r' )
        plt.axhline( 1, linestyle='--', color='k' )
        plt.xticks( rotation=20 )
        return None

    def plot_target( self, df, var, y_l='density' ):
        fig, ax = plt.subplots( 1, 2, figsize=(15, 4) )
        for i in zip( df, ax, ['r', 'b'] ):
            i[1].hist(i[0], **args(bins=100, label=var, c=i[2]));
            i[1].set_xlabel('sales')
            i[1].set_ylabel(var)
            i[1].legend();

        return None

    def plot_categorical( self, df, color ):
        fig, ax = plt.subplots( figsize=(15, 12) )
        for j in catt_att.columns.tolist()[1:4]:
            if j == 'state_holiday':
                plt.subplot( 3, 2, 1 )
                for i in zip( ['christmas', 'easter_holiday', 'public_holiday'], color[:-1] ):
                    sns.kdeplot( df[df[j] == i[0]]['sales'], label=i[0], color=i[1], 
                                 shade=True, linewidth=2 )
                plt.legend();
            if j == 'store_type':
                plt.subplot( 3, 2, 3 )
                for i in zip( ['a', 'b', 'c', 'd'], color ):
                    sns.kdeplot( df[df[j] == i[0]]['sales'], label=i[0], color=i[1], 
                                 shade=True, linewidth=2 )
                plt.legend();
            else:
                plt.subplot( 3, 2, 5 )
                for i in zip( ['basic', 'extra', 'extended'], color[:-1] ):
                    sns.kdeplot( df[df[j] == i[0]]['sales'], label=i[0], color=i[1], 
                                 shade=True, linewidth=2 )
                plt.legend();
                for i in zip( range( 2, 8, 2 ), catt_att.columns.tolist()[1:4] ):
                    plt.subplot( 3, 2, i[0] )
                    sns.countplot( aux[i[1]], color=(1, 1, 1, 0), edgecolor=['navy', 'r', 'g', 'k'], linewidth=3 )

        return None

    def plot_num_att( self, df, cols ):
        fig, ax = plt.subplots( nrows=int(df.shape[1]/3), ncols=3, figsize=(20, 20) )
        for i in zip( ax.flatten(), cols, list(itertools.chain( *[['r', 'g', 'b'] for _ in range(0, int(df.shape[1]/3))]))):
            i[0].hist( df[i[1]], **args(bins=25, c=i[2], lwidth=3, label=i[1]) )
            i[0].legend()

        return None

    def progression_bar( self, df1, df2, target, test, x_label0, x_label1, text_dims0=100, text_dims1=100, text='Prejuizo\nem Vendas' ):
        fig, ax = plt.subplots( 2, 1, figsize=(15, 10) )                                        # Need Upgrade This Function
        ax[0].bar( aux1[test], aux1[target], **args_b(lwidth=3, edgecolor=['navy', 'k']) );     # its too Raw!
        ax[0].set_xlabel( x_label0 )
        ax[0].set_ylabel( target )
        ax[1].bar( aux2[test], aux2[target], **args_b(lwidth=3, edgecolor=['navy', 'k']) );
        ax[1].text(text_dims0, text_dims1, text, bbox={'facecolor':'white'})
        ax[1].set_xlabel( x_label1 )
        ax[1].set_ylabel( target )

        return None

    def plot_bar( self, df, test, hue=None, df_hue=None, scatter=False, xl='sales', df_scatter_x=None,  df_scatter_y=None ):
        if scatter == False:
            fig, ax = plt.subplots( 1, 2, figsize=(15, 5))
            ax = ax.flatten()
            ax[0].bar( df[test], df['sales'], **args_b(lwidth=3));
            ax[0].set_xlabel('sales')
            ax[0].set_ylabel('sales')
            ax[1] = sns.barplot( x=test, y='sales', hue=hue, data=df_hue, palette='winter' )
        else:
            fig, ax = plt.subplots( 1, 2, figsize=(15, 5))
            ax = ax.flatten()
            ax[0].bar( df[test], df['sales'], **args_b(lwidth=3));
            ax[0].set_xlabel( xl )
            ax[0].set_ylabel('sales')
            ax[1].scatter( df_scatter_x, df_scatter_y, c='red')
            plt.xticks( rotation=30 )

        return None
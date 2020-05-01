import pandas as pd
import numpy as np

def prepare_data():
    df = pd.read_csv('train-data/train.csv')

    # revenue = price * time viewed
    df['revenue'] = df['price'] * df['times_viewed']
    df['date'] = pd.to_datetime(df.year * 10000 + df.month*100 + 1,format='%Y%m%d')

    # top countries x others
    n_top_countries = 10 # number of countries to be considered individually
    top_countries = [country for country in df.groupby(['country']).agg({'revenue':'sum'})\
                    .sort_values('revenue', ascending = False).index[:n_top_countries]]

    df['country_group'] = df.country.apply(lambda x: x if x in top_countries else 'others')

    # we will get time series dataframes for each country
    agg_dict = {'customer_id' : 'nunique',
                'invoice' : 'nunique',
                'price' : 'mean',
                'revenue':'sum',
                'stream_id' : 'nunique',
                'times_viewed':'sum'
            }
    country_dict = {}
    
    for country in np.unique(df['country_group']):
        df_temp = df[df['country_group']==country].resample('D', on='date').agg(agg_dict)
        df_temp = df_temp.fillna(0)
        df_temp['country'] = country
        
        country_dict[country] = df_temp

    # adding new variables such as sum of revenue in the past x days
    for key in country_dict.keys():
        df_temp = country_dict[key]
        df_temp['revenue_past_3days'] = df_temp.rolling('3d')['revenue'].sum()
        df_temp['revenue_past_7days'] = df_temp.rolling('7d')['revenue'].sum()
        df_temp['revenue_past_30days'] = df_temp.rolling('30d')['revenue'].sum()
        df_temp['target'] = df_temp.rolling('30d')['revenue'].sum().shift(-30) #revenue in the next 30 days
        country_dict[key] = df_temp

    df = pd.concat([df for df in country_dict.values()], axis=0)

    df.to_csv('train-data/train-prepared.csv', index=True)

    print("Data prepared successfully!")

if __name__ == '__main__':
    prepare_data()
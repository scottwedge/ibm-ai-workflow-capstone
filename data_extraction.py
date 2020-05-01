import pandas as pd
import glob, os

def extract_data():
    df_list = [pd.read_json(file) for file in glob.glob("train-data/*.json")]
    df_list = [df.rename(columns = {'StreamID' : 'stream_id',
                                    'TimesViewed' : 'times_viewed',
                                    'total_price' : 'price'}) for df in df_list]

    df = pd.concat(df_list, sort=True)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    df.to_csv('train-data/train.csv', index=False)

    print("Data extracted successfully!")

if __name__ == '__main__':
    extract_data()
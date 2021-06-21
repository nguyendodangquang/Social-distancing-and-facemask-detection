import pandas as pd
import datetime as dt
from csv import writer

def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

def search_history(from_time, to_time, ID):
    df = pd.read_csv('./Capture/result.csv')
    df['Date'] = pd.to_datetime(df['Date'], format="%Y/%m/%d")
    df['Time'] = pd.to_datetime(df['Time'], format="%H:%M:%S").dt.time
    def concat(row):
        return dt.datetime.combine(row['Date'], row['Time'])
    df['Date_time'] = df.apply(concat, axis=1)
    new_df = df[(df['Date_time']>=from_time) & (df['Date_time']<=to_time)].iloc[:,0:6]
    def display_date(row):
        return str(row['Date'])[:10]
    new_df['Date'] = new_df.apply(display_date, axis=1)
    if ID == 'All':
        return new_df
    elif ID != 'All':
        return new_df[new_df['Camera_ID'] == ID]
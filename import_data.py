import os
import pandas as pd

def import_summary(path, data_file):
    if os.path.isfile(data_file):
        df = pd.read_pickle(data_file)
    else:
        df = pd.DataFrame()

        # only import files beginning with 20 and ending with xls
        file_xls = [path + f for f in os.listdir(path) 
                if f[:2] == '20' and f[-3:] == 'xls']

        # sort list by date
        file_xls.sort()

        # add data to new dataframe
        for f in file_xls:
            data = pd.read_excel(f, sheet_name=1, parse_dates=['Date'])
            df = df.append(data)

        print(df)

        df.Date = df.Date + pd.to_timedelta(df.Hour - 1, unit='h')
        del df['Hour']
        df.set_index('Date', inplace=True, drop=True)
        df.index.name = 'Date'
        df.sort_index()

        df.to_pickle(data_file)

    return df

def import_regions(path, data_file):
    # check if data already stored in serialized list
    if os.path.isfile(data_file):
        dfs = pd.read_pickle(data_file)

    # otherwise import data from xls files
    else:
        # setup dictionary of dataframes to match tabs in .xls files
        regions = ['ME', 'NH', 'VT', 'CT', 'RI', 'SEMASS', 'WCMASS',
                'NEMASSBOST']

        df_dict = {}
        for region in regions:
            df_dict[region] = pd.DataFrame()

        # only import files beginning with 20 and ending with xls
        file_xls = [path + f for f in os.listdir(path) 
                if f[:2] == '20' and f[-3:] == 'xls']

        # sort list by date
        file_xls.sort()

        # add data to new dataframe
        for f in file_xls:
            for region in regions:
                data = pd.read_excel(f, sheet_name=region, parse_dates=['Date'])
                df_dict[region] = df_dict[region].append(data)


        # convert hour to time stamp and convert to date
        for region in regions:
            df_dict[region].Date = df_dict[region].Date + pd.to_timedelta(
                    df_dict[region].Hour - 1, unit='h')
            del df_dict[region]['Hour']
            df_dict[region].set_index('Date', inplace=True, drop=True)
            df_dict[region].sort_index()


        # convert dictionary to multi-index dataframe
        dfs = pd.concat(df_dict, names=['Region', 'Date'])

        # write to file for fast access
        dfs.to_pickle(data_file)

    return dfs

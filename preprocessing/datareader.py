# load dependencies
import pandas as pd
import numpy as np


# read and clean data
def read_and_clean(filepath,separator = '\t', batch_size = 10000):
        iter_batch = pd.read_csv(
            filepath_or_buffer = filepath, 
            sep = separator, 
            chunksize = batch_size, 
            engine = 'python')

        data = pd.DataFrame()
        for each_batch in iter_batch:
            data = data.append(each_batch, ignore_index = True)

        # split features
        features_splited = data['features'].str.split(',', expand = True).astype(np.int16)
        features_splited.index = data['id_job']

        # get unique feature types
        unique_feature_types = features_splited.loc[:,0].unique()

        # in case if there is multiple feature types - each dataframe will be stored in dictionary
        data_dict = {}
        for each in features_splited.loc[:,0].unique():
            mask = features_splited.loc[:,0] == each
            temporary = features_splited[mask]
            temporary.drop(0,axis = 1,inplace = True)
            indexes = temporary.index

            data_dict[each] = temporary

        print("{path} - successfuly loaded. Avaliable feature types: {f_types}".format(path = filepath, f_types = list(data_dict.keys())))
        # return data as dictionary
        return data_dict
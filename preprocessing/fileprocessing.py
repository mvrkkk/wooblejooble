# load dependencies
import pandas as pd
import numpy as np
import multiprocessing as mp

def parse_chunk(dataframe):
    features_splited = dataframe['features'].str.split(',', expand = True).astype(np.int16)
    features_splited.insert(0,'id_job', dataframe['id_job'])#.astype(np.int16))
    return features_splited

def read_and_clean(path,separator = "\t",chunksize = 1000):
        
        batch_reader = pd.read_csv(
            filepath_or_buffer = path, 
            sep = separator, 
            chunksize = chunksize, 
            engine = 'python')
        
        with mp.Pool(mp.cpu_count()) as pool:
            result = pool.map(parse_chunk, batch_reader)
            data = pd.concat(result)
               
        unique_feature_types = data.loc[:,0].unique()
        data_dict = {}
        for each in data.loc[:,0].unique():
            mask = data.loc[:,0] == each
            temporary = data[mask]
            idx = temporary.pop('id_job')
            temporary.index = idx
            temporary.drop(0,axis = 1,inplace = True)
            

            data_dict[each] = temporary
        return data_dict 
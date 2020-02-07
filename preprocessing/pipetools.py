# load dependencies
from sklearn.pipeline import Pipeline, FeatureUnion
from joblib import dump, load
from pandas import DataFrame

class PipelineBuilder:
    """Pipeline craetor object"""
    
    # init class
    # will contains list of each pipelinesteps
    def __init__(self):
        self.steps = []
        pass
        
    # add step to pipeline 
    def add_step(self, step_title, transformer_obj):
        self.steps.append((step_title, transformer_obj)) 
        print("Pipeline steps: ", self.steps)
        return self
    
    # remove step from pipeline
    def remove_step(self, step_title):
        for index,i in enumerate(self.steps):
            if step_title in i[0]:
                self.steps.pop(index)
        print("Pipeline steps: ", self.steps)
        return self
    
    # freeze transformer
    def build_transformer(self, return_final_pipeline = True):
        if return_final_pipeline:
            self.feature_unions = FeatureUnion(self.steps, n_jobs = -1, )
            self.pipeline = Pipeline([('feature_unions', self.feature_unions)])
            return self.pipeline
        else:
            return self
        
    # add sklearn model opject to freezed Pipeline
    def add_model_to_pipe(self, step_title, model):
        raise NotImplementedError("Adding sklearn model's to pipeline not implemented yet.")
        
        
class JooblePipe:
    """Train or apply pipeline to data

       Parameters
       ----------
       """
    
    # initial
    def __init_(self):
        pass
    
    # train pipeline
    def train(self, new_pipeline_object, train_data):
        """
        Train method for class
        
        Parametrs
        ---------
        new_pipeline_object: PipelineBuilder object, freezed
        train_data: pandas' Dataframe with train subset 
        
        """
        #self.trained_pipeline = new_pipeline_object.fit(train_data)
        self.trained_pipeline = new_pipeline_object.fit(train_data)
        return self
        
    def save_transformer_state(self, filepath_name):
        """
        Saves pipeline in .pkl/.joblib file for further usage
        
        Parametrs
        ---------
        filepath_name: string, filepath_name where transformer will be saved as pkl/joblib file
        """
        if ('.pkl' in  filepath_name) or ('.joblib' in  filepath_name):
            dump(self.trained_pipeline, filepath_name)
        else:
            raise NameError('Add .pkl or .joblib to filepath_name.')
    
    
    def load_transformer(self, filepath_name):
        """
        Loads pkl/joblib file with transformer pipeline
        """
        self.trained_pipeline = load(filepath_name)

        return self 
    
    def transform_test(self, test_data, return_dataframe = True):
        if not return_dataframe:
            return self.trained_pipeline.transform(test_data)
        else:
            return DataFrame(self.trained_pipeline.transform(test_data))

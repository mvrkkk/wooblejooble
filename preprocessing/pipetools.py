# load dependencies
from sklearn.pipeline import Pipeline, FeatureUnion
from joblib import dump, load

class PipelineBuilder:
    """Main pipeline craetor object"""
    
    # init class
    def __init__(self):
        self.steps = []
        pass
        
    # add step method 
    def add_step(self, step_title, transformer_obj):
        self.steps.append((step_title, transformer_obj)) 
        print("Pipeline steps: ", self.steps)
        return self
    
    # remove step
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
            pipeline = Pipeline([('feature_unions', self.feature_unions)])
            return pipeline
        else:
            return self
    # add sklearn model opject to freezed Pipeline
    def add_model_to_pipe(self, step_title, model):
        raise NotImplementedError("Adding sklearn model's to pipeline not implemented yet.")
        
        
def save_transformer_state(pipe_obj, filepath_name):
    """
    Saves pipeline in .pkl/.joblib file for further usage
    """
    dump(pipe_obj, filepath_name)


def load_transformer(filepath_name):
    """
    Loads pkl/joblib file with transformer pipeline
    """
    loaded_pipe = load(filepath_name)

    return loaded_pipe

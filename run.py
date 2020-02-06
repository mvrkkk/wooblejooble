# dependencies
import pandas as pd

from itertools import chain
from preprocessing.datareader import read_and_clean
from preprocessing.transformers import Z_Score, MaxFeatureIndex, MaxFeatureAbsMeanDiff, CustomNormalizer
from preprocessing.pipetools import PipelineBuilder, save_transformer_state, load_transformer


# set constants
TRAIN_DATA_DIR = "data/train.tsv"
TEST_DATA_DIR = "data/test.tsv"

if __name__ == '__main__':
    
    # load and clean data
    train_data_dict = read_and_clean(TRAIN_DATA_DIR)
    test_data_dict = read_and_clean(TEST_DATA_DIR)
    
    # build pipeline
    pipeline = PipelineBuilder()
    pipeline.add_step('Z_Score',Z_Score())
    pipeline.add_step('MaxFeatureIndex', MaxFeatureIndex())
    pipeline.add_step('MaxFeatureAbsMeanDiff', MaxFeatureAbsMeanDiff())
    pipeline = pipeline.build_transformer()
    
    result = []
    for each_key in train_data_dict.keys():
        train_dx = train_data_dict[each_key].index
        test_idx = test_data_dict[each_key].index

        train_set = train_data_dict[each_key]
        test_set = test_data_dict[each_key]

        piper.fit(train_set)
        save_transformer_state(pipeline,'states/feature_{i}_transformer.pkl'.format(i = each_key))
        test_set_transformed = pd.DataFrame(pipeline.transform(test_set))
        test_set_transformed.set_index(test_idx, inplace=True)

        # generate column names
        standarterized_col_names = ['feature_{i}_stand_{i2}'.format(i = each_key, i2 = feature_index) for feature_index in range(0,256)]
        max_feature_index_colname = ['max_feature_{i}_index'.format(i = each_key)]
        max_feat_abs_diff_colname = ["max_feature_{i}_abs_mean_diff".format(i = each_key)]
        final_cols = list(chain(standarterized_col_names,max_feature_index_colname, max_feat_abs_diff_colname))

        test_set_transformed.columns = final_cols
        result.append(test_set_transformed)

    main_df = pd.concat(result)
    main_df.to_csv('output/test_proc.tsv')
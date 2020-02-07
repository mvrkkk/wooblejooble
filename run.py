# dependencies
import pandas as pd

from itertools import chain
from preprocessing.fileprocessing import read_and_clean
from preprocessing.transformers import Z_Score, MaxFeatureIndex, MaxFeatureAbsMeanDiff, CustomNormalizer
from preprocessing.pipetools import PipelineBuilder, JooblePipe


# set constants
TRAIN_DATA_DIR = "data/train.tsv"
TEST_DATA_DIR = "data/test.tsv"

if __name__ == '__main__':

    # load data
    train_data_dict = read_and_clean(TRAIN_DATA_DIR)
    test_data_dict = read_and_clean(TEST_DATA_DIR)

    # build pipeline
    # for test case - each feature types has similar pipeline-steps
    builder = PipelineBuilder()
    builder.add_step('z_score', Z_Score())
    builder.add_step('mfi', MaxFeatureIndex())
    builder.add_step('mfamd', MaxFeatureAbsMeanDiff())
    builded_pipe = builder.build_transformer()


    # apply pipeline
    result = []

    # each feature type must be processed with,own transformers and states
    for each_key in train_data_dict.keys():

        # split each data by it's feature type
        train_dx = train_data_dict[each_key].index
        test_idx = test_data_dict[each_key].index

        train_set = train_data_dict[each_key]
        test_set = test_data_dict[each_key]

        # for each feature type's train and test sets - build its own pipeline
        pipeline = JooblePipe()

        # train pipeline from builder object
        pipeline.train(builded_pipe, train_set)

        # save state (if needed)
        pipeline.save_transformer_state('states/feature_{i}_transformer.pkl'.format(i = each_key))

        # apply pipeline for feature type I for train and test sets
        test_set_transformed = pipeline.transform_test(test_set)
        test_set_transformed.set_index(test_idx, inplace=True)

        # generate column names
        standarterized_col_names = ['feature_{i}_stand_{i2}'.format(i = each_key, i2 = feature_index) for feature_index in range(0,256)]
        max_feature_index_colname = ['max_feature_{i}_index'.format(i = each_key)]
        max_feat_abs_diff_colname = ["max_feature_{i}_abs_mean_diff".format(i = each_key)]
        final_cols = list(chain(standarterized_col_names,max_feature_index_colname, max_feat_abs_diff_colname))

        test_set_transformed.columns = final_cols
        result.append(test_set_transformed)

    # stack results if there multiple feature types  
    main_df = pd.concat(result)
    main_df.to_csv('output/test_proc', sep='\t')
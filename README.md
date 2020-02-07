# Jooble Employment  task
This module was created in testing cases on purpose of Jooble

### Main features
- Flexible data processing pipeline creator
- Train, Transform, Save and Load your pipelines
- Handles multiple feature types in dataset
- Multiprocessing under the hood

### Description
Module contains main 3 submodules:
- **transformers** - where static and **custom** transformers are stored
- **pipetools** - built, train,save,apply your data processing pipeline
- **filerprocessing** - for reading .tsv files with fixed datastructure

This module contains a sample py-script  - **run.py** - which generates test sample of module's usage - *'otput/test_proc.tsv'*

### Dependencies -
- pandas
- numpy
- sklearn
- multiprocessing
- joblib

### Data requirements
- .tsv file with two columns, separated with tabspace:
	- **id_job** - unique job identifier
	- **features** - 257 columns, separated with comma, where:
		- 1 column in **features** block - unique feature_id identifier
		- other 256 columns - numerical attributes of given job_id and feature_id

Usage case provided in showcase.ipnyb file



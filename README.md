# Named Entity Recognition for Podcasts

Hello!

The most important files here are ner.py which contains our model and topic_modeling.py which contains our prototype downstream topic modeler. 

Running ner.py will load the samples from within the annotated_data folder and train and evaluate our model on them.

Running topic_modeling.py will print the results of our topic modeler as described in our paper.

Several utility files are included as well: txt2jsonl_batches.py, add_ep_ids.py, and json2jsonl.py which all perform some variety of file manipulation task which was needed at some step in the pipeline.
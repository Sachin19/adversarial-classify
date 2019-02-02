Use run.sh to run the code

model.py contains two classifier modules, one has an RNN encoder while other uses a CNN encoder. Look at train.py to setup options for both. 

datasets.py contains information about how the paths should be specified. 
train file contains the following tab separated info: <text> <label> <username> <space separated topic distribution - 50 scalars>
dev and test file contain the same information except for the topic distribution.

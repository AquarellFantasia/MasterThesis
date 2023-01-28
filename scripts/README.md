# Results of the training
OlderIterations - contain old, less organized results
iter* - results of specific iteration of training. Inside, it has 4 directories:
DevelopHyperparameterTunning.ipynb - notebook to queue batch of jobs or terminate all jobs
 - Graphs - Saved graphs of the loss function and accuracy values through training of each model.
 - Logs - Logs of the training, containing all the model details and the process of training. Also contains the history of loss function per epoch
 - Models - Saved models after each training
 - Scripts - Generated bsub scripts to queue the jobs. Important for debugging, if the model doesn't work
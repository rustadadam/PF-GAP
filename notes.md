# Helpful Stuff

Link to Resource Manager:
https://statrm.byu.edu/frontend/build

# Tasks
1. Build out half of a pipeline. 
    a. It needs to take data in, the model (like QUANT or PFGAP) and eventually static variables.
    b. It needs to output the RF proximities in a numpy array. (This assumes the estimator to be a random forest) 
2. Add Static abilities to QUANT. 
3. Add in other methods to the pipeline one by one. 


# Changes and Updates
1. Built out the static variables for basic Quant
2. Built out the pipeline to handle basic Quant
3. Adjusted QGAP to handle static variables
4. Built Pipeline to handle both Quant models and return proximities

# Questions before out meeting
1. Looking at the java code, to run it in python you need the data files. Is that how all of the data we will be given will look like? Sourced in files? Or should I convert the data to a file, and then clean it up. 
    a. I am currently planning on just using the java implementation but wrapping it a little tighter to manage all of the files (whats made throughout the process is deleted), so the code changes will be minimal. 
    b. Also, could you share the data files you have been using?
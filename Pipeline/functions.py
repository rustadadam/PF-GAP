import numpy as np

#Function to convert data to proximities
def data_to_proximities(model, Xtrain, ytrain, Xtest, Xstatic_train = None, Xstatic_test = None):

    #Fit model
    model = fit_model(model, Xtrain, ytrain, Xstatic_train)
    
    #Get proximities
    proximites = get_proximities(model, Xtrain) # Does this need test data instead: Xtest, Xstatic_test?

    return np.array(proximites)

def get_proximities(model, Xtrain):

    # Static Quant has proximites as an attribute
    if hasattr(model, "proximities"):
        return model.proximities
    
    #Get proximities -> Maybe make a switch case thing here
    elif hasattr(model, "get_proximities"): #QGAP
        prox = model.get_proximities()

        #If the matrix is sparse, convert to dense
        if hasattr(prox, "todense"):
            return prox.todense()
        else:
            return prox
        
    elif hasattr(model, "get_ensemble_proximities"):
        return model.get_ensemble_proximities(Xtrain, group = "all")

    raise AttributeError("Model does not have expected Methods.")

def fit_model(model, Xtrain, ytrain, Xstatic_train = None):
    try:

        #Using Static variables
        if Xstatic_train is None:
            model.fit(Xtrain, ytrain)
        else:
            model.fit(Xtrain, ytrain, static = Xstatic_train)

        return model
    except Exception as e:
        print("Error in fit_model: ", e)
        raise TypeError("Check the arguments passed into the function. The model may not be able to be fit with the given data.")
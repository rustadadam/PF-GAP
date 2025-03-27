

#Function to convert data to proximities
def data_to_proximities(model, Xtrain, ytrain, Xtest, Xstatic_train = None, Xstatic_test = None):

    #Fit model
    model = fit_model(model, Xtrain, ytrain, Xstatic_train)
    
    #Get proximities
    proximites = model.predict_proximities(Xtest, static = Xstatic_test) #NOTE: May want to abstract this away later

    return proximites

def fit_model(model, Xtrain, ytrain, Xstatic_train = None):
    try:

        #Using Static variables
        if Xstatic_train is not None:
            model.fit(Xtrain, ytrain, static = Xstatic_train)
        else:
            model.fit(Xtrain, ytrain)

        return model
    except Exception as e:
        print("Error in fit_model: ", e)
        raise TypeError("Check the arguments passed into the function. The model may not be able to be fit with the given data.")
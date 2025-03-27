

#Function to convert data to proximities
def data_to_proximities(model, Xtrain, ytrain, Xtest, Xstatic_train = None, Xstatic_test = None):

    #Fit model
    model.fit(Xtrain, ytrain, static = Xstatic_train) #NOTE: May want to abstract this away later

    #Get proximities
    proximites = model.predict_proximities(Xtest, static = Xstatic_test) #NOTE: May want to abstract this away later

    return proximites
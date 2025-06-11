# Model Helpers file

#& Imports
from rfgap import RFGAP

#& Model Get Prediction Methods

# TODO: Fix this method
def get_rfgap_pred():
    rfgap = RFGAP(prediction_type="classification", y = labels, oob_score = True)
    rfgap.fit(train, labels)


from RFGAP_Rocket.RFGAP_Rocket import RFGAP_Rocket

def get_rocket_pred(X_train, y_train, X_test, static_train, static_test, return_proximities=False, **rocket_params):
    rocket = RFGAP_Rocket(**rocket_params) #? Do we rather use the optimal as the defaults here instead?
    
    rocket.fit(X_train, y_train, static_train, weights = None)

    if return_proximities:
        return rocket.predict(X_test, static_test), rocket.get_proximities().toarray(), rocket.get_test_proximities(X_test, static_test).toarray()
    else: 
        return rocket.predict(X_test, static_test)
    



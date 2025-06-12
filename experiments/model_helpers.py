# Model Helpers file

#& Imports
import sys
sys.path.insert(0, '/yunity/arusty/PF-GAP/')


#Models
from rfgap import RFGAP
from RDST.rdst import RDST_GAP
from RFGAP_Rocket.RFGAP_Rocket import RFGAP_Rocket
from QGAP.qgap import QGAP
from Redcomets.Redcomets import REDCOMETS
from FreshPrince.FreshPrince import FreshPRINCE_GAP



# Other imports
import numpy as np


#& Model Get Prediction Methods

def get_rfgap_pred(X_train, y_train, X_test, static_train, static_test, return_proximities=False, **kwargs):

    #Format the data
    train = np.concat([X_train, static_train], axis=1)
    test = np.concat([X_test, static_test], axis=1)

    rfgap = RFGAP(prediction_type="classification", y = y_train, oob_score = True, **kwargs)
    rfgap.fit(train, y_train)

    if return_proximities:
        return rfgap.predict(train), rfgap.get_proximities().toarray(), rfgap.extend_prox(test).toarray()
    
    return rfgap.predict(test)


def get_rocket_pred(X_train, y_train, X_test, static_train, static_test, return_proximities=False, **rocket_params):
    rocket = RFGAP_Rocket(**rocket_params) #? Do we rather use the optimal as the defaults here instead?
    
    rocket.fit(X_train, y_train, static_train, weights = None)

    if return_proximities:
        return rocket.predict(X_test, static_test), rocket.get_proximities().toarray(), rocket.get_test_proximities(X_test, static_test).toarray()
    else: 
        return rocket.predict(X_test, static_test)
    
def get_rdst_pred(X_train, y_train, X_test, static_train, static_test, return_proximities=False, **kwargs):
    rdst = RDST_GAP(save_transformed_data = True, **kwargs)
    rdst.fit(X_train, y_train, static = static_train)

    if return_proximities:
        return rdst.predict(X_test, static = static_test), rdst.get_proximities().toarray(), rdst.extend_prox(X_test, static_test).toarray()
    
    return rdst.predict(X_test, static = static_test)

def get_qgap_pred(X_train, y_train, X_test, static_train, static_test, return_proximities=False, **kwargs):
    qgap = QGAP(**kwargs)
    qgap.fit(X_train, y_train, static = static_train)

    if return_proximities:
        return qgap.predict(X_test, static = static_test), qgap.get_proximities(), np.array(qgap.prox_extend(X_test, static_test))
    
    return qgap.predict(X_test, static = static_test)

    
def get_redcomets_pred(X_train, y_train, X_test, static_train, static_test, **kwargs):
    rc = REDCOMETS(static = static_train, variant=3, **kwargs)
    rc.fit(X_train, y_train)
    return rc.predict(X_test, static = static_test), None, None
    

def get_fresh_pred(X_train, y_train, X_test, static_train, static_test, return_proximities = False, **kwargs):
    fp = FreshPRINCE_GAP(**kwargs)
    fp.fit(X_train, y_train, static = static_train)

    if return_proximities:
        return fp.predict(X_test, static = static_test), np.array(fp.get_proximities().todense()), np.array(fp.get_extend(X_test, static_test).todense())
    
    return fp.predict(X_test, static = static_test)
# Helpful Stuff

Link to Resource Manager:
https://statrm.byu.edu/frontend/build

# Week of April 3rd

## Changes and Updates
1. Updated QGAP to fit just once
2. Created the RFGAP Rocket class
    a. Created the test notebook
    b. Allowed it to compute Static variables as well
    c. Updated Pipeline to take it in
3. Added FreshPrince (though it doesn't work)
    a. However, tsfresh seems finicky and it won't load correctly. 
4. Added RDST
    a. Added Static handling (for the proximities - won't predict with static yet)
    b. Enabled it to work in the pipeline
    
# Week of April 9th

## Changes and Updates
1. Investigating Redcomets Extra Rows:
    a. It applies the SMOTE algorithm to ensure we have a min_neighbors requirement for each point.
2. Applied an MDS to the four methods we have. 
    a. Data looks spread uniformly across the classes. All the methods seem to return similar values, with random points
    b. Trained A KNN on the MDS. Prediction score were not great with MDS reduction, but perfect with the full proximites. 
        1. How can we better visualize this without losing so much data? Look at the corresponding spread as a line for each row?
            a. Built several more visualizations. Note: simply doing a PCA shift preserves class accuracy. 


# Week of April 15th

## Tasks
2. Use RF_OOB scores for the prediction scoring as well
3. Tests against other benchmarks.
4. Look at the PHATE visualization to the proximities (use precomputed_affinity)
    a. Compare this to a PHATE just applied to the original dataset. 
    b. Maybe we can just focus on univariate

## Changes and Updates
1. Updated Redcomets proximity function to use RF GAP proximities
2. Updated Redcomets to handle the static features additions correctly. Oversampling techniques.
3. Worked on TSF. Decided not worth the effort

## Questions
1. TSF related questions
    1. I believe its trains a bunch of estimators on random parts of the interval. This makes me wonder how you could even get the proximities out of it? Because each estimator is a tree (not a forest) and is trained on different data. ( I guess intervas are selcted by Random, supervised, or random-supervised)
    2. 


# Week of April 29th

## Tasks
1. Create labels. File from Ben: https://drive.google.com/file/d/176LkdgGEVFJCW70DXx2pNWZHOgXZ3Ucq/view?usp=sharing 
2. K-fold similarities and tests
3. Look at the differing distance measures (DTW - Parralell, Euclidean, return correlation ,etc)
4. Look at weighting the static variables


## Possibilities to expand the project
1. New distances measures for PF-GAP
    1. Maybe using a regression adaptation.
    2. Imputations with proximities?
1. DTW - Parrelell 
    1. Might allow for multivariable time series. This could compute time-series by handling static features. https://pypi.org/project/dtwParallel/#description 


## Changes and Updates
1. Fixed PFGAP to work in the pipeline
    a. Built a wrapper class to fit and things
    b. It does use the train data as the test data -> This doesn't effect the proximities though, right?
2. Created the labels. NOTE: there are a few that have missing values... I wonder if the file I have has been corrupted
3. Retrieved the proximities with the data across all five models
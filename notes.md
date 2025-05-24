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
2. Created the labels from the data that was provided. NOTE: there are a few that have missing values... I wonder if the file I have has been corrupted. 
3. Retrieved the proximities with the provided data from BlackRock across all five models
    a. To Note: PF-GAP is significantly slower - (It took an hour and a half to run what the other models did in about 2 minutes) 
    b. Implemented the Static features to got the proximities with the provided data.
4. Implemented the K-fold validation tests
    a. NOTE: None of the methods to validate this are "KOSHER" - meaning they all kind of cheat in their own ways. To avoid this, we can do the rf-extend approach here, though we would have to implement this all the way from the beggining manually because of the nature of transforming the features with the proximity models we use.
    b. The cheater scores look good 
    c. The MDS visualizations look terrible. Did a K-Clustering algorithm to verify if results are optimal. They aren't. PCA is better though.
5. Got time_series_forest to work in a partial way (meaning that it returns proximities, but that it returns way too many nan's and doesn't aggregate the different models in the right way)
    a. I'm thinking we can just ignore this method
6. Successfully uncorrupted the requirements file so that FreshPrince works
    a. Needs implementation still: Difficulties -> "RotationForestClassifier is designed to use decision trees as base estimators, but it applies PCA transformations to subsets of the features before training the trees. This transformation makes it non-trivial to directly access the leaf indices of the trees because the input data is altered before being passed to the trees."
    b. To overcome this, I implemented a custom method. It now works, though we should talk through the process. 
7. Added a ton of distance measures. 
    a. Added them to the testing pipeline as well
    b. Implemented the return correlation
8. Added feature importances
    a. The static features have little overall importance, but they do have some (since there are so many features, any importance could be significant).
9. Added Static-feature weighting to RF-Rocket
    a. Applied weights to static features. It doesn't seem to make much of a difference. 
    


# Week of May 5th

## Tasks
5. Test DTW and other distances accuracy
6. Think about how much of the time-series matter
    a. We want to know what part of the series matter the most
        - Which part is contributing to the results the most?

    
## Possibilities to expand the project
1. Using an imputation method to classify the NaNs and the following. 

From Ben:
By the way, for DTW and Euclidean distances, you can get a classification accuracy like for the other models by using a 1-NN classifier. This is common in time series classification.

## Changes and Updates
1. Fixed missing data errors by manually filling them in. 
    a. Saved the data files for future ease of use. 
    b. Additionally, I also stored the prox files genereated by the classes. 
    c. Returned this to just dropping the files
2. Adapted QUANT and all other models to store the OOB scores. 
    a. It reaches around 60%, which seems more likely for ten classes.
    b. REDCOMETS scores the highest.
3. Wrote file to predict the time-series distances and save the numpy files. 
4. Added just a forest method by itself for the predictions. 
    a. Acheived a high OOB score. 
    b. Got the test score using data from another year. Got less accuracy, but still 70%
    c. Implemented all the test results for the differing folds
    d. Added MDS and PHATE Visualizations
5. Implemented ROCKET test accuracies in the test_accuracy file. 
    a. Implemented function to automate process
    b. Now returns f1, and other scores
6. Implemented RDST test accuracies in the test_accuracy file. 
    a. Added a predict method to the class rdst_GAP
7. Also added QGAP K-fold tests
8. Built out the Redcomets k-fold tests
9. Added FreshPrince to the stock_data -> Created the correct proximities
    a. Additionally, added it to the k-folds tests
10. Added MDS prox file. This shows the similarities between classes.
    a. Added PHATE and UMAP dimensionality techniques. They seem to be much more helpful. 

# Week of May 21st

## Tasks
2. Think about how much of the time-series matter
    a. We want to know what part of the series matter the most
        - Which part is contributing to the results the most?
3. Get data from YAHOO finance
    a. We want to test at different time-scales
    b. Start with Hourly -> Then we can go more fine-grained if we want
    c. Then also go to weekly and monthly
    d. NOTE: Don't use return values. We want to use the percent returns / z-scores
    e. We can linearly interpolate the missing values
4. Add the KNN model tests onto the resulting proximities
    a. We can put it into the model testing pipeline
    b. If the KNN scores on the proximities are much higher than the OOB score than we may have a problem with how the proximities are being defined
5. To investigate further -> we can visualize the the test points alongside the training points through our visualizations with PHATE and UMAP
6. Train on all the data with optimized hyperparameters
7. EXTRA TIME: how does this change with different ideas of creating proximities
    a. Can change the argument to "original" and "OOB" and proximity type. 

## Changes and updates
1. Tested the accuracy of the straight up distance measures. They seem to get better accuracy than the other models do. This is in test_accuracy file
2. 

## Possibilities to expand the project
1. Using an imputation method to classify the NaNs and the following. 

From Ben:
By the way, for DTW and Euclidean distances, you can get a classification accuracy like for the other models by using a 1-NN classifier. This is common in time series classification.

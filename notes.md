# Helpful Stuff

Link to Resource Manager:
https://statrm.byu.edu/radial 

#Model Spreadsheet
https://docs.google.com/spreadsheets/d/1h0BBgRL-IzkJKwZYiBjJfESYnEBhiJoYlBrqSk4YXPg/edit?gid=0#gid=0 

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

## Changes and updates
1. Tested the accuracy of the straight up distance measures. They seem to get better accuracy than the other models do. This is in test_accuracy file
2. Created hourly returns with percent changes and included labels and removed missing values
    a. Ran tests with Hourly scores. They are different that the daily. Redcommets did a little better, and Rocket did a little worse.
    b. It feels similar to having the different time_scopes
    c. Updated the visuals to demonstrate the hourly visuals as well
3. Can get the test proximities now for the following functions
    a. Rocket
    b. RDST
    c. Quant
    d. Freshprince

The KNN results seem similar (or often smaller than) the RF test results, which is similar to the OOB (though I didn't save these). 

4. Added a plotting function for the k-fold results
5. Sent the task of retreiving SHAP values to Kelvyn
6. Built a pipeline for getting the best hyperparameters. We have gotten them, and they really aren't all that much better. 
7. Created the tuned proximities for each matrix
    - Fresh Prince Looks the best, though RedComets has the highest f1 score
8. 

## Possibilities to expand the project
1. Using an imputation method to classify the NaNs and the following. 

From Ben:
By the way, for DTW and Euclidean distances, you can get a classification accuracy like for the other models by using a 1-NN classifier. This is common in time series classification.

# Week of Jun 4

## Todo:
5. Check out a new distance measure
 - Shape DTW
 - Don't do any DTW any the hourly scale (takes a long time)
 - Running this! (It was taking too long for some reason)

 ## Completed Stuff
 1. Got the hourly points for a year 
    - NOTE: I could not get it for the same time frame, as Yahoo finance deletes data older than 730 days. So instead I got the last full year of the same time frame. 
    - Reran the proximities to get OOB scores and made as fair as comparisons as possible. The daily returns do seem to do better overall. See these results on the stock_data.ipynb file
    - Applied these to the straight RF model. This lead to about a five percent decrease in accuracy
2. Fixed the non-diagonal entries of the similarity matricies to be 1 on the prox-visuals file. 
 - Fixed the color scheme as well.
 - Coordinated the sectors
3. Kelvyns stuff
4. Applied the Hourly points 
5. Optimized the straight Random Forest
6. Implemented shape_dtw
7. Saved Results files for both original and RF gap proximity types
    - Also saved results for distances
    - Created comparing_results.ipynb to visualize the results across different graphs. 
    - It seems like the distance measures often outpreform the models -> this could be that a 1 knn model is just an easier estimation
8. 

# Week of Jun 11

## TODO:
5. Implement new dataset
6. 

## Completed
1. Created a way to keep the dates associated with the data
2. Generated the confusion matricies. Its clear here that the daily predictions are infact better than the hourly. 
 - Its also interesting to see which class is often wrong. Financials seems to be the most easily mis-classified. 
 - For the distance matricies, I am realizing that I didn't preform the same train/ test split. Maybe this i why they are scoring higher as they seem to have more test data. 
3. Changed data to only use the 2024 and before. The OOB scores here as actually better this way (for the most part). This thereby is using the holdout year
- Also saved the rf-prox matrix
4. Ran testing results and made necessary changes to get cross validated results for the holdout year
- Some notes -> The Cross validation set was on the same year as the training data (I never tested on he 2024 data or used the 2024 static data ever). This is because we would need equal length time-series to make the classification. 
5. Made a google Collab with Sofia | Make a pipeline for handling the data
- sofiamaia789@gmail.com | smaia13@byu.edu
6. The algorithms clearly do better against benchmark data like the gunpoint data. 

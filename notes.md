# Helpful Stuff

Link to Resource Manager:
https://statrm.byu.edu/frontend/build

# Week of April 3rd
## Tasks 
3. Add in other methods to the pipeline one by one. 
    b. **DrCIFClassifier (Intervals)
    c. **RotationForestClassifier (Not a model for Time Series)

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
## Tasks
1. Investigating Redcomets extra rows
2. Applying an MDS transform to the proximities and visualizing them as points. 
    a. See what we get. Look at the data integrity. Is there a distinction between the results
    b. We can calculate the outlier scores -> we can plot these as well. (Not necessary)
    c. Run KNN and verify prediction (We would expect a similar score as the forest)
    d. Use the picture -> predictions using raw data
3. Figure out the Java Code. 

## Changes and Updates
1. Investigating Redcomets Extra Rows:
    a. It applies the SMOTE algorithm to ensure we have a min_neighbors requirement for each point.
2. Applied an MDS to the four methods we have. 
    a. Data looks spread uniformly across the classes. All the methods seem to return similar values, with random points
    b. Trained A KNN on the MDS. Prediction score were not great with MDS reduction, but perfect with the full proximites. 
        1. How can we better visualize this without losing so much data? Look at the corresponding spread as a line for each row?
            a. Built several more visualizations. Note: simply doing a PCA shift preserves class accuracy. 
    c.



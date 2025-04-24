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
    c.


# Week of April 15th

## Tasks
2. Use RF_OOB scores for the prediction scoring as well
3. Tests against other benchmarks.
4. Look at the PHATE visualization to the proximities (use precomputed_affinity)
    a. Compare this to a PHATE just applied to the original dataset. 
    b. Maybe we can just focus on univariate
5. Called TSF (Times Series Forest) -> Look into that. 
6. Figure out the Java Code. 

## Changes and Updates
1. Updated Redcomets proximity function to use RF GAP proximities
2. Updated Redcomets to handle the static features additions correctly. Oversampling techniques.
3. 

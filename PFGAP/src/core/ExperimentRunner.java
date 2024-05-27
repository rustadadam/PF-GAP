package core;

import java.io.File;
import java.io.PrintWriter;

import datasets.ListDataset;
import org.apache.commons.lang3.ArrayUtils;
import trees.ProximityForest;
import util.PrintUtilities;

import static application.PFApplication.UCR_dataset;
import static core.PFGAP.ForestProximity;

/**
 * 
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 *
 */

public class ExperimentRunner {

	ListDataset train_data;
	ListDataset test_data;
	private static String csvSeparatpr = "\t"; //for tsv files.

	public ExperimentRunner(){

	}

	public void run() throws Exception {
		//read data files
		//we assume no header in the csv files, and that class label is in the first column, modify if necessary
		ListDataset train_data_original =
				CSVReader.readCSVToListDataset(AppContext.training_file, AppContext.csv_has_header,
						AppContext.target_column_is_first, csvSeparatpr);
		ListDataset test_data_original =
				CSVReader.readCSVToListDataset(AppContext.testing_file, AppContext.csv_has_header,
						AppContext.target_column_is_first, csvSeparatpr);


		/**
		 * We do some reordering of class labels in this implementation,
		 * this is not necessary if HashMaps are used in some places in the algorithm,
		 * but since we used an array in cases where we need HashMaps to store class distributions maps,
		 * I had to to keep class labels contiguous.
		 *
		 * I intend to change this later, and use a library like Trove, Colt or FastUtil which implements primitive HashMaps
		 * After thats done, we will not be reordering class here.
		 *
		 */
		train_data = train_data_original.reorder_class_labels(null);
		test_data = test_data_original.reorder_class_labels(train_data._get_initial_class_labels());


		AppContext.setTraining_data(train_data);
		AppContext.setTesting_data(test_data);

		//allow garbage collector to reclaim this memory, since we have made copies with reordered class labels
		train_data_original = null;
		test_data_original = null;
		System.gc();

		//setup environment
		File training_file = new File(AppContext.training_file);
		String datasetName = training_file.getName().replaceAll("_TRAIN.txt", "");	//this is just some quick fix for UCR datasets
		AppContext.setDatasetName(datasetName);

		PrintUtilities.printConfiguration();

		System.out.println();

		//if we need to shuffle
		if (AppContext.shuffle_dataset) {
			System.out.println("Shuffling the training set...");
			train_data.shuffle();
		}


		for (int i = 0; i < AppContext.num_repeats; i++) {

			if (AppContext.verbosity > 0) {
				System.out.println("-----------------Repetition No: " + (i+1) + " (" +datasetName+ ") " +"  -----------------");
				PrintUtilities.printMemoryUsage();
			}else if (AppContext.verbosity == 0 && i == 0){
				System.out.println("Repetition, Dataset, Accuracy, TrainingTime(ms), TestingTime(ms), MeanDepthPerTree");
			}

			//create model
			ProximityForest forest = new ProximityForest(i);

			//train model
			forest.train(train_data);

			//test model
			ProximityForestResult result = forest.test(test_data);

			//Calculate array of forest proximities.
			System.out.println("Computing Forest Proximities...");
			double t5 = System.currentTimeMillis();
			Double[][] PFGAP = new Double[train_data.size()][train_data.size()];
			for (Integer k=0; k< train_data.size(); k++){
				for (Integer j=0; j< train_data.size(); j++){
					Double prox = ForestProximity(k,j,forest);
					PFGAP[k][j] = prox;
				}
			}
			double t6 = System.currentTimeMillis();
			System.out.print("Done Computing Forest Proximities. ");
			System.out.print("Computation time: ");
			System.out.println(t6-t5 + "ms");

			//print and export resultS
			result.printResults(datasetName, i, "");

			//export level is integer because I intend to add few levels in future, each level with a higher verbosity
			if (AppContext.export_level > 0) {
				result.exportJSON(datasetName, i);
			}

			//Now we print the PFGAP array to a text file.
			PrintWriter writer = new PrintWriter("ForestProximities.txt", "UTF-8");
			writer.print(ArrayUtils.toString(PFGAP));
			writer.close();
			Integer[] ytrain = new Integer[train_data.size()];
			for(Integer k=0; k< train_data.size(); k++){
				ytrain[k] = train_data.get_class(k);
			}
			PrintWriter writer2 = new PrintWriter("ytrain.txt", "UTF-8");
			writer2.print(ArrayUtils.toString(ytrain));
			writer2.close();

			if (AppContext.garbage_collect_after_each_repetition) {
				System.gc();
			}

		}

	}

}

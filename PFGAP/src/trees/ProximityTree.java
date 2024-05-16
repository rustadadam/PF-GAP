package trees;

import java.lang.reflect.Array;
import java.util.*;

import core.AppContext;
import core.TreeStatCollector;
import core.contracts.Dataset;
import datasets.ListDataset;
import distance.elastic.DistanceMeasure;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Random;
import java.util.stream.IntStream;

/**
 * 
 * @author shifaz
 * @email ahmed.shifaz@monash.edu
 *
 */

public class ProximityTree{
	protected int forest_id;	
	private int tree_id;
	protected Node root;
	protected int node_counter = 0;
	
	protected transient Random rand;
	public TreeStatCollector stats;
	protected ArrayList<Node> leaves;
	
	protected DistanceMeasure tree_distance_measure; //only used if AppContext.random_dm_per_node == false

	public ProximityTree(int tree_id, ProximityForest forest) {
		this.forest_id = forest.forest_id;
		this.tree_id = tree_id;
		this.rand = AppContext.getRand();
		stats = new TreeStatCollector(forest_id, tree_id);
		this.leaves = new ArrayList<Node>();
	}

	public Node getRootNode() {
		return this.root;
	}

	public ArrayList<Node> getLeaves() {return this.leaves;}
	
	public void train(Dataset data) throws Exception {
		//System.out.println("Training a tree");
		
		if (AppContext.random_dm_per_node ==  false) {	//DM is selected once per tree
			int r = AppContext.getRand().nextInt(AppContext.enabled_distance_measures.length);
			tree_distance_measure = new DistanceMeasure(AppContext.enabled_distance_measures[r]);		
			//params selected per node in the splitter class
		}
		
		this.root = new Node(null, null, ++node_counter, this);
		//Try putting the in- and out-of-bag stuff here.
		Dataset inbagData = new ListDataset();
		Dataset oobData = new ListDataset();

			//First we get the in-bag indices.
			int dummySize = data.size();
			int[]  randomIntsArray = IntStream.generate(() -> new Random().nextInt(dummySize)).limit(data.size()).toArray();
			//InBagIndices = new ArrayList<Integer>();
			for(int i=0; i<randomIntsArray.length; i++){
				this.getRootNode().InBagIndices.add(randomIntsArray[i]);
			}
			//int[] num = IntStream.range(0, data.size()).toArray();
			int[] distinctInBag = Arrays.stream(randomIntsArray).distinct().toArray();
			//setInBagIndices(distinctInBag);
			//setInBagIndices(randomIntsArray);

			this.getRootNode().multiplicities = new HashMap<Integer,Integer>();
			//now we compute the multiplicities (number of times the indices occur in the in-bag sample)
			for (int i=0; i<this.getRootNode().InBagIndices.size(); i++){
				if (this.getRootNode().multiplicities.containsKey((this.getRootNode().InBagIndices.get(i)))){
					this.getRootNode().multiplicities.put(this.getRootNode().InBagIndices.get(i),this.getRootNode().multiplicities.get(this.getRootNode().InBagIndices.get(i))+1);
				}
				else{
					this.getRootNode().multiplicities.put(this.getRootNode().InBagIndices.get(i),1);
				}
			}

			//now we need to get the out-of-bag indices.
			ArrayList<Integer> result = new ArrayList<Integer>(); //out-of-bag
			//ArrayList<Integer> resultA = (ArrayList<Integer>) InBagIndices; //new ArrayList<Integer>(); //for converting inbag
			for (int i = 0; i < randomIntsArray.length; i++){
				Boolean isInBag = ArrayUtils.contains(distinctInBag, i);
				if (!isInBag){result.add(i);}
				//else{resultA.add(i);}
			}
			int[] result2 = result.stream().mapToInt(i -> i).toArray();
			//OutOfBagIndices = new ArrayList<Integer>();
			for(int i=0; i<result2.length; i++){
				this.getRootNode().OutOfBagIndices.add(result2[i]);
			}
			//setOutOfBagIndices(result2);
			data.set_indices(this.getRootNode().InBagIndices);

			// Now we need to get the sub sample corresponding to the in-bag indices.
			Dataset data2 = new ListDataset(); //data;
			for (int index : this.getRootNode().InBagIndices){
				//System.out.println(Arrays.toString(data.get_series(index)));
				//System.out.println(data.get_index(index));
				data2.add(data.get_class(index), data.get_series(index), index);

			}
			//data = data2; //we're only training on in-bag samples.
			inbagData = data2;
			// Now we need to get the sub sample corresponding to the out-of-bag indices.
			Dataset data3 = new ListDataset();
			for (int index : this.getRootNode().OutOfBagIndices){
				//System.out.println(Arrays.toString(data.get_series(index)));
				//System.out.println(data.get_index(index));
				data3.add(data.get_class(index), data.get_series(index), index);

			}
			oobData = data3;
			//System.out.println(Arrays.toString(data.get_series(0)));
			//System.out.println(data.get_index(0));


			//Integer[] result = s1.toArray(new Integer[s1.size()]);



		this.root.train(inbagData, oobData);
	}
	
	public Integer predict(double[] query) throws Exception {
		Node node = this.root;

		while(!node.is_leaf()) {
			node = node.children[node.splitter.find_closest_branch(query)];
		}

		return node.label();
	}	

	
	public int getTreeID() {
		return tree_id;
	}

	
	//************************************** START stats -- development/debug code
	public TreeStatCollector getTreeStatCollection() {
		
		stats.collateResults(this);
		
		return stats;
	}	
	
	public int get_num_nodes() {
		if (node_counter != get_num_nodes(root)) {
			System.out.println("Error: error in node counter!");
			return -1;
		}else {
			return node_counter;
		}
	}	

	public int get_num_nodes(Node n) {
		int count = 0 ;
		
		if (n.children == null) {
			return 1;
		}
		
		for (int i = 0; i < n.children.length; i++) {
			count+= get_num_nodes(n.children[i]);
		}
		
		return count+1;
	}
	
	public int get_num_leaves() {
		return get_num_leaves(root);
	}	
	
	public int get_num_leaves(Node n) {
		int count = 0 ;
		
		if (n.children == null) {
			return 1;
		}
		
		for (int i = 0; i < n.children.length; i++) {
			count+= get_num_leaves(n.children[i]);
		}
		
		return count;
	}
	
	public int get_num_internal_nodes() {
		return get_num_internal_nodes(root);
	}
	
	public int get_num_internal_nodes(Node n) {
		int count = 0 ;
		
		if (n.children == null) {
			return 0;
		}
		
		for (int i = 0; i < n.children.length; i++) {
			count+= get_num_internal_nodes(n.children[i]);
		}
		
		return count+1;
	}
	
	public int get_height() {
		return get_height(root);
	}
	
	public int get_height(Node n) {
		int max_depth = 0;
		
		if (n.children == null) {
			return 0;
		}

		for (int i = 0; i < n.children.length; i++) {
			max_depth = Math.max(max_depth, get_height(n.children[i]));
		}
		
		return max_depth+1;
	}
	
	public int get_min_depth(Node n) {
		int max_depth = 0;
		
		if (n.children == null) {
			return 0;
		}

		for (int i = 0; i < n.children.length; i++) {
			max_depth = Math.min(max_depth, get_height(n.children[i]));
		}
		
		return max_depth+1;
	}
	
//	public double get_weighted_depth() {
//		return printTreeComplexity(root, 0, root.data.size());
//	}
//	
//	// high deep and unbalanced
//	// low is shallow and balanced?
//	public double printTreeComplexity(Node n, int depth, int root_size) {
//		double ratio = 0;
//		
//		if (n.is_leaf) {
//			double r = (double)n.data.size()/root_size * (double)depth;
////			System.out.format("%d: %d/%d*%d/%d + %f + ", n.label, 
////					n.data.size(),root_size, depth, max_depth, r);
//			
//			return r;
//		}
//		
//		for (int i = 0; i < n.children.length; i++) {
//			ratio += printTreeComplexity(n.children[i], depth+1, root_size);
//		}
//		
//		return ratio;
//	}		
	
	
	//**************************** END stats -- development/debug code
	
	
	
	
	
	
	
	public class Node{
	
		protected ArrayList<Integer> InBagIndices; //int[] InBagIndices;
		protected ArrayList<Integer> OutOfBagIndices; //ArrayList<Integer> OutOfBagIndices;
		protected Map<Integer, Integer> multiplicities;
		protected transient Node parent;	//dont need this, but it helps to debug
		protected transient ProximityTree tree;		
		
		protected int node_id;
		protected int node_depth = 0;

		protected boolean is_leaf = false;
		protected Integer label;

//		protected transient Dataset data;				
		protected Node[] children;
		protected Splitter splitter;
		//System.out.print()
		
		public Node(Node parent, Integer label, int node_id, ProximityTree tree) {
			this.parent = parent;
//			this.data = new ListDataset();
			this.node_id = node_id;
			this.tree = tree;
			this.InBagIndices = new ArrayList<>();
			this.OutOfBagIndices = new ArrayList<>();
			this.multiplicities = null;
			
			if (parent != null) {
				node_depth = parent.node_depth + 1;
			}
		}
		
		public boolean is_leaf() {
			return this.is_leaf;
		}
		
		public Integer label() {
			return this.label;
		}	
		
		public Node[] get_children() {
			return this.children;
		}

		public void setInBagIndices(ArrayList<Integer> indices) { this.InBagIndices=indices;}

		public ArrayList<Integer> getInBagIndices() {return this.InBagIndices;}

		public void setOutOfBagIndices(ArrayList<Integer> indices) { this.OutOfBagIndices=indices;}

		//public ArrayList<Integer> getOutOfBagIndices() {return this.OutOfBagIndices;}
		public ArrayList<Integer> getOutOfBagIndices() {return this.OutOfBagIndices;}

		public void setMultiplicities(Map<Integer, Integer> multiplicities) { this.multiplicities=multiplicities;}

		public Map<Integer, Integer> getMultiplicities() {return this.multiplicities;}
//		public Dataset get_data() {
//			return this.data;
//		}		
		
		public String toString() {
			return "d: ";// + this.data.toString();
		}		

		
//		public void train(Dataset data) throws Exception {
//			this.data = data;
//			this.train();
//		}		
		
		public void train(Dataset data, Dataset oobData) throws Exception {
//			System.out.println(this.node_depth + ":   " + (this.parent == null ? "r" : this.parent.node_id)  +"->"+ this.node_id +":"+ data.toString());
			//System.out.println("The train method was called on a node");
			//Debugging check
			if (data == null || data.size() == 0) {
				throw new Exception("possible bug: empty node found");
//				this.is_leaf = true;
//				return;
			}
			
			if (data.gini() == 0) {
				this.label = data.get_class(0);
				this.is_leaf = true;
				this.tree.leaves.add(this);
				return;
			}

			this.splitter = new Splitter(this);
						
			// Here we define the in-bag and out-of-bag indices for the tree (root node of the tree)
			//Dataset data2 = new ListDataset();
			/*Dataset inbagData = new ListDataset();
			Dataset oobData = new ListDataset();
			if (this.node_depth==0){
				//First we get the in-bag indices.
				int dummySize = data.size();
				int[]  randomIntsArray = IntStream.generate(() -> new Random().nextInt(dummySize)).limit(data.size()).toArray();
				InBagIndices = new ArrayList<Integer>();
				for(int i=0; i<randomIntsArray.length; i++){
					InBagIndices.add(randomIntsArray[i]);
				}
				//int[] num = IntStream.range(0, data.size()).toArray();
				int[] distinctInBag = Arrays.stream(randomIntsArray).distinct().toArray();
				//setInBagIndices(distinctInBag);
				//setInBagIndices(randomIntsArray);

				this.multiplicities = new HashMap<Integer,Integer>();
				//now we compute the multiplicities (number of times the indices occur in the in-bag sample)
				for (int i=0; i<this.InBagIndices.size(); i++){
					if (this.multiplicities.containsKey((this.InBagIndices.get(i)))){
						this.multiplicities.put(this.InBagIndices.get(i),this.multiplicities.get(this.InBagIndices.get(i))+1);
					}
					else{
						this.multiplicities.put(this.InBagIndices.get(i),1);
					}
				}

				//now we need to get the out-of-bag indices.
				ArrayList<Integer> result = new ArrayList<Integer>(); //out-of-bag
				//ArrayList<Integer> resultA = (ArrayList<Integer>) InBagIndices; //new ArrayList<Integer>(); //for converting inbag
				for (int i = 0; i < randomIntsArray.length; i++){
					Boolean isInBag = ArrayUtils.contains(distinctInBag, i);
					if (!isInBag){result.add(i);}
					//else{resultA.add(i);}
				}
				int[] result2 = result.stream().mapToInt(i -> i).toArray();
				OutOfBagIndices = new ArrayList<Integer>();
				for(int i=0; i<result2.length; i++){
					OutOfBagIndices.add(result2[i]);
				}
				//setOutOfBagIndices(result2);
				data.set_indices(InBagIndices);

				// Now we need to get the sub sample corresponding to the in-bag indices.
				Dataset data2 = new ListDataset(); //data;
				for (int index : this.InBagIndices){
					//System.out.println(Arrays.toString(data.get_series(index)));
					//System.out.println(data.get_index(index));
					data2.add(data.get_class(index), data.get_series(index), index);

				}
				//data = data2; //we're only training on in-bag samples.
				inbagData = data2;
				// Now we need to get the sub sample corresponding to the out-of-bag indices.
				Dataset data3 = new ListDataset();
				for (int index : this.OutOfBagIndices){
					//System.out.println(Arrays.toString(data.get_series(index)));
					//System.out.println(data.get_index(index));
					data3.add(data.get_class(index), data.get_series(index), index);

				}
				oobData = data3;
				//System.out.println(Arrays.toString(data.get_series(0)));
				//System.out.println(data.get_index(0));


				//Integer[] result = s1.toArray(new Integer[s1.size()]);
			}*/

			/*Dataset data2 = new ListDataset(); //data;
			for (int index : this.InBagIndices){
				//System.out.println(Arrays.toString(data.get_series(index)));
				//System.out.println(data.get_index(index));
				data2.add(data.get_class(index), data.get_series(index), index);

			}*/


			// Now we need to get the sub sample corresponding to the out-of-bag indices.
			/*Dataset data3 = new ListDataset();
			System.out.println(this.OutOfBagIndices);
			for (int index : this.OutOfBagIndices) {
				System.out.println(index);
				System.out.println(this.InBagIndices);
				//System.out.println(Arrays.toString(data.get_series(index)));
				//System.out.println(data.get_index(index));
				data3.add(data.get_class(index), data.get_series(index), index);
			}*/

			/*Dataset[] best_splits = new Dataset[]{};
			if(this.node_depth==0){
				best_splits = splitter.find_best_split(inbagData);
			}

			else{
				best_splits = splitter.find_best_split(data);
			}*/


			Dataset[] best_splits = splitter.find_best_split(data);
			Dataset[] oob_splits = new Dataset[best_splits.length];
			this.children = new Node[best_splits.length];
			for (int i = 0; i < children.length; i++) {
				this.children[i] = new Node(this, i, ++tree.node_counter, tree);
				this.children[i].setInBagIndices(best_splits[i]._internal_indices_list());
				oob_splits[i] = new ListDataset();
			}
			//Now we need to let the oob indices trickle down (set the oob indices for the children).
			//System.out.println(oobData.size());
			for (int i=0; i<oobData.size(); i++){
				int ind = oobData.get_index(i);
				int label = oobData.get_class(i);
				double[] series = oobData.get_series(i);
				int branch = splitter.find_closest_branch(oobData.get_series(i));
				//if (this.children[branch] != null){
				//	this.children[branch].OutOfBagIndices.add(ind);
				//}
				this.children[branch].OutOfBagIndices.add(ind);
				oob_splits[branch].add(label,series,ind);
			}

			// Now train on the children.
			for (int i = 0; i < best_splits.length; i++) {

				//this.children[i].train(best_splits[i]);
				this.children[i].train(best_splits[i],oob_splits[i]);
			}
		}

		public Splitter getSplitter() {
			return splitter;
		}



	}
	
}

package core;

import trees.ProximityForest;
import trees.ProximityTree;

import java.util.ArrayList;

public class PFGAP{

    public static ArrayList<ProximityTree> getSi(Integer i, ProximityForest pf){
        ArrayList<ProximityTree> Si = new ArrayList<ProximityTree>();
        ProximityTree[] trees = pf.getTrees();
        for(ProximityTree tree:trees){
            ArrayList<Integer> oob = tree.getRootNode().getOutOfBagIndices();
            if(oob.contains(i)){
                Si.add(tree);
            }
        }
        return Si;
    }

    //public static ProximityTree.Node getJiLeaf(Integer i, ProximityTree t){
    public static ArrayList<Integer> getJi(Integer i, ProximityTree t){
        ArrayList<Integer> Ji = new ArrayList<>();
        ArrayList<ProximityTree.Node> leaves = t.getLeaves();
        //System.out.println(leaves.size());
        for (ProximityTree.Node leaf : leaves){
            ArrayList<Integer> inbags = leaf.getInBagIndices();
            ArrayList<Integer> oob = leaf.getOutOfBagIndices();
            //System.out.println(oob);
            //System.out.println(i);
            if(oob.contains(i)){
                Ji = inbags;
                //System.out.print(Ji);
            }
        }
        return Ji;
    }

    public static Double ForestProximity(Integer i, Integer j, ProximityForest pf){
        ArrayList<ProximityTree> Si = getSi(i,pf);
        //Double[] terms = new Double[]{};
        ArrayList<Double> terms = new ArrayList<>();
        for (ProximityTree t : Si){
            Integer cj = t.getRootNode().getMultiplicities().get(j);
            ArrayList<Integer> Mi = getJi(i,t);
            if (Mi.contains(j)){
                double Cj = (double) cj;
                terms.add((Cj/Mi.size())/ Si.size());
            }
            else{
                terms.add((double) 0);
            }
        }
        double sum = 0;
        for (Double term : terms){
            sum += term;
        }
        return sum;
    }
}
package classification;

import java.io.FileReader;
import java.util.Arrays;
import java.util.function.IntBinaryOperator;

import weka.core.*;

public class Caku {

	/**
	 * The decision table.
	 */
	Instances data;
	/**
	 * 0 represent  virgin
	 * 1 represent query 
	 * 2 represent classify
	 */
	int[] instanceStatus;
	
	double[] labels;
	
	/**
	 ************************* 
	 * Constructor.
	 * 
	 * @param paraFilename
	 *            The given file.
	 ************************* 
	 */
	public Caku(String paraFilename) {
		data = null;
		try {
			FileReader fileReader = new FileReader(paraFilename);
			data = new Instances(fileReader);
			fileReader.close();
			data.setClassIndex(data.numAttributes() - 1);
			System.out.println(data);
		} catch (Exception ee) {
			System.out.println("Cannot read the file: " + paraFilename + "\r\n" + ee);
			System.exit(0);
		} // Of try

		// Initialize
		// instanceStates = new int[data.numInstances()];
		// labels = new int[data.numInstances()];
		// Arrays.fill(labels, -1);
	}// Of Kak\\\\\\\\\\\\\\\\\\
	
	public void querySplitClassify(int[] paraBlockInstances, int paraInitialPoint) {
		//Step 1. How many instances to query
		int[] tempMaxInstancesToQuery = lookup(paraBlockInstances.length);

		//Step 2. Which instances have been queried in this block.
		int[] tempQueried = new int[tempMaxInstancesToQuery.length];
		int tempIndex = 0;
		for (int i = 0; i < paraBlockInstances.length; i ++) {
			if (instanceStatus[paraBlockInstances[i]] == 1) {
				tempQueried[tempIndex] = paraBlockInstances[i];
				tempIndex ++;
			}
		}
		
		//Step 3. These queried instances are pure?
		

		//Step 4. Query the first instance
		if (instanceStatus[paraInitialPoint] == 0) {
			labels[paraInitialPoint] = data.instance(paraInitialPoint).value(data.numAttributes() - 1);
			instanceStatus[paraInitialPoint] = 1;
		}
		int tempFirstLabel = labels[paraInitialPoint];

		boolean tempPure = true;
		//Step 5. Query other instances one by one
		for (int i = 0; i < tempMaxInstancesToQuery[0] - 1; i ++) {
			//Find the farthest point
			int tempFarthest = findFarthest();
			//Query
			
			//An instance with different label is queried.
			if (tempCurrentLabel != tempFirstLabel) {
				tempSecondCenter = tempFarthest; 
				tempPure = false;
				break;
			}
		}//Of for i
		
		//Step 6. Now split and 
		if (!tempPure) {
			int[][] tempSplitted = cluster(paraBlockInstances, 2);
//			int[][] tempSplitted = cluster(paraBlockInstances, tempFirstCenter, tempSecondCenter);

			querySplitClassify(tempSplitted[0]);
			querySplitClassify(tempSplitted[1]);
		} else {
			
		}
	}//Of querySplitClassify
	
	/**
	 ********************* 
	 * Look up optimal R and B
	 ********************* 
	 */
	private int[] lookup(int pSize) {
		// Linear Search
		//star[] :note the tmpMinCost instance in every query
		//starLoose: the final MinCost instance index
		//ra :the expect number of positive instances
		//ra[0] : compute the final starLoose as the compute the tmpCost
		//isFind : find the MinCost instance
		double[] tmpMinCost = new double[] { 0.5 * mCost[0] * pSize, 0.5 * mCost[1] * pSize };
		//double[] tmpMinCost = new double[] {mCost[0] * pSize, mCost[1] * pSize }; //pair with the starLoose computation so use the 0.5
		int[] star = new int[2];
		int[] starLoose = new int[2];
		boolean[] isFind = new boolean[2];
		double[] ra = new double[pSize + 1];
		ra[0] = 0.5;
		double[] tmpCost = new double[2];
		for (int i = 1; i <= pSize; i++) {
			if (pSize >= 1000) {
				ra[i] = (i + 1.0) / (i + 2.0); //the expect numbers of positive instances  similarly the N approach  infinite
			} else {
				ra[i] = expectPosNum(i, 0, pSize) / pSize; //else use the CADU equation 4 to compute the ra
			}
			for (int j = 0; j < 2; j++) {
				tmpCost[j] = (1 - ra[i]) * mCost[j] * pSize + tCost * i;
				if (tmpCost[j] < tmpMinCost[j]) {
					tmpMinCost[j] = tmpCost[j];
					star[j] = i;
					if (i == pSize) {
						Arrays.fill(isFind, true);
					} // Of if
				} else {
					isFind[j] = true;
				} // Of if
			} // Of for j
				//System.out.println("star : " + intArrayToString(star));
				//System.out.println("cost :" + Arrays.toString(tmpCost));
				//System.out.println("QueriedInstance: " + i + ", cost: " + tmpCost[0] + "\t");
			if (isFind[0] && isFind[1]) {
				Arrays.fill(isFind, false);
				for (int j = 0; j <= i; j++) {
					for (int k = 0; k < 2; k++) {
						tmpCost[k] = (1 - ra[j]) * mCost[k] * pSize + tCost * j;
						if (tmpCost[k] <= tmpMinCost[k] && !isFind[k]) {
							isFind[k] = true;
							starLoose[k] = j;
						} // Of if
						if (isFind[0] && isFind[1]) {
							//System.out.println("------starLoose is : " + intArrayToString(starLoose));
							return starLoose;
						} // Of if
					} // Of for k
				} // Of for j
			} // Of if
		} // Of for i
		return new int[] { 0, 0 };
		// throw new RuntimeException("Error occured in lookup("+pSize+")");
	}// Of lookup
	
	/**
	 ************************* 
	 * Find the next farthest instance of the block.
	 * 
	 * @param paraCurrentBlock
	 *            The given block.
	 * @param paraLabeledInstances
	 *            Labeled instances in the current block.
	 ************************* 
	 */
	public double manhattanDistance(int paraFirstIndex, int paraSecondIndex) {
		double resultDistance = 0;
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			resultDistance += Math.abs(data.instance(paraFirstIndex).value(i) - data.instance(paraSecondIndex).value(i));
		}// Of for i
		return resultDistance;
	}// Of manhattanDistance
	
	/**
	 ***************
	 * Compute the distance between an object and an array
	 ***************
	 */
	public double distance(int paraIndex, double[] paraArray) {
		double resultDistance = 0;
		for (int i = 0; i < paraArray.length; i++) {
			resultDistance += Math.abs(data.instance(paraIndex).value(i) - paraArray[i]);
		} // Of for i
		return resultDistance;
	}// Of distance
	/**
	 ************************* 
	 * Find the next farthest instance of the block.
	 * 
	 * @param paraCurrentBlock
	 *            The given block.
	 * @param paraLabeledInstances
	 *            Labeled instances in the current block.
	 ************************* 
	 *            public int findFarthestAndQuery(int[] paraCurrentBlock, int[]
	 *            paraLabeledInstances) {
	 * 
	 *            }//Of
	 */

	/**
	 ************************* 
	 * Find the next farthest instance of the block.
	 * 
	 * @param paraCurrentBlock
	 *            The given block.
	 * @param paraLabeledInstances
	 *            Labeled instances in the current block.
	 ************************* 
	 */
	public int findFarthest(int[] paraCurrentBlock, int[] paraLabeledInstances) {
		int resultFarthest = -1;

		double tempMaxDistanceSum = -1;
		for (int i = 0; i < paraCurrentBlock.length; i++) {
			double tempDistanceSum = 0;
			for (int j = 0; j < paraLabeledInstances.length; j++) {
				if (paraCurrentBlock[i] == paraLabeledInstances[j]) {
					tempDistanceSum = -1;
					break;
				}// Of if

				tempDistanceSum += manhattanDistance(paraCurrentBlock[i],
						paraLabeledInstances[j]);
			}// Of for j

			System.out.println("" + paraCurrentBlock[i] + " to labeled = " + tempDistanceSum);

			// Update
			if (tempDistanceSum > tempMaxDistanceSum + 1e-6) {
				resultFarthest = paraCurrentBlock[i];
				tempMaxDistanceSum = tempDistanceSum;
			}// Of if
		}// Of for i
		return resultFarthest;
	}// Of findFarthest

	/**
	 ************************* 
	 * Find the next farthest instance of the block.
	 * 
	 * @param paraCurrentBlock
	 *            The given block.
	 * @param paraLabeledInstances
	 *            Labeled instances in the current block.
	 ************************* 
	 */
	public void testFindFarthest() {
		int[] tempBlock = { 1, 4, 6, 7, 10, 59, 77 };
		int[] tempQuried = { 4, 7 };

		int tempFarthest = findFarthest(tempBlock, tempQuried);

		System.out.println("The farthest one is: " + tempFarthest);
	}// Of testFindFarthest

	
	/**
	 ************************* 
	 * The main entrance
	 * @author Fan Min
	 ************************* 
	 */
	public static void main(String[] args) {
		System.out.println("Hello.");
		Caku tempKaku = new Caku("data/iris.arff");

		tempKaku.testFindFarthest();
	}// Of main

}// Of class Kaku

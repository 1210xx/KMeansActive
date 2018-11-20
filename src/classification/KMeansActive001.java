package classification;

import java.io.FileReader;
import java.math.BigDecimal;
import java.util.Arrays;
import java.util.PrimitiveIterator.OfDouble;
import java.util.Random;
import java.util.function.IntPredicate;

import weka.core.Instances;

public class KMeansActive001 {
	/**
	 * The random function 
	 */
	static Random random =new Random();
	/**
	 * the data of class instance
	 */
	Instances data;

	/**
	 * the states of instances
	 *  1 represents bought. 
	 *  2 represents predicted. 
	 *  0 is untreated. 
	 */
	int[] instanceStates; // 1 represents bought and 2 represents predicted.  and 0 is untreated 

	/**
	 * the decision attribution of instances 
	 */
	int[] labels;

	/**
	 * the center of KMeans 
	 */

	double[][] currentCenters;
	/**
	 * the cost of teach
	 */
	double tCost;

	/**
	 *  the cost of misclassification
	 */
	double[] mCost;

	/**
	*The constructor.Read the data and initialize the misclassification cost and teach cost 
	* @param paraFilename
	* @param paraMisclassificationCost
	* @param paraTeachCost
	*/
	public KMeansActive001(String paraFilename, double[] paraMisclassificationCost, double paraTeachCost) {
		// Of the first constructor
		data = null;
		try {
			FileReader fileReader = new FileReader(paraFilename);
			data = new Instances(fileReader);
			fileReader.close();
			data.setClassIndex(data.numAttributes() - 1);
			//			System.out.println("data.instance[data.numAttributes - 1]:  " + data.instance(1).value(data.numAttributes() - 1));
		} catch (Exception ee) {
			System.out.println("Cannot read the file: " + paraFilename + "\r\n" + ee);
			System.exit(0);
		} // Of try
			// Initialize
		instanceStates = new int[data.numInstances()];
		Arrays.fill(instanceStates, 0);
		labels = new int[data.numInstances()];
		Arrays.fill(labels, -1);
		mCost = paraMisclassificationCost;
		tCost = paraTeachCost;
		System.out.println(data);
	}

	/**
	 ***************
	 * The entrance. 
	 ***************
	 */
	public static void main(String args[]) {
		//why mcost initialize is {2,4} 

		double[] mCost = { 2, 4 };
		double tCost = 1;
		//int[] tempBlock = { 1, 4, 5, 6, 59, 100 };
		//int[] tempStates = { 0, 0, 1, 1, 2, 0 };
		//		System.out.println("The clustering result is: " + Arrays.toString(tempLeaner.cluster(3, tempBlock)));
		KMeansActive001 tempLeaner = new KMeansActive001("Data/CAKU.arff", mCost, tCost);
		//int[] tempBlock = {1, 4, 5, 6, 59, 121};
		//System.out.println("The clustering result is: " + Arrays.toString(tempLeaner.cluster(tempBlock)));
		tempLeaner.learn(tempLeaner.preLearn());
		System.out.println("DONE.....");	  
		System.out.println(tempLeaner.totalCost());

		
		//the 20 times execute
		//why mcost initialize is {2,4} 
//
//		double[] mCost = { 2, 4 };
//		double tCost = 1;
//		double avgCost = 0;
//		KMeansActive001 tempLeaner = new KMeansActive001("Data/CAKU.arff", mCost, tCost);
//		//int[] tempBlock = { 1, 4, 5, 6, 59, 100 };
//		//int[] tempStates = { 0, 0, 1, 1, 2, 0 };
//		System.out.println("The clustering result is: " + Arrays.toString(tempLeaner.cluster(3, tempBlock)));
//		tempLeaner.preLearn();
//		avgCost += tempLeaner.totalCost();
//		System.out.println("OK");
//		for (int i = 0; i < 20; i++) {
//			//			int[] tempBlock = {1, 4, 5, 6, 59, 121};
//			System.out.println("The clustering result is: " + Arrays.toString(tempLeaner.cluster(tempBlock)));
//			tempLeaner.learn(tempLeaner.preLearn());
//			avgCost += tempLeaner.totalCost();
//			//		tempLeaner.instanceStates = tempStates;
//			//		tempLeaner.findQueriedInstances(tempBlock);
//			System.out.println("OK");
//		}
//		System.out.println(tempLeaner.totalCost());
//
//		System.out.println(avgCost / 20);

	}// Of main

	/**
	 ***************
	 * preLearn, read the data into a array 
	 ***************
	 */
	public int[] preLearn() {
		int[] originalBlock = new int[data.numInstances()];
		for (int i = 0; i < originalBlock.length; i++) {
			originalBlock[i] = i;
		} // Of for i
		System.out.println("instanceStates: " + Arrays.toString(instanceStates));
		//cluster(originalBlock);
		System.out.println("instanceStates: " + Arrays.toString(instanceStates));
		System.out.println("labels: " + Arrays.toString(labels));
		return originalBlock;
	}// Of preLearn

	/**
	 ***************
	 * Active learn. 
	 ***************
	 */
	public void learn(int[] paraBlock) {
		//if the Block is to small buy the label directly.
		System.out.println("Learn: " + Arrays.toString(paraBlock));
		if (paraBlock.length < 5) { //5 is randomly assign according to the size of data
			System.out.println("The block is too small, buy these labels directly");
			for (int i = 0; i < paraBlock.length; i++) {
				instanceStates[paraBlock[i]] = 1;
				labels[paraBlock[i]] = (int) data.instance(paraBlock[i]).classValue();
			} // Of for i
			return;
		} // Of if

		// Step 1. Scan the existing labels
		int tempFirstLabel = -1;
		int tempCurrentIndex = -1;
		int tempCurrentLabel = -1;
		//find the first brought point
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStates[paraBlock[i]] == 1) {
				tempFirstLabel = (int) data.instance(paraBlock[i]).classValue();
				tempCurrentIndex = i;
				break;
			} // Of if
		} // Of for i

		// Is it already impure?
		//find next point of block and decide the next step to split or classify from the next of the tempCurrent index
		if (tempCurrentIndex != -1) {
			for (int i = tempCurrentIndex + 1; i < paraBlock.length; i++) {
				if (instanceStates[paraBlock[i]] == 1) {
					tempCurrentLabel = (int) data.instance(paraBlock[i]).classValue();
					//if current label not equal to first label split
					if (tempCurrentLabel != tempFirstLabel) {
						// note this method to modify *****
						splitAndLearn(paraBlock);
						return;
					} else {
						// else Step 3. Predict others in this block.
						for (int j = 0; j < paraBlock.length; j++) {
							if (instanceStates[paraBlock[j]] != 1) {
								instanceStates[paraBlock[j]] = 2;
								labels[paraBlock[j]] = tempFirstLabel;
							} // Of if
						} // Of for i
					}
				} // Of if
			} // Of for i
		} // Of if

		//find the queried instances 
		int[] tempQueriedInstance = new int[paraBlock.length];
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStates[i] == 1) {
				for (int j = 0; j < paraBlock.length; j++) {
					tempQueriedInstance[j] = instanceStates[i];
				}
			}
		}
		// Step 1. Find representatives that the points nearest to the centers.
		//		int[] tempReprentatives = findRepresentatives(paraBlock, tempFirstLabel);
		int tempReprentatives = -1;
		tempReprentatives = findFarthest(paraBlock, findQueriedInstances(paraBlock));
		if (tempReprentatives != -1) {
			System.out.println("tempReprentatives = " + paraBlock[tempReprentatives]);
			int tempIndex = 0;
			if (tempCurrentIndex == -1) {
				// Buy the label of the first instances.
				tempFirstLabel = (int) data.instance(paraBlock[0]).classValue();
				instanceStates[paraBlock[0]] = 1;
				labels[paraBlock[0]] = tempFirstLabel;
				tempIndex++;
			} // Of if

			// Step 2. Buy labels one by one, and split in two if there are different labels
			tempCurrentLabel = (int) data.instance(tempReprentatives).classValue();
			instanceStates[tempReprentatives] = 1;
			labels[tempReprentatives] = tempCurrentLabel;
			if (tempCurrentLabel != tempFirstLabel) {
				splitAndLearn(paraBlock);
				return;
			} // Of if
		} // Of if
		return;
	}// Of learning

	/**
	 ***************
	 * Cluster. 2-Means Cluster
	 ***************
	 */
	public int[] cluster( int[] paraBlock, int paraK) {
		System.out.println("----------clustering K-Means");
		System.out.println("----------paraK = " + paraK);
		System.out.println("+++++++++++BlockSize = " + paraBlock.length);
		// Step 1. Initialize
		paraK = 2;
		int tempBlockSize = paraBlock.length;
		int[] tempCluster = new int[tempBlockSize];
		double[][] tempCenters = new double[paraK][data.numAttributes() - 1];
		double[][] tempNewCenters = new double[paraK][data.numAttributes() - 1];

		// Step 2.select 2 data points.
		for (int i = 0; i < paraK; i++) {
			int tempIndex = random.nextInt(tempBlockSize);
			System.out.println("The current index is: " + tempIndex);
			for (int j = 0; j < data.numAttributes() - 1; j++) {
				tempNewCenters[i][j] = data.instance(paraBlock[tempIndex]).value(j);
			} // Of for j
		} // Of for i
		System.out.println("Randomly selection: the new centers are: " + Arrays.deepToString(tempNewCenters));

		// Step 3. Cluster and compute new centers.
		while (!doubleMatricesEqual(tempCenters, tempNewCenters)) {
			//while (!Arrays.deepEquals(tempCenters, tempNewCenters)) {
			tempCenters = tempNewCenters;
			// Cluster
			for (int i = 0; i < tempBlockSize; i++) {
				double tempDistance = Double.MAX_VALUE;
				for (int j = 0; j < paraK; j++) {
					double tempCurrentDistance = distance(paraBlock[i], tempCenters[j]);
					if (tempCurrentDistance < tempDistance) {
						tempCluster[i] = j;
						tempDistance = tempCurrentDistance;
					} // Of cluster
				} // Of for j
			} // Of for i

			System.out.println("Current cluster: " + Arrays.toString(tempCluster));

			// Compute new centers   count the number of  instances in different class
			int[] tempCounters = new int[paraK];
			for (int i = 0; i < tempCounters.length; i++) {
				tempCounters[i] = 0;
			} // Of for i

			//1. sum all in one kind
			//tempNewCenters = new double[paraK][data.numAttributes() - 1];  //why define tempNewCenter twice
			for (int i = 0; i < tempBlockSize; i++) {
				tempCounters[tempCluster[i]]++; //nice expect the center
				for (int j = 0; j < data.numAttributes() - 1; j++) {
					tempNewCenters[tempCluster[i]][j] += data.instance(paraBlock[i]).value(j); // include the center
				} // Of for j
			} // Of for i
			System.out.println("............tempNewCenters is " + Arrays.deepToString(tempNewCenters));
			System.out.println("            tempCounters " + Arrays.toString(tempCounters));
			//2. Average   Means  conclude the new centers
			for (int i = 0; i < paraK; i++) {
				for (int j = 0; j < data.numAttributes() - 1; j++) {
					tempNewCenters[i][j] /= tempCounters[i];
				} // Of for j
			} // Of for i

			currentCenters = tempNewCenters;
			System.out.println("----The currentCenters are" + Arrays.deepToString(currentCenters));
			System.out.println("-----The centers are: " + Arrays.deepToString(tempCenters));
			System.out.println("-----The new centers are: " + Arrays.deepToString(tempNewCenters));
		} // Of while

		return tempCluster;
	}// Of cluster


	/**
	 ***************
	 * Split in two and learn.
	 ***************
	 */
	public void splitAndLearn(int[] paraBlock) {
		// Step 1. Split
		//tempClustering is the array after 2-means clustering
		//System.out.println("------------This is splitAndLearn");
		int splitTimes = 1;
		System.out.println("The split time is " + splitTimes++);

		int[] tempClutering = cluster(paraBlock);
		int tempFirstBlockSize = 0;
		for (int i = 0; i < tempClutering.length; i++) {
			if (tempClutering[i] == 0) {
				tempFirstBlockSize++;
			} // Of if
		} // Of for i
			//after 2-means cluster generate 2 block and initialize the total array tempBlocks of 2 blocks
		int[][] tempBlocks = new int[2][];
		tempBlocks[0] = new int[tempFirstBlockSize];
		tempBlocks[1] = new int[paraBlock.length - tempFirstBlockSize];

		int[] tempCounters = new int[2];
		for (int i = 0; i < tempClutering.length; i++) {
			tempBlocks[tempClutering[i]][tempCounters[tempClutering[i]]++] = paraBlock[i];
		} // Of for i

		System.out.println("Splitted into two blocks: " + Arrays.toString(tempBlocks[0]) + "\r\n"
				+ Arrays.toString(tempBlocks[1]));

		// Step 2. Learn
		learn(tempBlocks[0]);
		learn(tempBlocks[1]);
	}// Of splitAndLearn

	/**
	 * cost compute cost sensitive 
	 * @return the sum of cost
	 */
	public double totalCost() {
		double cost = 0;
		for (int i = 0; i < data.numInstances(); i++) {
			if (instanceStates[i] == 1) {
				cost += tCost;
			} else {
				if (labels[i] == 0 && (int) data.instance(i).classValue() == 1) {
					cost += mCost[0];
				} else if (labels[i] == 1 && (int) data.instance(i).classValue() == 0) {
					cost += mCost[1];
				} // Of if
			}
		}
		return cost;
	}

	/**
	 * compute the number of buy 
	 * @param paraBlock the result of preLearn that is a array of data
	 * @param tempFirstLabel the first brought label
	 * @return the number to buy
	 */
	public int numToBuy(int[] paraBlock, int tempFirstLabel) {
		int[] tempNumBuys = lookup(paraBlock.length);// lookup find optimal R B
		int tempLabels = 0;
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStates[paraBlock[i]] == 1) {
				tempLabels++;
			} // Of if
		} // Of for i
		int tempBuyLabels = 0;
		if (tempFirstLabel == 0 || tempFirstLabel == 1) {
			tempBuyLabels = tempNumBuys[tempFirstLabel] - tempLabels;
		} else {//tempFirstLabel = -1;
			tempBuyLabels = Math.max(tempNumBuys[0], tempNumBuys[1]) - tempLabels;
		} // Of if
		return tempBuyLabels;
	}

	/**
	 ***************
	 * Find some representatives from given the instances.
	 ***************
	 */
	public int[] findRepresentatives(int[] paraBlock, int tempFirstLabel) {
		// Step 1. How many labels do we already bought? 
		int tempLabels = 0;
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStates[paraBlock[i]] == 1) {
				tempLabels++;
			} // Of if
		} // Of for i

		// Step 2. How many labels to buy?
		//int[] tempNumBuys = {0,0};
		int[] tempNumBuys = lookup(paraBlock.length); // lookup find optimal R B
		int tempBuyLabels = 0;
		if (tempFirstLabel == 0 || tempFirstLabel == 1) {
			tempBuyLabels = tempNumBuys[tempFirstLabel] - tempLabels;
		} else {//tempFirstLabel = -1;
			tempBuyLabels = Math.max(tempNumBuys[0], tempNumBuys[1]) - tempLabels;
		} // Of if

		//int tempBuyLabels = pureThreshold(paraBlock.length) - tempLabels;
		if (pureThreshold(paraBlock.length) - tempBuyLabels > 0) {
			System.out.println("+");
		} else if (pureThreshold(paraBlock.length) - tempBuyLabels < 0) {
			System.out.println("-");
		}
		if (tempBuyLabels <= 0) {
			System.out.println("Intend to buy " + tempLabels + " however at most " + pureThreshold(paraBlock.length));
			return null;
		} // Of if

		// Step 3. Cluster   Re-cluster   buy labels is as K  recalculate the currentCenters
		cluster(paraBlock);

		// Step 4. Find representatives, the point nearest the center. 
		//Step 4. Find representatives, the point furthest the 
		//find the representatives in the clustered block with tempBuyLabels centers 
		int[] tempRepresentatives = new int[tempBuyLabels];
		for (int i = 0; i < tempBuyLabels; i++) {
			double tempMinimalDistance = Double.MAX_VALUE;
			for (int j = 0; j < paraBlock.length; j++) {
				if (instanceStates[paraBlock[j]] == 1) {
					continue;
				} // Of if
					// the representatives data is minimal distance of center and block
				double tempDistance = distance(paraBlock[j], currentCenters[i]);
				if (tempDistance < tempMinimalDistance) {
					tempMinimalDistance = tempDistance;
					tempRepresentatives[i] = paraBlock[j];
				} // Of if
			} // Of for j
		} // Of for i

		//System.out.println("Representatives of this block: " + Arrays.toString(tempRepresentatives));

		return tempRepresentatives;
	}// Of findRepresentatives

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
	 * find querid instances from the block  
	 * @param paraBlock the given block
	 * @return the index of querid instances
	 */
	public int[] findQueriedInstances(int[] paraBlock) {
		int tempIndex = 0; // the index of the paraBlock
		int[] tempQueriedInstances = new int[paraBlock.length];
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStates[i] == 1) {
				tempQueriedInstances[tempIndex] = paraBlock[i];
				tempIndex++;
			} //of if
		} //of for i
		System.out.println("The input array is " + Arrays.toString(paraBlock));
		System.out.println("The queried array is: " + Arrays.toString(tempQueriedInstances));
		return tempQueriedInstances;
	}// of findQueriedInstances

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
				} // Of if

				tempDistanceSum += manhattanDistance(paraCurrentBlock[i], paraLabeledInstances[j]);
			} // Of for j

			System.out.println("" + paraCurrentBlock[i] + " to labeled = " + tempDistanceSum);

			// Update
			if (tempDistanceSum > tempMaxDistanceSum + 1e-6) {
				resultFarthest = paraCurrentBlock[i];
				tempMaxDistanceSum = tempDistanceSum;
			} // Of if
		} // Of for i
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
	public double manhattanDistance(int paraFirstIndex, int paraSecondIndex) {
		double resultDistance = 0;
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			resultDistance += Math
					.abs(data.instance(paraFirstIndex).value(i) - data.instance(paraSecondIndex).value(i));
		} // Of for i
		return resultDistance;
	}// Of manhattanDistance

	/**
	 ********************* 
	 * Compute the expect number of positive instances.
	 * 
	 * @param R
	 *            the number of positive instances checked.
	 * @param B
	 *            the number of negative instances checked.
	 * @param N
	 *            the total number of instances.
	 * @return the expect number of positive instances.
	 * 
	 * CADU equation 4
	 ********************* 
	 */
	public static double expectPosNum(int R, int B, int N) {
		BigDecimal fenzi = new BigDecimal("0");
		BigDecimal fenmu = new BigDecimal("0");
		for (int i = R; i <= N - B; i++) {
			BigDecimal a = A(R, i).multiply(A(B, N - i));
			fenzi = fenzi.add(a.multiply(new BigDecimal("" + i)));
			fenmu = fenmu.add(a);
			//System.out.println("a: " + a + ", fenzi: " + fenzi + ", fenmu: " + fenmu );
		} // Of for i
		return fenzi.divide(fenmu, 4, BigDecimal.ROUND_HALF_EVEN).doubleValue();
	}// Of expectPosNum

	/**
	 ********************* 
	 * Compute arrangement of A^m_n where m <= B
	 * 
	 *compute the  permutation 
	 ********************* 
	 */
	public static BigDecimal A(int m, int n) {
		if (m > n) {
			return new BigDecimal("0");
		} // Of if
		BigDecimal re = new BigDecimal("1");
		for (int i = n - m + 1; i <= n; i++) {
			re = re.multiply(new BigDecimal(i));
		} // Of if
		return re;
	}// Of A

	/**
	 ***************
	 * Is the given matrices equal?
	 * Judge the center and the new center is equal?
	 ***************
	 */
	public static boolean doubleMatricesEqual(double[][] paraMatrix1, double[][] paraMatrix2) {
		for (int i = 0; i < paraMatrix1.length; i++) { //the number of line
			for (int j = 0; j < paraMatrix1[0].length; j++) { // the number of elements in a line
				if (Math.abs(paraMatrix1[i][j] - paraMatrix2[i][j]) > 1e-6) { // the precision is 10^-6
					return false;
				} // Of if
			} // Of for j
		} // Of for i
		return true;
	}// Of doubleMatricesEqual

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
	 ***************
	 * With how many instances with the same label can we say it is pure?
	 * suppose the threshold is sqrt(N)
	 ***************
	 */
	public int pureThreshold(int paraSize) {
		return (int) Math.sqrt(paraSize);
	}// Of pureThreshold

	/**
	 * overwrite the to string method
	 * @param paraArray
	 * @return
	 */
	public static String intArrayToString(int[] paraArray) {
		String tempString = "";
		for (int i = 0; i < paraArray.length - 1; i++) {
			tempString += paraArray[i] + ",";
		} // Of for i

		tempString += paraArray[paraArray.length - 1];

		return tempString;
	}// Of intArrayToString

}// Of class KMeanActive

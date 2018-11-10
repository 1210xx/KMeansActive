package Classification;

import java.io.FileReader;
import java.math.BigDecimal;
import java.util.Arrays;
import java.util.Random;
import java.util.function.IntPredicate;

import weka.*;
import weka.core.Instances;


public class RJ_CAKU {
	
	/*
	 * 0 represents untreated
	 * 1 represents query 
	 * 2 represents classify
	 */
	int [] instanceStatus;
	
	Instances data;
	
	double[][] currentCenters;
	
	int[] labels;
	
	double[]  mCost;
	
	double tCost;
	
	static Random random = new Random();
	
	/**
	 * the constructor to read the data and initialize array
	 */
	public RJ_CAKU(String paraFilename) {
		// TODO Auto-generated constructor stub
		data = null;
		try {
			FileReader fileReader = new FileReader(paraFilename);
			data = new Instances(fileReader);
			fileReader.close();
			data.setClassIndex(data.numAttributes()-1);
//			System.out.println("data.instance[data.numAttributes - 1]:  " + data.instance(1).value(data.numAttributes() - 1));
		} catch (Exception ee) {
			System.out.println("Cannot read the file: " + paraFilename + "\r\n" + ee);
			System.exit(0);
		} // Of try
		// Initialize
		instanceStatus = new int[data.numInstances()];
		Arrays.fill(instanceStatus, 0);
		labels = new int[data.numInstances()];
		Arrays.fill(labels, -1);
	}

	/**
	 ***************
	 * Learning test.
	 ***************
	 */
	void preLearning() {
		int[] originalBlock = new int[data.numInstances()];
		for (int i = 0; i < originalBlock.length; i++) {
			originalBlock[i] = i;
		} // Of for i
		querySplitClassify(originalBlock,originalBlock[0]);
		System.out.println("instanceStatus: " + Arrays.toString(instanceStatus));
		System.out.println("labels: " + Arrays.toString(labels));
	}// Of learningTest
	
	
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
			labels[paraInitialPoint] =  (int) data.instance(paraInitialPoint).value(data.numAttributes()-1);
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
	 ***************
	 * Cluster. 2-Means Cluster split 
	 ***************
	 */
	public int[] cluster(int[] paraBlock,int paraK) {
		//System.out.println("----------clustering K-Means");
		//System.out.println("----------paraK = " + paraK);
		//System.out.println("+++++++++++BlockSize = " + paraBlock.length);
		// Step 1. Initialize
		paraK = 2;
		int tempBlockSize = paraBlock.length;
		int[] tempCluster = new int[tempBlockSize];
		double[][] tempCenters = new double[paraK][data.numAttributes() - 1];
		double[][] tempNewCenters = new double[paraK][data.numAttributes() - 1];
		int[] tempQueriedInstances = new int[tempBlockSize];
 		//System.out.println("paraK :" + paraK);
		//System.out.println("tempBlocksize " + tempBlockSize);
		//System.out.println("The current index is: " + tempIndex);
		// Step 2. Randomly select 1 data points.
		int tempFixedIndex = paraBlock[0];
		//int tempIndex = random.nextInt(tempBlockSize);
		int tempFirstCenter = tempFixedIndex;	
		int tempSecondCenter = findFarthest(tempQueriedInstances,paraBlock);
		int[] tempAllIndex = {tempFirstCenter,tempSecondCenter};
		for (int i = 0; i < instanceStatus.length; i++) {
			if (instanceStatus[i] == 1) {
				for (int j = 0; j < instanceStatus.length; j++) {
					tempQueriedInstances[j] = i;
				}
 			}
		}
				
		for (int i = 0; i < paraK; i++) {
			for (int j = 0; j < data.numAttributes() - 1; j++) {
				tempNewCenters[i][j] = data.instance(paraBlock[tempAllIndex[paraK]]).value(j);
			} // Of for j
		} // Of for i
			//System.out.println("Randomly selection: the new centers are: " + Arrays.deepToString(tempNewCenters));
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
			//System.out.println("Current cluster: " + Arrays.toString(tempCluster));
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
				//System.out.println("............tempNewCenters is " + Arrays.deepToStringr(tempNewCenters));
				//System.out.println("            tempCounters " + Arrays.toString(tempCounters));
				//2. Average   Means  conclude the new centers
			for (int i = 0; i < paraK; i++) {
				for (int j = 0; j < data.numAttributes() - 1; j++) {
					tempNewCenters[i][j] /= tempCounters[i];
				} // Of for j
			} // Of for i
	
			currentCenters = tempNewCenters;
			//System.out.println("----The currentCenters are" + Arrays.deepToString(currentCenters));
			//System.out.println("-----The centers are: " + Arrays.deepToString(tempCenters));
			//System.out.println("-----The new centers are: " + Arrays.deepToString(tempNewCenters));
		} // Of while
		return tempCluster;
	}// Of cluster

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

				tempDistanceSum += manhattanDistance(paraCurrentBlock[i],paraLabeledInstances[j]);
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
	public double manhattanDistance(int paraFirstIndex, int paraSecondIndex) {
		double resultDistance = 0;
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			resultDistance += Math.abs(data.instance(paraFirstIndex).value(i) - data.instance(paraSecondIndex).value(i));
		}// Of for i
		
	/**
	 * cost compute cost sensitive 
	 * @return 
	 */
	public double totalCost() {
		double cost = 0;
		for (int i = 0; i < data.numInstances(); i++) {
			if (instanceStatus[i] == 1) {
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
	
	public static String intArrayToString(int[] paraArray) {
		String tempString = "";
		for (int i = 0; i < paraArray.length - 1; i++) {
			tempString += paraArray[i] + ",";
		} // Of for i

		tempString += paraArray[paraArray.length - 1];

		return tempString;
	}// Of intArrayToString
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

}

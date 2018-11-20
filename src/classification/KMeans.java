package Classification;

import java.util.Arrays;
//
//import javax.swing.text.Highlighter;
//
//import weka.datagenerators.Test;


//import classfication.KNN;

/**
 ************************************
 * Top-10 data mining algorithms. SMALE-http://scs.swpu.edu.cn/smale A k-Means
 * algorithm for clustering.
 * 
 * @author Fan Min minfanphd@163.com 2015/07/09
 ************************************
 */

public class KMeans {
	/**
	 * A dataset for simple testing. Real datasets should be read from file.
	 */
	public static double[][] simpleDataset =
		{ 
			{ 5.1, 3.5, 1.4, 0.2 },//0
			{ 4.9, 3.0, 1.4, 0.2 },//1
			{ 4.7, 3.2, 1.3, 0.2 },//2
			{ 4.6, 3.1, 1.5, 0.2 },//3
			{ 5.0, 3.6, 1.4, 0.2 },//4
			{ 7.0, 3.2, 4.7, 1.4 },//5
			{ 6.4, 3.2, 4.5, 1.5 },//6
			{ 6.9, 3.1, 4.9, 1.5 },//7
			{ 5.5, 2.3, 4.0, 1.3 },//8
			{ 6.5, 2.8, 4.6, 1.5 },//9
			{ 5.7, 2.8, 4.5, 1.3 },//10
			{ 6.5, 3.0, 5.8, 2.2 },//11
			{ 7.6, 3.0, 6.6, 2.1 },//12
			{ 4.9, 2.5, 4.5, 1.7 },//13
			{ 7.3, 2.9, 6.3, 1.8 },//14
			{ 6.7, 2.5, 5.8, 1.8 },//15
			{ 6.9, 3.1, 5.1, 2.3 } //16
			};

	/**
	 * A dataset for testing.
	 */
	double[][] dataset;

	/**
	 * The k value.
	 */
	int k;

	/**
	 * Which cluster does a point belong to.
	 */
	int[] clusterArray;

	/**
	 * Centers.
	 */
	double[][] centers;
	
	//count how times the cluster  was called 
	int times = 1;

	/**
	 ************************************
	 * The constructor.
	 ************************************
	 */
	public KMeans(double[][] paraDataset, int paraK) {
		dataset = paraDataset;
		k = paraK;
	}// Of the constructor

	/**
	 ************************************
	 * Cluster using the centers stored in the object.
	 ************************************
	 */
	public int[] cluster() {
		return cluster(centers);
	}// Of cluster

	/**
	 ************************************
	 * Cluster. The result is stored in clusterArray.
	 ************************************
	 */
	public int[] cluster(double[][] paraCenters) {
		double tempLeastDistance;
		double tempDistance;
		int tempIndex = 0;
		clusterArray = new int[dataset.length];

		for (int i = 0; i < dataset.length; i++) {
			tempLeastDistance = Double.MAX_VALUE;
			for (int j = 0; j < paraCenters.length; j++) {
				tempDistance = distance(dataset[i], paraCenters[j]);
				if (tempDistance < tempLeastDistance) {
					tempLeastDistance = tempDistance;
					tempIndex = j;
				}// Of if
			}// Of for j
			clusterArray[i] = tempIndex;
		}// Of for i

		return clusterArray;
	}// Of cluster

	/**
	 ************************************
	 * Compute the center of a number of points.
	 ************************************
	 */
	public double[][] computeCenters() {
		centers = new double[k][];
		for (int i = 0; i < k; i++) {
			centers[i] = computeCenter(i);
		}// Of for i
		return centers;
	}// Of computeCenters

	/**
	 ************************************
	 * Compute the center of a number of points.
	 * the only one point can compute
	 * @param
	 * paraIndex
	 * the cluster ----k
	 ************************************
	 */
	public double[] computeCenter(int paraIndex) {
		//test which time to call this method
		//System.out.println("this is " + times++ + " times to call computeCenter");
		
		double[] tempCenter = new double[dataset[0].length];
		// Step 1. How many points are there in this cluster?
		int tempValidPoints = 0;
		for (int i = 0; i < clusterArray.length; i++) {
			if (clusterArray[i] == paraIndex) {
				tempValidPoints++;
			}// Of if
		}// Of for i

		// Step 2. Compute the total values.
		for (int i = 0; i < clusterArray.length; i++) {
			//System.out.println("***********");
			if (clusterArray[i] != paraIndex) {
				//System.out.println("============" + clusterArray[i]);
				continue;
			}// Of if

			for (int j = 0; j < dataset[0].length; j++) {
				tempCenter[j] += dataset[i][j];
			}// Of for j
		}// Of for i

		// Step 3. Compute the average values.
		for (int i = 0; i < dataset[0].length; i++) {
			tempCenter[i] /= tempValidPoints;
		}// Of for j

		return tempCenter;
	}// Of computeCenter

	/**
	 ************************************
	 * The distance between two points in the space.
	 * 
	 * @param paraFirstPoint
	 *            the first point.
	 * @param paraSecondPoint
	 *            the second point.
	 ************************************
	 */
	public double distance(double[] paraFirstPoint, double[] paraSecondPoint) {
		double distance = 0;
		for (int i = 0; i < paraFirstPoint.length; i++) {
			distance += Math.abs(paraFirstPoint[i] - paraSecondPoint[i]);
		}// Of for i
		//System.out.println("The disntance is " + distance);
		return distance;
	}// Of distance

	/**
	 ************************************
	 * Set the centers.
	 ************************************
	 */
	public void setCenters(double[][] paraCenters) {
		centers = paraCenters;
	}// Of setCenters

	/**
	 ************************************
	 * Set the centers.
	 ************************************
	 */
	public void setCenters(int[] paraCenterIndices) {
		centers = new double[k][dataset[0].length];
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < dataset[0].length; j++) {
				centers[i][j] = dataset[paraCenterIndices[i]][j];
			}// Of for j
		}// Of for i
	}// Of setCenters

	/**
	 ************************************
	 * The distance between two points in the space.
	 * 
	 * @param paraPoints
	 *            the given two points in the space.
	 ************************************
	 */
	public double distance(double[][] paraPoints) {
		double distance = 0;
		for (int i = 0; i < paraPoints.length; i++) {
			distance += Math.abs(paraPoints[0][i] - paraPoints[1][i]);
		}// Of for i

		return distance;
	}// Of distance

	/**
	 ************************************
	 * The distance between two points in the dataset.
	 * 
	 * @param paraFirstPoint
	 *            the index of the first point in the dataset.
	 * @param paraSecondPoint
	 *            the index of the second point in the dataset.
	 ************************************
	 */
	public double distance(int paraFirstPoint, int paraSecondPoint) {
		double distance = 0;
		for (int i = 0; i < dataset[0].length; i++) {
			distance += Math.abs(dataset[paraFirstPoint][i] - dataset[paraSecondPoint][i]);
		}// Of for i

		return distance;
	}// Of distance

	/**
	 ************************************
	 * Are two int array equal.
	 ************************************
	 */
	public static boolean intArrayEqual(int[] paraFirstArray,int[] paraSecondArray) {
		for (int i = 0; i < paraFirstArray.length; i++) {
			if (paraFirstArray[i] != paraSecondArray[i]) {
				return false;
			}// Of if
		}// Of for i

		return true;
	}// Of intArrayEqual

	/**
	 ************************************
	 * Convert an int array to a string
	 ************************************
	 */
	public static String intArrayToString(int[] paraArray) {
		String tempString = "";
		for (int i = 0; i < paraArray.length - 1; i++) {
			tempString += paraArray[i] + ",";
		}// Of for i

		tempString += paraArray[paraArray.length - 1];

		return tempString;
	}// Of intArrayToString

	/**
	 ************************************
	 * Compute the distance sum.
	 ************************************
	 */
	public double computeDistanceSum() {
		double tempDistanceSum = 0;
		for (int i = 0; i < dataset.length; i ++) {
			tempDistanceSum += distance(centers[clusterArray[i]], dataset[i]);
		}//Of for i
		
		return tempDistanceSum;
	}// Of computeDistanceSum
	
	/**
	 ************************************
	 * Express myself.
	 ************************************
	 */
	public String toString() {
		String tempString = "This is a K-Means algorithm.\r\n" + "There are "
				+ k + " centers. \r\n"
				+ "The cluster information is as follows:\r\n"
				+ intArrayToString(clusterArray);

		return tempString;
	}// Of toString

	/**
	 ************************************
	 * Test the class
	 ************************************
	 */
	public static void main(String args[]) {
		KMeans tempKMeans = new KMeans(simpleDataset, 2);
		int[] tempCenters = { 0, 7 };
		tempKMeans.setCenters(tempCenters);
		
		int[] tempPreviousCluster = new int[simpleDataset.length];
		int[] tempCurrentCluster = new int[simpleDataset.length];
		double tempTotalDistance;

		while (true) {
			tempCurrentCluster = tempKMeans.cluster();
			System.out.println(tempKMeans);
			if (intArrayEqual(tempCurrentCluster, tempPreviousCluster)) {
				break;
			}// Of if
			else {
			tempTotalDistance = tempKMeans.computeDistanceSum();
			System.out.println("The current total distance is: " + tempTotalDistance);

			tempPreviousCluster = tempCurrentCluster;

			tempKMeans.computeCenters();
			}
		}// Of while
		
		tempTotalDistance = tempKMeans.computeDistanceSum();
		
		System.out.println("The total distance is: " + tempTotalDistance);
		//test();
	}// Of main
	
	
	static void test() {
		System.out.println(Math.abs(-0.2));
		System.out.println(Math.abs(0.8));
		System.out.println(Arrays.deepToString(simpleDataset));
		int[][] testArray = new int [4][3];
		System.out.println("the length of the whole array "  + testArray.length);  // this is the row of the array
		System.out.println("the length od the line of the array " + testArray[0].length); // this is collum of array 
	
 	}
}// Of class KMeans

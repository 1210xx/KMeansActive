package Classification;

import java.util.Arrays;

public class KMeans_Rj {
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
		
	int K;
	double[][] DataSet;
	int[] clusterArray;
	double[][] centers;
	public KMeans_Rj() {
		// TODO Auto-generated constructor stub
	}
	public KMeans_Rj(double[][] paraDataSet, int paraK) {
		K = paraK;
		DataSet = paraDataSet;
	}
	
	public int[] cluster() {
		return cluster(centers);
	}
	public int[] cluster(double[][] paraCenters) {
		double tempLeastDistance;
		double tempDistance ;
		int tempIndex = 0;
		clusterArray = new int [DataSet.length];
		
		for (int i = 0; i < DataSet.length; i++) {
			tempLeastDistance = Double.MAX_VALUE;
			for (int j = 0; j < paraCenters.length; j++) {
				tempDistance = distance(DataSet[i],paraCenters[j]);
				if (tempDistance < tempLeastDistance) {
					tempLeastDistance = tempDistance;
					tempIndex = j;
				}
				clusterArray[i] = tempIndex;
			}
		}
		return clusterArray;
	}
	public double[][] computeCenter(){
		centers = new double [K][];
		for (int i = 0; i < K; i++) {
			centers[i] = computerCenter(i);
		}
		return centers;
	}
	
	public double[] computerCenter(int paraIndex) {
		int tempValidPoint = 0;
		double[] tempCenter = new double[DataSet[0].length];
		//total valid point
		for (int i = 0; i < clusterArray.length; i++) {
			if (clusterArray[i] == paraIndex) {
				tempValidPoint++;
			}
		}
		System.out.println("valid point is" + tempValidPoint);
		//
		for (int i = 0; i < clusterArray.length; i++) {
			if (clusterArray[i] != paraIndex) {
				continue;
			}
			for (int j = 0; j < DataSet[0].length; j++) {
				tempCenter[j] += DataSet[i][j];
			}
		}
		for (int i = 0; i < DataSet[0].length; i++) {
			tempCenter[i] /= tempValidPoint;
		}
		return tempCenter;
	}
	public double distance(double[] paraFirstPoint, double[] paraSecondPoint) {
		double distance = 0;
		for (int i = 0; i < paraFirstPoint.length; i++) {
			distance += Math.abs(paraFirstPoint[i] - paraSecondPoint[i]);
		}
		return distance;
	}
	public void setCenter(double[][] tempCenters) {
		// TODO Auto-generated method stub
		centers = tempCenters;
	}
	public  void setCenter(int[] paraIndices) {
		centers = new double[K][DataSet[0].length];
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < DataSet[0].length; j++) {
				centers[i][j] = DataSet[paraIndices[i]][j];
			}
		}
	}
	public double computeDistanceSum() {
		double tempDistanceSum = 0;
		for (int i = 0; i < DataSet.length; i++) {
			System.out.println("the center["+ clusterArray[i] +"] is " + KMeans_Rj.doubleArrayToString(centers[clusterArray[i]]));
			System.out.println("the dataset["+ i +"]is :" + KMeans_Rj.doubleArrayToString(DataSet[i]));
			System.out.println("The distance between centers[clusterArray[i]] ans dataset[i] is "+ distance(centers[clusterArray[i]], DataSet[i]));
			tempDistanceSum += distance(centers[clusterArray[i]], DataSet[i]);
		}
		return tempDistanceSum;
	}
	private static boolean intArrayEqual(int[] paraFirstArray, int[] paraSecondArray) {
		// TODO Auto-generated method stub
		for (int i = 0; i < paraFirstArray.length; i++) {
			if (paraFirstArray[i] != paraSecondArray[i]) {
				return false;
			}
		}
		return true;
	}
	
	public static String intArrayToString(int[] paraArray) {
		String tempString = "";
		for (int i = 0; i < paraArray.length - 1; i++) {
			tempString += paraArray[i] + ",";
		}// Of for i

		tempString += paraArray[paraArray.length - 1];

		return tempString;
	}// Of intArrayToString
	public static String doubleArrayToString(double[] paraArray) {
		String tempString = "";
		for (int i = 0; i < paraArray.length - 1; i++) {
			 tempString += paraArray[i] + ",";
		}
		tempString += paraArray[paraArray.length - 1];
		return tempString;
	}
	public String toString() {
		String tempString = "This is a K-Means algorithm.\r\n" + "There are "
				+ K + " centers. \r\n"
				+ "The cluster information is as follows:\r\n"
				+ intArrayToString(clusterArray);

		return tempString;
	}// Of toString

	public static void main(String args[]) {
		KMeans_Rj kMeans = new KMeans_Rj(simpleDataset,2);
		int[] tempCenters = { 0, 7 };
		kMeans.setCenter(tempCenters);
		int[] tempCurrentCluster = new int[simpleDataset.length];
		int[] tempPreviousCluster = new int[simpleDataset.length];
		double tempTotalDistance;
		int times = 0;
		while (true) {
			System.out.println(times++);
			tempCurrentCluster = kMeans.cluster();
			System.out.println(kMeans);
			if (intArrayEqual(tempCurrentCluster,tempPreviousCluster)) {
				System.out.println("+++++");
				break;
			}
			else {
				tempTotalDistance = kMeans.computeDistanceSum();
				System.out.println("THe current total distance is: " + tempTotalDistance);
				tempPreviousCluster = tempCurrentCluster;
				kMeans.computeCenter();
				System.out.println("_____");
			}
		}
		tempTotalDistance = kMeans.computeDistanceSum();
		System.out.println("The total distance is: " + tempTotalDistance);
	}



}

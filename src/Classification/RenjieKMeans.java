package Classification;

import java.util.Arrays;

public class RenjieKMeans {
	public static double[][] simpleDataset = { 
			{ 5.1, 3.5, 1.4, 0.2 }, //0
			{ 4.9, 3.0, 1.4, 0.2 }, //1
			{ 4.7, 3.2, 1.3, 0.2 }, //2
			{ 4.6, 3.1, 1.5, 0.2 }, //3
			{ 5.0, 3.6, 1.4, 0.2 }, //4
			{ 7.0, 3.2, 4.7, 1.4 }, //5
			{ 6.4, 3.2, 4.5, 1.5 }, //6
			{ 6.9, 3.1, 4.9, 1.5 }, //7
			{ 5.5, 2.3, 4.0, 1.3 }, //8
			{ 6.5, 2.8, 4.6, 1.5 }, //9
			{ 5.7, 2.8, 4.5, 1.3 }, //10
			{ 6.5, 3.0, 5.8, 2.2 }, //11
			{ 7.6, 3.0, 6.6, 2.1 }, //12
			{ 4.9, 2.5, 4.5, 1.7 }, //13
			{ 7.3, 2.9, 6.3, 1.8 }, //14
			{ 6.7, 2.5, 5.8, 1.8 }, //15
			{ 6.9, 3.1, 5.1, 2.3 } //16
	};
	int k;
	int[] clusterArray ;
	double[][] dataset;
	double[][] centers;

	public RenjieKMeans(int paraK, double[][] paraData) {
		// TODO Auto-generated constructor stub
		k = paraK;
		dataset = paraData;
		//System.out.println("The data set is:\r\n" + Arrays.deepToString(dataset));
	}//Of the constructor

	public void setCenters(double[][] paraCenters) {
		centers = paraCenters;
	}//of setCenters

	public void setCenters(int[] paraIndex) {
		centers = new double[k][dataset[0].length];
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < dataset[0].length; j++) {
				centers[i][j] = dataset[paraIndex[i]][j];
			}//of j
		}//of i
	}//of setCenters

	public double distance(double[] paraFirstPoint, double[] paraSencondPoint) {
		double tempDistance = 0;
		for (int i = 0; i < paraSencondPoint.length; i++) {
			tempDistance += Math.abs(paraFirstPoint[i] - paraSencondPoint[i]);
		}//of for
		return tempDistance;
	}//of distance

	public int[] cluster() {
		return cluster(centers);
	}//of cluster

	public int[] cluster(double[][] paraCenter) {
		double tempLeastDistance;
		double tempDistance;
		clusterArray = new int [dataset.length];
		int Index = 0;
		for (int i = 0; i < dataset.length; i++) {
			tempLeastDistance = Double.MAX_VALUE;
			for (int j = 0; j < paraCenter.length; j++) {
				tempDistance = distance(dataset[i], paraCenter[j]);
				if (tempDistance < tempLeastDistance) {
					tempLeastDistance = tempDistance;
					Index = j;
				}// of if
				clusterArray[i] = Index;
			}//of j

		}//of i
		return clusterArray;
	}//of cluster

	public double[][] computerCenters() {
		centers = new double[k][];
		for (int i = 0; i < k; i++) {
			centers[i] = computerCenters(i);
		}//of i
		return centers;
	}//of computerCenters

	public double[] computerCenters(int paraIndex) {
		double[] tempCenter = new double[dataset[0].length];
		int tempValidPoints = 0;
		for (int i = 0; i < clusterArray.length; i++) {
			if (clusterArray[i] == paraIndex) {
				tempValidPoints++;		
			}//of if
		}//of For
		System.out.println("valid point is" + tempValidPoints);
		for (int i = 0; i < clusterArray.length; i++) {
			if (clusterArray[i] != paraIndex) {
				continue;
			}//of if
			for (int j = 0; j < dataset[0].length; j++) {
				tempCenter[j] += dataset[i][j];
			}//of j
		}//of i
		for (int i = 0; i < dataset[0].length; i++) {
			tempCenter[i] /= tempValidPoints;
		}//of for

		return tempCenter;
	}//of computerCenters

	private static boolean intArrayEqual(int[] paraFirstArray, int[] paraSecondArray) {
		// TODO Auto-generated method stub
		for (int i = 0; i < paraSecondArray.length; i++) {
			if (paraFirstArray[i] != paraSecondArray[i]) {
				return false;
			}// of if
		}// of for
		return true;
	}// of intAttayEqual

	public static String intArrayToString(int[] paraArray) {
		String tempString = "";
		for (int i = 0; i < paraArray.length - 1; i++) {
			tempString += paraArray[i] + ",";
		}//of i

		tempString += paraArray[paraArray.length - 1];

		return tempString;
	}// of ineAttayToSting

	public String toString() {
		String tempString = "This is a K-Means algorithm.\r\n" + "There are " + k + " centers. \r\n"
				+ "The cluster information is as follows:\r\n" + intArrayToString(clusterArray) + "\rThe centers is: \r\n" + Arrays.deepToString(centers);
		return tempString;
	}//of toString
	
	public static String doubleToString(double[] paraArray) {
		String tempString = ""; 
		for (int i = 0; i < paraArray.length - 1; i++) {
			 tempString += paraArray[i] + ",";
		}//of for
		tempString += paraArray[paraArray.length - 1];
		return tempString;
	}//of doubleToString
	private double computeDistanceSum() {
		// TODO Auto-generated method stub

		double tempDistanceSum = 0;
		for (int i = 0; i < dataset.length; i++) {
			System.out.println("the center["+ clusterArray[i] +"] is " + RenjieKMeans.doubleToString(centers[clusterArray[i]]));
			System.out.println("the dataset["+ i +"]is :" + RenjieKMeans.doubleToString(dataset[i]));
			System.out.println("The distance between centers[clusterArray[i]] ans dataset[i]is "+ distance(centers[clusterArray[i]], dataset[i]));
			tempDistanceSum += distance(centers[clusterArray[i]], dataset[i]);
		}//of for
		return tempDistanceSum;
	}//of computerDistanceSum

	public static void main(String args[]) {
		RenjieKMeans kMeans = new RenjieKMeans(2, simpleDataset);
		int[] tempCenters = { 0, 7 };
		kMeans.setCenters(tempCenters);
//		double[] tempDataPoint1 = {1, 2, 3};
//		double[] tempDataPoint2 = {2, 3, 3};
//		double tempDistance = kMeans.distance(tempDataPoint1, tempDataPoint2);
//		System.out.println("The distance is: " + tempDistance);
		int times = 1;
		int equalT = 1;
		int notequalT = 1;
		int[] tempCurrentPoints = new int[simpleDataset.length];
		int[] tempPreviousoints = new int[simpleDataset.length];
		double tempAllDistance;
		while (true) {
			System.out.println("the time into while " + times++);
			tempCurrentPoints = kMeans.cluster();
			System.out.println(kMeans);
			if (intArrayEqual(tempCurrentPoints, tempPreviousoints)) {
				System.out.println("if");
				System.out.println("Equal times is " + equalT++);
				System.out.println("The equal cluster is :");
				System.out.println(RenjieKMeans.intArrayToString(tempCurrentPoints));
				break;
			}// of if
			else {
			System.out.println("else");
			System.out.println("Not equal times is " + notequalT++);
			tempAllDistance = kMeans.computeDistanceSum();
			tempPreviousoints = tempCurrentPoints;
			System.out.println("The current total distance is :" + tempAllDistance);
			kMeans.computerCenters();
			}//of else
			tempAllDistance = kMeans.computeDistanceSum();
			System.out.println("The total distance is: " + tempAllDistance);
		}//of while

	}//of main


}//of class

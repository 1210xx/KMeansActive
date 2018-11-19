package classification;

import java.io.FileReader;
import java.math.BigDecimal;
import java.util.*;
import weka.core.*;

public class KMeansDistance {

	static Random random = new Random();

	Instances data;

	int[] instanceStates; // 1 represents bought and 2 represents predicted.

	int[] labels;

	double[][] currentCenters;

	double tCost;

	double[] mCost;

	public static void main(String args[]) {

		double[] mCost = { 2, 4 };

		double avgCost = 0;
		// for (int i = 0; i < 20; i++)
		// {
		KMeansDistance tempLeaner = new KMeansDistance(
				"C:/Users/zhangshiming\\Desktop\\binaryclassdata_mat_and_arff\\arrrfff\\DCCC.arff",
				mCost, 1);
		// int[] tempBlock = {1, 4, 5, 6, 59, 121};
		// //System.out.println("The clustering result is: " +
		// Arrays.toString(tempLeaner.cluster(3, tempBlock)));
		tempLeaner.learningTest();
		avgCost += tempLeaner.totalCost();
		System.out.println("OK");
		System.out.println("onceOfCost" + avgCost);
		// }

		// System.out.println("20avgCost" + avgCost / 20);
	}// Of main

	public double totalCost() {
		double cost = 0;
		for (int i = 0; i < data.numInstances(); i++) {
			if (instanceStates[i] == 1) {
				cost += tCost;
			} else {
				if (labels[i] == 0 && (int) data.instance(i).classValue() == 1) {
					// ������� labels�����ʾ����ÿ��Instance���ֳ��ĸ���
					cost += mCost[0];
				} else if (labels[i] == 1
						&& (int) data.instance(i).classValue() == 0) {
					cost += mCost[1];
				}// Of if
			}//Of if
		}
		return cost;
	}

	/**
	 ************************* 
	 * Constructor.
	 * 
	 * @param paraFilename
	 *            The given file.
	 ************************* 
	 */
	public KMeansDistance(String paraFilename, double[] paraMisclassificationCosts, double paraTeacherCost) {
		data = null;
		try {
			FileReader fileReader = new FileReader(paraFilename);
			data = new Instances(fileReader);
			fileReader.close();
			data.setClassIndex(data.numAttributes() - 1);
			// System.out.println(data);
		} catch (Exception ee) {
			// System.out.println("Cannot read the file: " + paraFilename +
			// "\r\n" + ee);
			System.exit(0);
		} // Of try

		// Initialize
		instanceStates = new int[data.numInstances()];
		labels = new int[data.numInstances()];
		Arrays.fill(labels, -1);
		
		mCost = paraMisclassificationCosts;
		tCost = paraTeacherCost;
	}// Of the first constructor

	void learningTest() {
		int[] originalBlock = new int[data.numInstances()];// ��ʼ
		for (int i = 0; i < originalBlock.length; i++) {
			originalBlock[i] = i;
		} // Of for i

		learn(originalBlock);// �����ֻ��ԭ���ݵ��±�

		System.out
				.println("instanceStates: " + Arrays.toString(instanceStates));
		System.out.println("labels: " + Arrays.toString(labels));

	}// Of learningTest

	void learn(int[] paraBlock) {
		int blockSize = paraBlock.length;
		int firstLabel = -1;
		int IndexOfFirstLabel = -1;
		int secondLabel = -1;
		int IndexOfSecondLabel = -1;
		// ����С��5ֱ�ӹ���
		if (blockSize <= 5) {
			for (int i = 0; i < blockSize; i++) {
				instanceStates[paraBlock[i]] = 1;
				labels[paraBlock[i]] = (int) data.instance(paraBlock[i]).value(
						data.numAttributes() - 1);

			}
			return;
		}// of if

		// �ҵ���һ���Ѿ�����ĵ�
		for (int i = 0; i < blockSize; i++) {
			if (instanceStates[paraBlock[i]] == 1) {
				IndexOfFirstLabel = i;
				firstLabel = (int) data.instance(paraBlock[i]).value(
						data.numAttributes() - 1);
				break;
			}
		}

		if (IndexOfFirstLabel != -1) {
			for (int i = IndexOfFirstLabel + 1; i < blockSize; i++) {
				if (instanceStates[paraBlock[i]] == 1) {
					IndexOfSecondLabel = i;
					secondLabel = (int) data.instance(paraBlock[i]).value(
							data.numAttributes() - 1);
					if (secondLabel != firstLabel) {
						splitAndLearn(paraBlock, IndexOfSecondLabel,
								IndexOfFirstLabel);
						return;
					}
				}// of for �ҵ�������ǩ��ͬ�ľͷֿ�
			}
		}

		/**
		 * ���ڿ�ʼ����Ҫ����Ĵ������
		 */

		int needToBuyNum = findRepresentatives(paraBlock, firstLabel);
		List<Integer> pointIndex = new ArrayList<Integer>();

		if (IndexOfFirstLabel != -1)
			pointIndex.add(IndexOfFirstLabel);
		else {
			pointIndex.add(0);
			// ���ڿ�ʼ˳�����µ�һ�������
			instanceStates[paraBlock[pointIndex.get(0)]] = 1;
			labels[paraBlock[pointIndex.get(0)]] = (int) data.instance(
					paraBlock[pointIndex.get(0)]).value(
					data.numAttributes() - 1);
		}
		while (pointIndex.size() <= (needToBuyNum - 1)) {
			// �����ݼ��ĵ�һ����������ĵ�ĵ�һ��

			// �����ҵڶ���
			int zuiyuandian = computeFathestPoint(paraBlock, pointIndex);// ���ص���paraBlock�ڲ��±꣬��������ԭʼ�ı��
			// ������Զ��ı�ǩ
			instanceStates[paraBlock[zuiyuandian]] = 1;
			labels[paraBlock[zuiyuandian]] = (int) data.instance(
					paraBlock[zuiyuandian]).value(data.numAttributes() - 1);

			if (labels[paraBlock[zuiyuandian]] != labels[paraBlock[0]]) {
				splitAndLearn(paraBlock, pointIndex.get(0), zuiyuandian);
				return;
			} else
				pointIndex.add(zuiyuandian);
		}// of while whileѭ�����֤�����е�ȫ�����Ѿ��͵�һ����һ�� ֱ�ӽ���Ԥ��

		// Step 3. Predict others in this block. ����Ǹ��ݴ����ֱ�Ӹ���
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStates[paraBlock[i]] != 1) {
				instanceStates[paraBlock[i]] = 2;
				labels[paraBlock[i]] = labels[paraBlock[pointIndex.get(0)]];
			} // Of if
		} // Of for i
		return;

	}// of learn

	public void splitAndLearn(int[] paraBlock, int a, int b) {
		// Step 1. Split
		int[] tempClutering = cluster(a, b, paraBlock);
		int tempFirstBlockSize = 0;
		for (int i = 0; i < tempClutering.length; i++) {
			if (tempClutering[i] == 0) {
				tempFirstBlockSize++;
			} // Of if
		} // Of for i

		int[][] tempBlocks = new int[2][];
		tempBlocks[0] = new int[tempFirstBlockSize];
		tempBlocks[1] = new int[paraBlock.length - tempFirstBlockSize];

		int[] tempCounters = { 0, 0 };

		for (int i = 0; i < tempClutering.length; i++) {
			tempBlocks[tempClutering[i]][tempCounters[tempClutering[i]]++] = paraBlock[i];
		} // Of for i

		System.out.println("SplittedAndLearn into two blocks: "
				+ Arrays.toString(tempBlocks[0]) + "\r\n"
				+ Arrays.toString(tempBlocks[1]));

		// Step 2. Learn
		learn(tempBlocks[0]);
		learn(tempBlocks[1]);
	}

	public int[] cluster(int a, int b, int[] paraBlock) {
		// Step 1. Initialize
		int tempBlockSize = paraBlock.length;
		int[] tempCluster = new int[tempBlockSize];
		double[][] tempCenters = new double[2][data.numAttributes() - 1];
		double[][] tempNewCenters = new double[2][data.numAttributes() - 1];

		for (int j = 0; j < data.numAttributes() - 1; j++) {
			tempNewCenters[0][j] = data.instance(paraBlock[a]).value(j);
			tempNewCenters[1][j] = data.instance(paraBlock[b]).value(j);
		}
		// Step 3. Cluster and compute new centers.
		while (!doubleMatricesEqual(tempCenters, tempNewCenters)) {
			// while (!Arrays.deepEquals(tempCenters, tempNewCenters)) {
			tempCenters = tempNewCenters;
			// Cluster
			for (int i = 0; i < tempBlockSize; i++) {
				double tempDistance = Double.MAX_VALUE;
				for (int j = 0; j < 2; j++) {
					double tempCurrentDistance = distance(paraBlock[i],
							tempCenters[j]);
					if (tempCurrentDistance < tempDistance) {
						tempCluster[i] = j;// ����ÿ���������ĸ�����
						tempDistance = tempCurrentDistance;
					} // Of cluster
				} // Of for j
			} // Of for i

			// System.out.println("Current cluster: " +
			// Arrays.toString(tempCluster));

			// Compute new centers
			int[] tempCounters = new int[2];
			for (int i = 0; i < tempCounters.length; i++) {
				tempCounters[i] = 0;
			} // Of for i

			tempNewCenters = new double[2][data.numAttributes() - 1];
			for (int i = 0; i < tempBlockSize; i++) {
				tempCounters[tempCluster[i]]++;// ����ÿ�������м�����
				for (int j = 0; j < data.numAttributes() - 1; j++) {
					tempNewCenters[tempCluster[i]][j] += data.instance(
							paraBlock[i]).value(j);
				} // Of for j
			} // Of for i

			// Average
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < data.numAttributes() - 1; j++) {
					tempNewCenters[i][j] /= tempCounters[i];
				} // Of for j
			} // Of for i
			currentCenters = tempNewCenters;// ��������������㵽���ĵ��ƽ������

			// // System.out.println("The centers are: " +
			// Arrays.deepToString(tempCenters));
			// // System.out.println("The new centers are: " +
			// // Arrays.deepToString(tempNewCenters));
		} // Of while
		System.out.println("ԭ����cluster�����ĸ�����" + Arrays.toString(tempCluster));
		return tempCluster;
	}// Of cluster

	double computeDistence(int point, int paraBlockPoint) {
		double tempDistence = 0;
		for (int i = 0; i < data.numAttributes() - 1; i++) {
			tempDistence += Math.abs(data.instance(point).value(i)
					- data.instance(paraBlockPoint).value(i));
		}
		return tempDistence;
	}

	/**
	 * ���㵱ǰ���ݿ���Զ�����ݵ㣨�����������ĵ���Զ�ĵ㣩
	 */
	public int computeFathestPoint(int[] paraBlock, List<Integer> centerPoints) {
		int blockSize = paraBlock.length;
		int listSize = centerPoints.size();
		int tempIndexOfMaxPoint = 0;
		int[] isFinded = new int[blockSize];
		for (int i = 0; i < listSize; i++) {
			isFinded[centerPoints.get(i)] = 1;
		}
		double tempMaxDistence = 0;
		for (int i = 0; i < blockSize; i++) {

			double tempDistence = 0;
			for (int j = 0; j < listSize; j++) {
				tempDistence += computeDistence(paraBlock[centerPoints.get(j)],
						paraBlock[i]);

			}

			if ((tempDistence > tempMaxDistence) && (isFinded[i] != 1)) {
				tempIndexOfMaxPoint = i;
				tempMaxDistence = tempDistence;
			}
		}

		System.out.println("��Զ�����������֪����Զ�����������е�������"
				+ paraBlock[tempIndexOfMaxPoint]);
		return tempIndexOfMaxPoint;
	}

	/**
	 ***************
	 * With how many instances with the same label can we say it is pure?
	 ***************
	 */
	public int pureThreshold(int paraSize) {
		return (int) Math.sqrt(paraSize);
	}// Of pureThreshold

	public static boolean doubleMatricesEqual(double[][] paraMatrix1,
			double[][] paraMatrix2) {
		for (int i = 0; i < paraMatrix1.length; i++) {
			for (int j = 0; j < paraMatrix1[0].length; j++) {
				if (Math.abs(paraMatrix1[i][j] - paraMatrix2[i][j]) > 1e-6) {
					return false;
				} // Of if
			} // Of for j
		} // Of for i
		return true;
	}// Of doubleMatricesEqual

	public static double expectPosNum(int R, int B, int N) {
		BigDecimal fenzi = new BigDecimal("0");
		BigDecimal fenmu = new BigDecimal("0");
		for (int i = R; i <= N - B; i++) {
			BigDecimal a = A(R, i).multiply(A(B, N - i));
			fenzi = fenzi.add(a.multiply(new BigDecimal("" + i)));
			fenmu = fenmu.add(a);
			// System.out.println("fenzi:" + fenzi + ", fenmu: " + fenmu);
		} // Of for i
		return fenzi.divide(fenmu, 4, BigDecimal.ROUND_HALF_EVEN).doubleValue();
	}// Of expectPosNum

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

	public double distance(int paraIndex, double[] paraArray) {
		double resultDistance = 0;
		for (int i = 0; i < paraArray.length; i++) {
			resultDistance += Math.abs(data.instance(paraIndex).value(i)
					- paraArray[i]);
		} // Of for i
			// System.out.println(resultDistance);
		return resultDistance;
	}// Of distance

	/**
	 ***************
	 * Find some representatives given the instances.
	 ***************
	 */
	public int findRepresentatives(int[] paraBlock, int tempFirstLabel) {
		// Step 1. How many labels do we already bought?
		int tempLabels = 0;
		for (int i = 0; i < paraBlock.length; i++) {
			if (instanceStates[paraBlock[i]] == 1) {
				tempLabels++;
			} // Of if
		} // Of for i

		// Step 2. How many labels to buy?
		int[] tempNumBuys = lookup(paraBlock.length);
		// ���ص������ǵڣ��±꣩����ʱ�飬���ٸ��ܴ�����С��
		// tempNumBuys[0]��tempNumBuys[1]�����һ����ߵڶ���� i�� ����������������ٸ���Ϊ�Ǵ��Ĳ��Ҵ�����С)
		int tempBuyLabels = 0;
		if (tempFirstLabel == 0 || tempFirstLabel == 1) {// ��tempFirstLabel�ǵ�һ����ߵڶ���
			tempBuyLabels = tempNumBuys[tempFirstLabel] - tempLabels;
			// ��������������ĸ�����ȥ�Ѿ���ĸ���
		} else {
			tempBuyLabels = Math.max(tempNumBuys[0], tempNumBuys[1])
					- tempLabels;
			// �������������lookup���ص����ֵ��ȥ�Ѿ���ĸ�����
		} // Of if

		// int tempBuyLabels = pureThreshold(paraBlock.length) - tempLabels;
		if (pureThreshold(paraBlock.length) - tempBuyLabels > 0) {// ���㴿���ż�-Ҫ��ı�ǩ��Ŀ
			System.out.println("+");
		} else if (pureThreshold(paraBlock.length) - tempBuyLabels < 0) {
			System.out.println("-");
		}

		return tempBuyLabels;
	}

	private int[] lookup(int pSize) {
		// Linear Search�������ݿ��С
		double[] tmpMinCost = new double[] { 0.5 * pSize * mCost[0],
				0.5 * pSize * mCost[1] };
		int[] star = new int[2];
		int[] starLoose = new int[2];
		boolean[] isFind = new boolean[2];
		double[] ra = new double[pSize + 1];
		// ra[]������ǳ鵽��i��ʱ���ж�������Ϊ�������Ǵ���
		ra[0] = 0.5;// ��һ������Ǻ���������ĸ��ʶ���0.5�����Ұ��������е��뷨��������Ķ���һ�µ���ɫ
		double[] tmpCost = new double[2];// 1���Ϊ2��Ĵ����2���Ϊ1��Ĵ�����󶼼���tcost��
		for (int i = 1; i <= pSize; i++) {
			if (pSize >= 1000) {
				ra[i] = (i + 1.0) / (i + 2.0);// ����
												// �����ȴ���1000�������鵽��i����ͬ��ɫʱ�򴿵ĸ�����i+1/i+2
			} else {
				ra[i] = expectPosNum(i, 0, pSize) / pSize;// ��С��1000�����ֱ�Ӽ��������ı���
															// ����101������
															// ����0-100����һ�����
															// ��ÿ������������˸��ʣ���ͣ�����80����80-100�ĺͣ����������101
															// ���������ĸ���
			} // ����ÿ�������ĸ����¿��ܵĴ��ۡ�
			for (int j = 0; j < 2; j++) {// 1���Ϊ2��Ĵ����2���Ϊ1��Ĵ�����󶼼���tcost��
				tmpCost[j] = (1 - ra[i]) * mCost[j] * pSize + tCost * i;// 0����1�������Ĵ���
				if (tmpCost[j] < tmpMinCost[j]) {// ����0����1��ǰ�����µ���С���ۣ����Ҽ�¼i�ĸ���
					tmpMinCost[j] = tmpCost[j];
					star[j] = i;// ���һ�����С���������i����������star[1]=i
					if (i == pSize) {// ��i==���ݿ�Ĵ�С
						Arrays.fill(isFind, true);// isFind��Ϊȫ���ҵ���,!!����ȡ����������һ��ȫȡ��Ϊ������С
					} // Of if
				} else {
					isFind[j] = true;// ֻ��Ϊ��j���ҵ������Ž�
				} // Of if
			} // Of for j
				// System.out.println("QueriedInstance: " + i + ", cost: " +
				// tmpCost[0] + "\t");
			if (isFind[0] && isFind[1]) {// ����0��͵�1�඼�ҵ������Ÿ���
				Arrays.fill(isFind, false);// ȫ��д��false
				for (int j = 0; j <= i; j++) {// ������������֮ǰ��
												// ����и�С�ĸ���ȥ�ó����Ž⣬�Ǿͼ����������ٵĸ���
					for (int k = 0; k < 2; k++) {
						tmpCost[k] = (1 - ra[j]) * mCost[k] * pSize + tCost * j;
						if (tmpCost[k] <= tmpMinCost[k] && !isFind[k]) {
							isFind[k] = true;
							starLoose[k] = j;
						} // Of if
						if (isFind[0] && isFind[1]) {
							return starLoose;// {1,2}���ص�һ��͵ڶ���Ӧ�ó�ȡ�ĸ���������˵Ԥ��ĸ�������i���Ƚ����Ǿͳ�i��
												// ������С
						} // Of if
					} // Of for k
				} // Of for j
			} // Of if
		} // Of for i
		return new int[] { 0, 0 };// ��û���ҵ��ͷ���0.0��
		// throw new RuntimeException("Error occured in lookup("+pSize+")");
	}// Of lookup
}//Of class KMeansDistance

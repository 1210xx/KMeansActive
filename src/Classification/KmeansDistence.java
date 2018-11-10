package Classification;
import java.io.FileReader;
import java.math.BigDecimal;
import java.util.*;
import weka.*;
import weka.core.Instances;
public class KmeansDistence
{

	static Random random = new Random();

	Instances data;

	int[] instanceStates; // 1 represents bought and 2 represents predicted.

	int[] labels;

	double[][] currentCenters;

	double tCost;

	double[] mCost;

	public static void main(String args[])
	{

		double[] mCost = { 2, 4 };

		double avgCost = 0;
	//for (int i = 0; i < 20; i++)
		//{
			KmeansDistence tempLeaner = new KmeansDistence("Data/flame.arff");
			tempLeaner.mCost = mCost;
			tempLeaner.tCost = 1;
			// int[] tempBlock = {1, 4, 5, 6, 59, 121};
			// //System.out.println("The clustering result is: " +
			// Arrays.toString(tempLeaner.cluster(3, tempBlock)));
			tempLeaner.learningTest();
			avgCost += tempLeaner.totalCost();
			System.out.println("OK");
			System.out.println("onceOfCost"+avgCost);
	//	}

		//System.out.println("20avgCost" + avgCost / 20);
	}// Of main

	public double totalCost()
	{
		double cost = 0;
		for (int i = 0; i < data.numInstances(); i++)
		{
			if(instanceStates[i] == 1)
			{
				cost += tCost;
			} else
			{
				if(labels[i] == 0 && (int) data.instance(i).classValue() == 1)
				{
					cost += mCost[0];// 错误分类 labels数组表示的是每个Instance被分成哪个类
				} else if(labels[i] == 1 && (int) data.instance(i).classValue() == 0)
				{
					cost += mCost[1];
				} // Of if
			}
		}
		return cost;
	}

	
	
	//Constructor
	public KmeansDistence(String paraFilename)
	{
		data = null;
		try
		{
			FileReader fileReader = new FileReader(paraFilename);
			data = new Instances(fileReader);
			fileReader.close();
			data.setClassIndex(data.numAttributes() - 1);
			// System.out.println(data);
		} catch (Exception ee)
		{
			// System.out.println("Cannot read the file: " + paraFilename + "\r\n" + ee);
			System.exit(0);
		} // Of try

		// Initialize
		instanceStates = new int[data.numInstances()];
		labels = new int[data.numInstances()];
		Arrays.fill(labels, -1);
	}// Of the first constructor
	
	void learningTest()
	{
		int[] originalBlock = new int[data.numInstances()];// 开始
		for (int i = 0; i < originalBlock.length; i++)
		{
			originalBlock[i] = i;
		} // Of for i

		learning(originalBlock);// 传入的只是原数据的下标

		System.out.println("instanceStates: " + Arrays.toString(instanceStates));
		System.out.println("labels: " + Arrays.toString(labels));


	}// Of learningTest
	
	void learning(int[] paraBlock) {
		int blockSize=paraBlock.length;
		int firstLabel=-1;
		int IndexOfFirstLabel=-1;
		int secondLabel=-1;
		int IndexOfSecondLabel=-1;
		//长度小于5直接购买
		if(blockSize<=5) {
			for(int i=0;i<blockSize;i++) {
				instanceStates[paraBlock[i]]=1;
				labels[paraBlock[i]]=(int)data.instance(paraBlock[i]).value(data.numAttributes()-1);
				
			}
			return;
		}//of if
		
		//找到第一个已经购买的点
		for(int i=0;i<blockSize;i++) {
			if(instanceStates[paraBlock[i]]==1) {
				IndexOfFirstLabel=i;
				firstLabel=(int)data.instance(paraBlock[i]).value(data.numAttributes()-1);
				break;
			}
		}
		
		
		if(IndexOfFirstLabel!=-1) {
		for(int i=IndexOfFirstLabel+1;i<blockSize;i++) {
			if(instanceStates[paraBlock[i]]==1) {
				IndexOfSecondLabel=i;
				secondLabel=(int)data.instance(paraBlock[i]).value(data.numAttributes()-1);
				if(secondLabel!=firstLabel) {
					splitAndLearn(paraBlock,IndexOfSecondLabel,IndexOfFirstLabel);
					return;
					}
				}//of for  找到两个标签不同的就分开
			}	
		}
		
	/**
	 * 现在开始找需要购买的代表点数
	 */
		
		int needToBuyNum=findRepresentatives(paraBlock,firstLabel);
		List<Integer> pointIndex=new ArrayList<Integer>();
		
		if(IndexOfFirstLabel!=-1)
		pointIndex.add(IndexOfFirstLabel);
		else {
		pointIndex.add(0);
		//现在开始顺序买下第一个代表点
		instanceStates[paraBlock[pointIndex.get(0)]]=1;
		labels[paraBlock[pointIndex.get(0)]]=(int)data.instance(paraBlock[pointIndex.get(0)]).value(data.numAttributes()-1);
		}
		while(pointIndex.size()<=(needToBuyNum-1)) {
		//将数据集的第一个点加入中心点的第一个
		
		//现在找第二个		
		int zuiyuandian=computeFathestPoint(paraBlock, pointIndex);//返回的是paraBlock内部下标，而不是最原始的编号
		//买下最远点的标签
		instanceStates[paraBlock[zuiyuandian]]=1;
		labels[paraBlock[zuiyuandian]]=(int)data.instance(paraBlock[zuiyuandian]).value(data.numAttributes()-1);
		
		if(labels[paraBlock[zuiyuandian]]!=labels[paraBlock[0]]) {
			splitAndLearn(paraBlock,pointIndex.get(0),zuiyuandian);
			return;
		}
		else
			pointIndex.add(zuiyuandian);
		}//of while while循环完毕证明所有点全部都已经和第一个点一样 直接进行预测
		
		
		
		// Step 3. Predict others in this block. 这就是根据代表点直接搞了
				for (int i = 0; i < paraBlock.length; i++)
				{
					if(instanceStates[paraBlock[i]] != 1)
					{
						instanceStates[paraBlock[i]] = 2;
						labels[paraBlock[i]] = labels[paraBlock[pointIndex.get(0)]];
					} // Of if
				} // Of for i
				return;
		
		
	}//of learning
	
	public void splitAndLearn(int[] paraBlock,int a,int b)
	{
		// Step 1. Split
		int[] tempClutering = cluster(a,b, paraBlock);
		int tempFirstBlockSize = 0;
		for (int i = 0; i < tempClutering.length; i++)
		{
			if(tempClutering[i] == 0)
			{
				tempFirstBlockSize++;
			} // Of if
		} // Of for i

		int[][] tempBlocks = new int[2][];
		tempBlocks[0] = new int[tempFirstBlockSize];
		tempBlocks[1] = new int[paraBlock.length - tempFirstBlockSize];

		int[] tempCounters = {0,0};
		

		for (int i = 0; i < tempClutering.length; i++)
		{
			tempBlocks[tempClutering[i]][tempCounters[tempClutering[i]]++] = paraBlock[i];
		} // Of for i

		System.out.println("SplittedAndLearn into two blocks: " + Arrays.toString(tempBlocks[0]) + "\r\n"
				+ Arrays.toString(tempBlocks[1]));

		// Step 2. Learn
		learning(tempBlocks[0]);
		learning(tempBlocks[1]);
	}
	
	public int[] cluster(int a,int b,int[] paraBlock)
	{
		// Step 1. Initialize
		int tempBlockSize = paraBlock.length;
		int[] tempCluster = new int[tempBlockSize];
		double[][] tempCenters = new double[2][data.numAttributes() - 1];
		double[][] tempNewCenters = new double[2][data.numAttributes() - 1];

			for(int j=0;j<data.numAttributes()-1;j++)
			{	tempNewCenters[0][j]=data.instance(paraBlock[a]).value(j);
			tempNewCenters[1][j]=data.instance(paraBlock[b]).value(j);
			}
		// Step 3. Cluster and compute new centers.
		while (!doubleMatricesEqual(tempCenters, tempNewCenters))
		{
			// while (!Arrays.deepEquals(tempCenters, tempNewCenters)) {
			tempCenters = tempNewCenters;
			// Cluster
			for (int i = 0; i < tempBlockSize; i++)
			{
				double tempDistance = Double.MAX_VALUE;
				for (int j = 0; j < 2; j++)
				{
					double tempCurrentDistance = distance(paraBlock[i], tempCenters[j]);
					if(tempCurrentDistance < tempDistance)
					{
						tempCluster[i] = j;// 计算每个点属于哪个中心
						tempDistance = tempCurrentDistance;
					} // Of cluster
				} // Of for j
			} // Of for i

			// System.out.println("Current cluster: " + Arrays.toString(tempCluster));

			// Compute new centers
			int[] tempCounters = new int[2];
			for (int i = 0; i < tempCounters.length; i++)
			{
				tempCounters[i] = 0;
			} // Of for i

			tempNewCenters = new double[2][data.numAttributes() - 1];
			for (int i = 0; i < tempBlockSize; i++)
			{
				tempCounters[tempCluster[i]]++;// 计算每个中心有几个点
				for (int j = 0; j < data.numAttributes() - 1; j++)
				{
					tempNewCenters[tempCluster[i]][j] += data.instance(paraBlock[i]).value(j);
				} // Of for j
			} // Of for i

			// Average
			for (int i = 0; i < 2; i++)
			{
				for (int j = 0; j < data.numAttributes() - 1; j++)
				{
					tempNewCenters[i][j] /= tempCounters[i];
				} // Of for j
			} // Of for i
			currentCenters = tempNewCenters;// 这个数组是其他点到中心点的平均距离

			//// System.out.println("The centers are: " + Arrays.deepToString(tempCenters));
			//// System.out.println("The new centers are: " +
			//// Arrays.deepToString(tempNewCenters));
		} // Of while
		System.out.println("原本的cluster属于哪个中心" + Arrays.toString(tempCluster));
		return tempCluster;
	}// Of cluster

	
	
	double computeDistence(int point,int paraBlockPoint) {
		double tempDistence=0;
		for(int i=0;i<data.numAttributes()-1;i++) {
			tempDistence+= Math.abs(data.instance(point).value(i)-data.instance(paraBlockPoint).value(i)); 
		}
		return tempDistence;
	}
	/**
	 * 计算当前数据块最远的数据点（距离所有中心点最远的点）
	 */
	public int computeFathestPoint(int[]paraBlock,List<Integer> centerPoints) {
		int blockSize=paraBlock.length;
		int listSize=centerPoints.size();
		int tempIndexOfMaxPoint = 0;
		int[] isFinded=new int[blockSize];
		for(int i=0;i<listSize;i++) {
			isFinded[centerPoints.get(i)]=1;
		}
		double tempMaxDistence=0;
		for(int i=0;i<blockSize;i++) {
		
			double tempDistence=0;
			for(int j=0;j<listSize;j++) {
				tempDistence+=computeDistence(paraBlock[centerPoints.get(j)], paraBlock[i]);
				
			}
			
			if((tempDistence>tempMaxDistence)&&(isFinded[i]!=1)) {
				tempIndexOfMaxPoint=i;
				tempMaxDistence=tempDistence;
}
		}
		
		System.out.println("最远点距离所有已知点最远在所有数据中的总索引"+paraBlock[tempIndexOfMaxPoint]);
		return tempIndexOfMaxPoint;
	} 
	/**
	 ***************
	 * With how many instances with the same label can we say it is pure?
	 ***************
	 */
	public int pureThreshold(int paraSize)
	{
		return (int) Math.sqrt(paraSize);
	}// Of pureThreshold
	
	public static boolean doubleMatricesEqual(double[][] paraMatrix1, double[][] paraMatrix2)
	{
		for (int i = 0; i < paraMatrix1.length; i++)
		{
			for (int j = 0; j < paraMatrix1[0].length; j++)
			{
				if(Math.abs(paraMatrix1[i][j] - paraMatrix2[i][j]) > 1e-6)
				{
					return false;
				} // Of if
			} // Of for j
		} // Of for i
		return true;
	}// Of doubleMatricesEqual
	
	
	public static double expectPosNum(int R, int B, int N)
	{
		BigDecimal fenzi = new BigDecimal("0");
		BigDecimal fenmu = new BigDecimal("0");
		for (int i = R; i <= N - B; i++)
		{
			BigDecimal a = A(R, i).multiply(A(B, N - i));
			fenzi = fenzi.add(a.multiply(new BigDecimal("" + i)));
			fenmu = fenmu.add(a);
			// System.out.println("fenzi:" + fenzi + ", fenmu: " + fenmu);
		} // Of for i
		return fenzi.divide(fenmu, 4, BigDecimal.ROUND_HALF_EVEN).doubleValue();
	}// Of expectPosNum
	
	public static BigDecimal A(int m, int n)
	{
		if(m > n)
		{
			return new BigDecimal("0");
		} // Of if
		BigDecimal re = new BigDecimal("1");
		for (int i = n - m + 1; i <= n; i++)
		{
			re = re.multiply(new BigDecimal(i));
		} // Of if
		return re;
	}// Of A
	
	
	
	public double distance(int paraIndex, double[] paraArray)
	{
		double resultDistance = 0;
		for (int i = 0; i < paraArray.length; i++)
		{
			resultDistance += Math.abs(data.instance(paraIndex).value(i) - paraArray[i]);
		} // Of for i
			// System.out.println(resultDistance);
		return resultDistance;
	}// Of distance

	/**
	 ***************
	 * Find some representatives given the instances.
	 ***************
	 */
	public int findRepresentatives(int[] paraBlock, int tempFirstLabel)
	{
		// Step 1. How many labels do we already bought?
		int tempLabels = 0;
		for (int i = 0; i < paraBlock.length; i++)
		{
			if(instanceStates[paraBlock[i]] == 1)
			{
				tempLabels++;
			} // Of if
		} // Of for i

		// Step 2. How many labels to buy?
		int[] tempNumBuys = lookup(paraBlock.length);
		// 返回的数组是第（下标）个类时抽，多少个总代价最小。
		// tempNumBuys[0]和tempNumBuys[1]代表第一类或者第二类抽 i个 （就是理论上买多少个认为是纯的并且代价最小)
		int tempBuyLabels = 0;
		if(tempFirstLabel == 0 || tempFirstLabel == 1)
		{// 若tempFirstLabel是第一类或者第二类
			tempBuyLabels = tempNumBuys[tempFirstLabel] - tempLabels;
			// 个数就是期望抽的个数减去已经买的个数
		} else
		{
			tempBuyLabels = Math.max(tempNumBuys[0], tempNumBuys[1]) - tempLabels;
			// 否则就你好你好是lookup返回的最大值减去已经买的个数输
		} // Of if

		// int tempBuyLabels = pureThreshold(paraBlock.length) - tempLabels;
		if(pureThreshold(paraBlock.length) - tempBuyLabels > 0)
		{// 计算纯的门槛-要买的标签数目
			System.out.println("+");
		} else if(pureThreshold(paraBlock.length) - tempBuyLabels < 0)
		{
			System.out.println("-");
		}
		
		return tempBuyLabels;
		}
	
	
	
	private int[] lookup(int pSize)
	{
		// Linear Search传入数据快大小
		double[] tmpMinCost = new double[] { 0.5 * pSize * mCost[0], 0.5 * pSize * mCost[1]};
		int[] star = new int[2];
		int[] starLoose = new int[2];
		boolean[] isFind = new boolean[2];
		double[] ra = new double[pSize + 1];
		// ra[]数组就是抽到第i个时候有多大概率认为整个块是纯的
		ra[0] = 0.5;// 第一个抽出是红球还是蓝球的概率都是0.5，并且按照论文中的想法接下来抽的都是一致的颜色
		double[] tmpCost = new double[2];// 1类分为2类的错误和2类分为1类的错误（最后都加上tcost）
		for (int i = 1; i <= pSize; i++)
		{
			if(pSize >= 1000)
			{
				ra[i] = (i + 1.0) / (i + 2.0);// 概率 若长度大于1000是无穷，则抽到第i个相同颜色时候纯的概率是i+1/i+2
			} else
			{
				ra[i] = expectPosNum(i, 0, pSize) / pSize;// 若小于1000个则就直接计算期望的比例 就是101个袋子 含有0-100个各一种情况
															// 求每个情况（个数乘概率）求和（大于80就求80-100的和）处以总情况101 就是期望的概率
			} // 计算每个期望的概率下可能的代价。
			for (int j = 0; j < 2; j++)
			{// 1类分为2类的错误和2类分为1类的错误（最后都加上tcost）
				tmpCost[j] = (1 - ra[i]) * mCost[j] * pSize + tCost * i;// 0或者1分类错误的代价
				if(tmpCost[j] < tmpMinCost[j])
				{// 计算0或者1当前期望下的最小代价，并且记录i的个数
					tmpMinCost[j] = tmpCost[j];
					star[j] = i;// 抽第一类的最小错误代价是i个，二类是star[1]=i
					if(i == pSize)
					{// 若i==数据块的大小
						Arrays.fill(isFind, true);// isFind认为全都找到了,!!就是取两类中其中一个全取认为代价最小
					} // Of if
				} else
				{
					isFind[j] = true;// 只认为第j类找到了最优解
				} // Of if
			} // Of for j
				// System.out.println("QueriedInstance: " + i + ", cost: " +
				// tmpCost[0] + "\t");
			if(isFind[0] && isFind[1])
			{// 若第0类和第1类都找到了最优个数
				Arrays.fill(isFind, false);// 全部写成false
				for (int j = 0; j <= i; j++)
				{// 从期望的数量之前找 如果有更小的个数去得出最优解，那就计算出这个更少的个数
					for (int k = 0; k < 2; k++)
					{
						tmpCost[k] = (1 - ra[j]) * mCost[k] * pSize + tCost * j;
						if(tmpCost[k] <= tmpMinCost[k] && !isFind[k])
						{
							isFind[k] = true;
							starLoose[k] = j;
						} // Of if
						if(isFind[0] && isFind[1])
						{
							return starLoose;// {1,2}返回第一类和第二类应该抽取的个数？或者说预测的个数？抽i个比较稳那就抽i个 代价最小
						} // Of if
					} // Of for k
				} // Of for j
			} // Of if
		} // Of for i
		return new int[] { 0, 0 };// 若没有找到就返回0.0了
		// throw new RuntimeException("Error occured in lookup("+pSize+")");
	}// Of lookup
	}

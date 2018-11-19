package KMeansActiceTest;

import static org.junit.Assert.*;

import java.math.BigDecimal;

import org.junit.Test;

import classification.KMeans;
import classification.KMeansActive;
import weka.gui.beans.ClassAssigner;

public class KMeansActiceUnittest {

	@Test
	public void testKMeansActive() {
		fail("Not yet implemented");
	}

	@Test
	public void testLearningTest() {
		fail("Not yet implemented");
	}

	@Test
	public void testLearning() {
		fail("Not yet implemented");
	}

	@Test
	public void testCluster() {
		fail("Not yet implemented");
	}

	@Test
	public void testSplitAndLearn() {
		fail("Not yet implemented");
	}

	@Test
	public void testTotalCost() {
		fail("Not yet implemented");
	}

	@Test
	public void testFindRepresentatives() {
		fail("Not yet implemented");
	}

	@Test
	public void testExpectPosNum() {
		fail("Not yet implemented");
	}

	@Test
	public void testA() {
		new KMeansActive("/Users/Rjv587/Downloads/Papers/Data/manmade/thyroid_train_re_last_test.arff");
		BigDecimal z = KMeansActive.A(3, 3);
		System.out.println(z);
	}

	@Test
	public void testDoubleMatricesEqual() {
		fail("Not yet implemented");
	}

	@Test
	public void testDistance() {
		fail("Not yet implemented");
	}

	@Test
	public void testPureThreshold() {
		int n = new KMeansActive("/Users/Rjv587/Downloads/Papers/Data/manmade/thyroid_train_re_last_test.arff").pureThreshold(-3);
		assertEquals(2, n);
	}

	@Test
	public void testDataTest() {
		fail("Not yet implemented");
	}

	@Test
	public void testIntArrayToString() {
		fail("Not yet implemented");
	}

}

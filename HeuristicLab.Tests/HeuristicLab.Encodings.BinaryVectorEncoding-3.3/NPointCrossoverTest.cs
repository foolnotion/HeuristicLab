﻿#region License Information
/* HeuristicLab
 * Copyright (C) 2002-2012 Heuristic and Evolutionary Algorithms Laboratory (HEAL)
 *
 * This file is part of HeuristicLab.
 *
 * HeuristicLab is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * HeuristicLab is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with HeuristicLab. If not, see <http://www.gnu.org/licenses/>.
 */
#endregion

using HeuristicLab.Core;
using HeuristicLab.Data;
using HeuristicLab.Encodings.BinaryVectorEncoding;
using HeuristicLab.Tests;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HeuristicLab.Encodings.BinaryVectorEncoding_33.Tests {


  /// <summary>
  ///This is a test class for SinglePointCrossoverTest and is intended
  ///to contain all SinglePointCrossoverTest Unit Tests
  ///</summary>
  [TestClass()]
  public class NPointCrossoverTest {


    private TestContext testContextInstance;

    /// <summary>
    ///Gets or sets the test context which provides
    ///information about and functionality for the current test run.
    ///</summary>
    public TestContext TestContext {
      get {
        return testContextInstance;
      }
      set {
        testContextInstance = value;
      }
    }

    #region Additional test attributes
    // 
    //You can use the following additional attributes as you write your tests:
    //
    //Use ClassInitialize to run code before running the first test in the class
    //[ClassInitialize()]
    //public static void MyClassInitialize(TestContext testContext)
    //{
    //}
    //
    //Use ClassCleanup to run code after all tests in a class have run
    //[ClassCleanup()]
    //public static void MyClassCleanup()
    //{
    //}
    //
    //Use TestInitialize to run code before running each test
    //[TestInitialize()]
    //public void MyTestInitialize()
    //{
    //}
    //
    //Use TestCleanup to run code after each test has run
    //[TestCleanup()]
    //public void MyTestCleanup()
    //{
    //}
    //
    #endregion

    /// <summary>
    ///A test for Cross
    ///</summary>
    [TestMethod()]
    [DeploymentItem("HeuristicLab.Encodings.BinaryVectorEncoding-3.3.dll")]
    public void NPointCrossoverCrossTest() {
      NPointCrossover_Accessor target = new NPointCrossover_Accessor(new PrivateObject(typeof(NPointCrossover)));
      ItemArray<BinaryVector> parents;
      TestRandom random = new TestRandom();
      bool exceptionFired;
      // The following test checks if there is an exception when there are more than 2 parents
      random.Reset();
      parents = new ItemArray<BinaryVector>(new BinaryVector[] { new BinaryVector(5), new BinaryVector(6), new BinaryVector(4) });
      exceptionFired = false;
      try {
        BinaryVector actual;
        actual = target.Cross(random, parents);
      }
      catch (System.ArgumentException) {
        exceptionFired = true;
      }
      Assert.IsTrue(exceptionFired);
      // The following test checks if there is an exception when there are less than 2 parents
      random.Reset();
      parents = new ItemArray<BinaryVector>(new BinaryVector[] { new BinaryVector(4) });
      exceptionFired = false;
      try {
        BinaryVector actual;
        actual = target.Cross(random, parents);
      }
      catch (System.ArgumentException) {
        exceptionFired = true;
      }
      Assert.IsTrue(exceptionFired);
    }

    /// <summary>
    ///A test for Apply
    ///</summary>
    [TestMethod()]
    public void NPointCrossoverApplyTest() {
      TestRandom random = new TestRandom();
      BinaryVector parent1, parent2, expected, actual;
      IntValue n;
      bool exceptionFired;
      // The following test is based on Eiben, A.E. and Smith, J.E. 2003. Introduction to Evolutionary Computation. Natural Computing Series, Springer-Verlag Berlin Heidelberg, p. 48
      random.Reset();
      n = new IntValue(1);
      random.IntNumbers = new int[] { 4 };
      parent1 = new BinaryVector(new bool[] { false, false, false, false, true, false, false, false, false });
      parent2 = new BinaryVector(new bool[] { true, true, false, true, false, false, false, false, true });
      expected = new BinaryVector(new bool[] { false, false, false, false, false, false, false, false, true });
      actual = NPointCrossover.Apply(random, parent1, parent2, n);
      Assert.IsTrue(Auxiliary.BinaryVectorIsEqualByPosition(actual, expected));

      // The following test is based on Eiben, A.E. and Smith, J.E. 2003. Introduction to Evolutionary Computation. Natural Computing Series, Springer-Verlag Berlin Heidelberg, p. 48
      random.Reset();
      n = new IntValue(2);
      random.IntNumbers = new int[] { 4, 5 };
      parent1 = new BinaryVector(new bool[] { false, false, false, false, true, false, false, false, false });
      parent2 = new BinaryVector(new bool[] { true, true, false, true, false, false, false, false, true });
      expected = new BinaryVector(new bool[] { false, false, false, false, false, false, false, false, false });
      actual = NPointCrossover.Apply(random, parent1, parent2, n);
      Assert.IsTrue(Auxiliary.BinaryVectorIsEqualByPosition(actual, expected));

      // The following test is based on Eiben, A.E. and Smith, J.E. 2003. Introduction to Evolutionary Computation. Natural Computing Series, Springer-Verlag Berlin Heidelberg, p. 48
      random.Reset();
      n = new IntValue(2);
      random.IntNumbers = new int[] { 4, 5 };
      parent2 = new BinaryVector(new bool[] { false, false, false, false, true, false, false, false, false });
      parent1 = new BinaryVector(new bool[] { true, true, false, true, false, false, false, false, true });
      expected = new BinaryVector(new bool[] { true, true, false, true, true, false, false, false, true });
      actual = NPointCrossover.Apply(random, parent1, parent2, n);
      Assert.IsTrue(Auxiliary.BinaryVectorIsEqualByPosition(actual, expected));

      // The following test is not based on any published examples
      random.Reset();
      random.IntNumbers = new int[] { 2 };
      parent1 = new BinaryVector(new bool[] { false, true, true, false, false }); // this parent is longer
      parent2 = new BinaryVector(new bool[] { false, true, true, false });
      exceptionFired = false;
      try {
        actual = NPointCrossover.Apply(random, parent1, parent2, n);
      }
      catch (System.ArgumentException) {
        exceptionFired = true;
      }
      Assert.IsTrue(exceptionFired);
    }

    /// <summary>
    ///A test for SinglePointCrossover Constructor
    ///</summary>
    [TestMethod()]
    public void NPointCrossoverConstructorTest() {
      NPointCrossover target = new NPointCrossover();
    }
  }
}

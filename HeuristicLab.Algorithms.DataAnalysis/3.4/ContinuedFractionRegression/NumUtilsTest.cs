﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace HeuristicLab.Algorithms.DataAnalysis.ContinuedFractionRegression {
  public class NumUtilsTests {
    /// <summary>
    /// Test to see if we can fit a parabola
    /// </summary>
    public static void SimplexTest1() {
      Console.WriteLine("Starting SimplexTest1");
      SimplexConstant[] constants = new SimplexConstant[] { new SimplexConstant(3, 1), new SimplexConstant(5, 1) };
      double tolerance = 1e-6;
      int maxEvals = 1000;
      ObjectiveFunctionDelegate objFunction = new ObjectiveFunctionDelegate(_objFunction1);
      RegressionResult result = NelderMeadSimplex.Regress(constants, tolerance, maxEvals, objFunction);
      _printResult(result);
    }

    private static double _objFunction1(double[] constants) {
      double a = 5;
      double b = 10;

      Console.Write("Called with a={0} b={1}", constants[0], constants[1]);

      // evaluate it for some selected points, with a bit of noise
      double ssq = 0;
      var r = new System.Random();
      for (double x = -10; x < 10; x += .1) {
        double yTrue = a * x * x + b * x + r.NextDouble();
        double yRegress = constants[0] * x * x + constants[1] * x;
        ssq += Math.Pow((yTrue - yRegress), 2);
      }
      Console.WriteLine("  SSQ={0}", ssq);
      return ssq;
    }

    /// <summary>
    /// Test on the Rosenbrock function
    /// </summary>
    public static void SimplexTest2() {
      Console.WriteLine("\n\nStarting SimplexTest2");

      // we are regressing for frequency, amplitude, and phase offset
      SimplexConstant[] constants = new SimplexConstant[] { new SimplexConstant(-1.2, .1), new SimplexConstant(1, .1) };
      double tolerance = 1e-10;
      int maxEvals = 1000;
      ObjectiveFunctionDelegate objFunction = new ObjectiveFunctionDelegate(_objFunction2);
      RegressionResult result = NelderMeadSimplex.Regress(constants, tolerance, maxEvals, objFunction);
      _printResult(result);
    }

    private static double _objFunction2(double[] constants) {
      Console.Write("Called with x1={0} x2={1} ", constants[0], constants[1]);
      double err = 100 * Math.Pow((constants[1] - constants[0] * constants[0]), 2) + Math.Pow((1 - constants[0]), 2);
      Console.WriteLine("  err={0}", err);

      return err;
    }

    private static void _printResult(RegressionResult result) {
      // a bit of reflection fun, why not...
      PropertyInfo[] properties = result.GetType().GetProperties();
      foreach (PropertyInfo p in properties) {
        Console.WriteLine(p.Name + ": " + p.GetValue(result, null).ToString());
      }
    }
  }
}

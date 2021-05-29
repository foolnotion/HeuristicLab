using System;
using HeuristicLab.Data;

namespace HeuristicLab.Algorithms.DataAnalysis.ContinuedFractionRegression {
  public class Transformation {

    DoubleMatrix dataMatrix;

    private double[] tempMinMax;
    private double[] phiMinMax;
    private double[] phipMinMax;

    public Transformation(DoubleMatrix dataMatrix) {
      this.dataMatrix = dataMatrix;
      transform0();
    }

    private void transform0() { // care! change the other two transformations accordingly!
      tempMinMax = findMinMax(0);
      minMaxTransformation(this.dataMatrix, tempMinMax, 0);
      log10Transformation(this.dataMatrix, 2);
    }

    public double[] transform0(double[] orig) {
      double[] trans = new double[3];
      trans[0] = minMaxTransformation(tempMinMax, orig[0]);
      trans[1] = orig[1];
      trans[2] = log10Transformation(orig[2]);

      return trans;
    }

    public void useSameTransformation(DoubleMatrix dataMatrix) {
      minMaxTransformation(dataMatrix, tempMinMax, 0);
      log10Transformation(dataMatrix, 2);
    }

    //another transformation - not as good as i thougt
    private void transform1() { // care! change retransformation accordingly!
      tempMinMax = findMinMax(0);
      minMaxTransformation(this.dataMatrix, this.tempMinMax, 0);
      phiMinMax = findMinMax(1);
      minMaxTransformation(this.dataMatrix, this.phiMinMax, 1);
      log10Transformation(2); // no negativ values!
      phipMinMax = findMinMax(2);
      minMaxTransformation(this.dataMatrix, this.phipMinMax, 2); // Min and Max have to be different!
    }

    public double[] transform1(double[] orig) {
      double[] trans = new double[3];
      trans[0] = minMaxTransformation(tempMinMax, orig[0]);
      trans[1] = minMaxTransformation(phiMinMax, orig[1]);
      trans[2] = log10Transformation(orig[2]);
      trans[2] = minMaxTransformation(phipMinMax, trans[2]);

      return trans;
    }
    //

    private double[] findMinMax(int column) {
      // the first row has index 0
      double[] MinMax = new double[2] { double.MaxValue, double.MinValue };
      // find min and max in column
      for (int i = 0; i < dataMatrix.Rows; i++) {
        if (dataMatrix[i, column] < MinMax[0])
          MinMax[0] = dataMatrix[i, column];
        if (dataMatrix[i, column] > MinMax[1])
          MinMax[1] = dataMatrix[i, column];
      }
      return MinMax;
    }

    private void minMaxTransformation(DoubleMatrix matrix, double[] minMax, int column) {
      // transform all values in column
      for (int i = 0; i < matrix.Rows; i++) {
        matrix[i, column] = (matrix[i, column] - minMax[0]) / (minMax[1] - minMax[0]);
      }
    }

    private double minMaxTransformation(double[] minMax, double x) {
      return (x - minMax[0]) / (minMax[1] - minMax[0]);
    }

    private void log10Transformation(DoubleMatrix matrix, int column) {
      // the first row has index 0
      for (int i = 0; i < matrix.Rows; i++) {
        matrix[i, column] = Math.Log10(matrix[i, column]);
      }
    }

    private double log10Transformation(double x) {
      return Math.Log10(x);
    }
  }

}

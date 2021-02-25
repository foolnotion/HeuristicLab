using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using HeuristicLab.Data;

namespace HeuristicLab.Algorithms.DataAnalysis.ContinuedFractionRegression {
  public class ObjFunctionNelderMead {
    Problem parent; // needed to have acces on calculate members

    DoubleMatrix sampleData;
    double[] coeff;
    bool[] binar;

    public ObjFunctionNelderMead(Problem parent, DoubleMatrix data, double[] coeff, bool[] binar) {
      this.parent = parent;
      this.sampleData = data;
      this.coeff = coeff;
      this.binar = binar;
    }

    public ObjFunctionNelderMead(Problem parent, DoubleMatrix data, int num, double[] coeff, bool[] binar) {
      this.parent = parent;
      this.sampleData = new DoubleMatrix(generateRandomPartialMatrix(data, num));
      this.coeff = coeff;
      this.binar = binar;
    }

    public DoubleMatrix getData() {
      return sampleData;
    }

    private double[,] generateRandomPartialMatrix(DoubleMatrix data, int num) {
      double[,] partialMatrix = new double[num, data.Columns];
      bool[] alreadyUsed = new bool[data.Rows]; //initialize all booleans by false
      int rand = 0;
      int count = 0;
      System.Random rnd = new System.Random();
      while (count < num) {
        rand = rnd.Next(1, data.Rows);
        if (alreadyUsed[rand - 1] == false) {
          for (int i = 0; i < data.Columns; i++) {
            partialMatrix[count, i] = data[rand - 1, i];
          }
          count++;
          alreadyUsed[rand - 1] = true;
        }
      }
      return partialMatrix;
    }

    public double objFunctionNelderMead(double[] constants) {
      double[] optimizedCoeff = new double[coeff.Length];
      int shortIndex = 0;
      for (int i = 0; i < coeff.Length; i++) {
        if (binar[i] == true) {
          optimizedCoeff[i] = constants[shortIndex];
          shortIndex++;
        } else {
          optimizedCoeff[i] = coeff[i];
        }
      }
      double LS = parent.calculateLS(optimizedCoeff, binar, sampleData);
      return parent.calculateQuality(optimizedCoeff, binar, LS);
    }
  }
}

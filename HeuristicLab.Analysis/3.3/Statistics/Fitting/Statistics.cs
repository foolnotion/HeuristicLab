using System;
using System.Collections.Generic;
using System.Linq;
using HeuristicLab.Common;
using HeuristicLab.Data;
using HeuristicLab.Optimization;

namespace HeuristicLab.Analysis.Statistics {
  // C# code is derived from MATLAB code published with
  //   Dianne P. O'Leary and Bert W. Rust,
  //   Variable Projection for Nonlinear Least Squares Problems,
  //   US National Inst. of Standards and Technology, 2010.
  public class Statistics {
    public double[] coeff;
    /// <summary>
    /// square-root of weighted residual norm squared divided by number of degrees of freedom.
    /// </summary>
    public double sigma;
    /// <summary>
    // weighted residual norm divided by the number of degrees of freedom.
    /// </summary>
    public double RMS;
    /// <summary>
    /// covariance matrix CovMx, which is sigma ^ 2 times the inverse of H'*H, where H = W *[dCoeff] contains the partial derivatives of wresid with respect to the parameters in coeff.
    /// </summary>
    public double[,] CovMx;
    /// <summary>
    /// estimated correlation matrix (numParam) x (numParam) for the parameters. 
    /// </summary>
    public double[,] CorMx;
    /// <summary>
    ///  Parameter standard deviation. The k-th element is the square root of the k-th main diagonal element of CovMx.
    /// </summary>
    public double[] std_param;
    /// <summary>
    ///  parameter estimates divided by their standard deviations.
    /// </summary>
    public double[] t_ratio;
    /// <summary>
    /// p-Value for the t_ratio (two-tailed)
    /// </summary>
    public double[] p_value;
    /// <summary>
    /// The k-th component is the k-th component of the weighted residual, divided by its standard deviation.
    /// </summary>
    public double[] standardized_wresid;

    public ResultCollection AsResultCollection(IEnumerable<string> predictorNames) {
      var statResults = new ResultCollection();
      statResults.Add(new Result("Sigma", "Square-root of weighted residual norm squared divided by number of degrees of freedom.", new DoubleValue(sigma)));
      statResults.Add(new Result("RMS", "weighted residual norm divided by the number of degrees of freedom.", new DoubleValue(RMS)));
      statResults.Add(new Result("Correlations", "Correlation matrix of parameters", new DoubleMatrix(CorMx, predictorNames, predictorNames)));
      statResults.Add(new Result("Covariance", "Covariance matrix of parameters", new DoubleMatrix(CovMx, predictorNames, predictorNames)));

      var m = new string[coeff.Length, 4];
      for (int i = 0; i < coeff.Length; i++) {
        m[i, 0] = coeff[i].ToString("g4");
        m[i, 1] = std_param[i].ToString("g4");
        m[i, 2] = t_ratio[i].ToString("g4");
        m[i, 3] = p_value[i].ToString("g4");
      }
      var paramStats = new StringMatrix(m) { ColumnNames = new string[] { "Value", "St.dev", "t-ratio", "p-value" }, RowNames = predictorNames };
      paramStats.SortableView = true;
      statResults.Add(new Result("Parameter statistics", "", paramStats));

      var wres = new DataTable("Standardized weighted residuals");
      wres.Rows.Add(new DataRow("Standarized weighted residuals", string.Empty, standardized_wresid));
      statResults.Add(new Result("Standardized weighted residuals", "The the weighted residuals, divided by their standard deviations", wres));

      return statResults;
    }

    public static Statistics CalculateLinearModelStatistics(double[,] jac, double[] coeff, double[] resid, double[] weight = null) {
      return CalculateParameterStatistics(jac, coeff, resid, weight);
    }

    // see http://sia.webpopix.org/nonlinearRegression.html
    public static Statistics CalculateParameterStatistics(double[,] jac, double[] coeff, double[] resid, double[] weight = null) {
      var w = weight;
      int numParam = coeff.Length;
      int numRows = jac.GetLength(0);
      if (jac.GetLength(1) != numParam) throw new ArgumentException("size of jac and coeff do not match"); // numParams includes offset
      if (w == null) w = Enumerable.Repeat(1.0, numRows).ToArray();

      var wresid = new double[numRows];
      var wresid_norm2 = 0.0;
      for (int i = 0; i < numRows; i++) {
        wresid[i] = w[i] * resid[i];
        wresid_norm2 += wresid[i] * wresid[i];
      }

      //  Calculate sample variance, the norm-squared of the residual
      //     divided by the number of degrees of freedom.
      var sigma2 = wresid_norm2 / (numRows - numParam);

      // Compute sigma:
      //               square-root of weighted residual norm squared divided
      //               by number of degrees of freedom.
      var sigma = Math.Sqrt(sigma2);

      // Compute RMS = sigma^2:
      //               the weighted residual norm divided by the number of
      //               degrees of freedom.
      //               RMS = wresid_norm / sqrt(m-n+q)
      var RMS = sigma2;

      /////////////////////////////////////////////////////////////////////////
      //
      // Compute some additional statistical diagnostics for the
      // solution
      //
      /////////////////////////////////////////////////////////////////////////
      // Calculate the covariance matrix CovMx, which is sigma ^ 2 times the
      // inverse of H'*H, where
      //              H = W * Jac
      // contains the partial derivatives of wresid with
      // respect to the parameters in coeff.
      double[,] H = new double[numRows, numParam];
      for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numParam; j++)
          H[i, j] = w[i] * jac[i, j];
      }

      // pseudo-inverse using QR
      var qr = (double[,])H.Clone();
      alglib.rmatrixqr(ref qr, numRows, numParam, out double[] tau);
      alglib.rmatrixqrunpackq(qr, numRows, numParam, tau, numParam, out double[,] q);  // alglib does not have a way to left-apply Q using Householder representation. It should not matter here as numParam is usually small.
      alglib.rmatrixqrunpackr(qr, numRows, numParam, out double[,] r);
      // 
      // 
      // // identity
      var T2 = new double[numParam, numParam];
      for (int i = 0; i < numParam; i++) T2[i, i] = 1.0;

      alglib.rmatrixlefttrsm(numParam, numParam, r, 0, 0, true, false, 0, ref T2, 0, 0);
      var CovMx_QR = new double[numParam, numParam];
      alglib.rmatrixgemm(numParam, numParam, numParam, sigma2, T2, 0, 0, 0, T2, 0, 0, 1, 0.0, ref CovMx_QR, 0, 0);

      // pseudo inverse using truncated SVD
      alglib.rmatrixsvd(H, numRows, numParam, uneeded: 1, vtneeded: 1, additionalmemory: 2, out var wDiag, out var u, out var vt);
      var rank = 0;
      var eps = 2.2204e-16;

      while (rank < wDiag.Length && wDiag[rank] > numParam * eps * wDiag[0]) rank++;
      // pseudo inverse H+ = V_r (D_r)^(-1) (U_r)^T
      var DU = new double[rank, numRows]; // (D^)-1 (U_r)^T
      for (int i = 0; i < rank; i++) {
        for (int j = 0; j < numRows; j++) {
          DU[i, j] = (1.0 / wDiag[i]) * u[j, i];
        }
      }
      var H_pInv = new double[numParam, numRows];
      alglib.rmatrixgemm(numParam, numRows, rank, 1.0, vt, 0, 0, 1, DU, 0, 0, 0, 0.0, ref H_pInv, 0, 0);

      var CovMx = new double[numParam, numParam];
      alglib.rmatrixgemm(numParam, numParam, numRows, sigma2, H_pInv, 0, 0, 0, H_pInv, 0, 0, 1, 0.0, ref CovMx, 0, 0);

      // Compute Regression.CorMx:
      //               estimated correlation matrix (numParam) x (numParam) for the
      //               parameters. 
      var CorMx = new double[numParam, numParam];
      for (int i = 0; i < numParam; i++)
        for (int j = 0; j < numParam; j++) {
          CorMx[i, j] = 1.0 / Math.Sqrt(CovMx[i, i]) *
                                   CovMx[i, j] *
                                   1.0 / Math.Sqrt(CovMx[j, j]);
        }

      // Compute Regression.std_param:
      //               The k-th element is the square root of the k-th main
      //               diagonal element of CovMx.
      var std_param = new double[numParam];
      for (int i = 0; i < numParam; i++) std_param[i] = Math.Sqrt(CovMx[i, i]);

      // Compute Regression.t_ratio:
      //               parameter estimates divided by their standard deviations.
      var t_ratio = new double[numParam];
      Array.Copy(coeff, t_ratio, coeff.Length);
      for (int i = 0; i < numParam; i++) t_ratio[i] /= std_param[i];

      // compute p-value from t-ratio
      var pValue = new double[numParam];
      for (int i = 0; i < numParam; i++) {
        if (t_ratio[i] == 0.0) pValue[i] = 1.0;
        else pValue[i] = 2 * (1 - alglib.studenttdistr.studenttdistribution(numRows - numParam, Math.Abs(t_ratio[i]), null)); // degrees of freedom: https://en.wikipedia.org/wiki/T-statistic
      }

      // only for linear models?
      // Compute Regression.standardized_wresid:
      //               The k-th component is the k-th component of the
      //               weighted residual, divided by its standard deviation.
      //               Let X = W*[Phi],
      //                   h(k) = k-th main diagonal element of covariance
      //                          matrix for wresid
      //                        = k-th main diagonal element of X*inv(X'*X)*X'
      //                        = k-th main diagonal element of Qj*Qj'.
      //               Then the standard deviation is estimated by
      //               sigma*sqrt(1-h(k)).
      // var temp = new double[numRows];
      var standardized_wresid = new double[numRows];
      // for (int k = 0; k < numRows; k++) {
      //   for (int i = 0; i < numParam; i++) {
      //     temp[k] += q[k, i] * q[k, i];
      //   }
      //   standardized_wresid[k] = wresid[k] / (sigma * Math.Sqrt(1 - temp[k]));
      // }

      return new Statistics() {
        coeff = coeff,
        sigma = sigma,
        RMS = RMS,
        CorMx = CorMx,
        CovMx = CovMx,
        std_param = std_param,
        t_ratio = t_ratio,
        p_value = pValue,
        standardized_wresid = standardized_wresid
      };
    }
  }
}

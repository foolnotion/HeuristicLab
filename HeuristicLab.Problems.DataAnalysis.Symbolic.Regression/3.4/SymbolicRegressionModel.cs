#region License Information
/* HeuristicLab
 * Copyright (C) Heuristic and Evolutionary Algorithms Laboratory (HEAL)
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

using System;
using System.Collections.Generic;
using HeuristicLab.Common;
using HeuristicLab.Core;
using HeuristicLab.Encodings.SymbolicExpressionTreeEncoding;
using HEAL.Attic;
using HeuristicLab.Data;
using System.Linq;

namespace HeuristicLab.Problems.DataAnalysis.Symbolic.Regression {
  /// <summary>
  /// Represents a symbolic regression model
  /// </summary>
  [StorableType("2739C33E-4DDB-4285-9DFB-C056D900B2F2")]
  [Item(Name = "Symbolic Regression Model", Description = "Represents a symbolic regression model.")]
  public class SymbolicRegressionModel : SymbolicDataAnalysisModel, ISymbolicRegressionModel {
    [Storable]
    private string targetVariable;
    public string TargetVariable {
      get { return targetVariable; }
      set {
        if (string.IsNullOrEmpty(value) || targetVariable == value) return;
        targetVariable = value;
        OnTargetVariableChanged(this, EventArgs.Empty);
      }
    }

    [Storable]
    private readonly double[,] parameterCovariance;
    [Storable]
    private readonly double sigma;

    [StorableConstructor]
    protected SymbolicRegressionModel(StorableConstructorFlag _) : base(_) {
      targetVariable = string.Empty;
    }

    protected SymbolicRegressionModel(SymbolicRegressionModel original, Cloner cloner)
      : base(original, cloner) {
      this.targetVariable = original.targetVariable;
      this.parameterCovariance = original.parameterCovariance; // immutable
      this.sigma = original.sigma;
    }

    public SymbolicRegressionModel(string targetVariable, ISymbolicExpressionTree tree,
      ISymbolicDataAnalysisExpressionTreeInterpreter interpreter,
      double lowerEstimationLimit = double.MinValue, double upperEstimationLimit = double.MaxValue, double[,] parameterCovariance = null, double sigma = 0.0)
      : base(tree, interpreter, lowerEstimationLimit, upperEstimationLimit) {
      this.targetVariable = targetVariable;
      if (parameterCovariance != null)
        this.parameterCovariance = (double[,])parameterCovariance.Clone();
      this.sigma = sigma;
    }

    public override IDeepCloneable Clone(Cloner cloner) {
      return new SymbolicRegressionModel(this, cloner);
    }

    public IEnumerable<double> GetEstimatedValues(IDataset dataset, IEnumerable<int> rows) {
      return Interpreter.GetSymbolicExpressionTreeValues(SymbolicExpressionTree, dataset, rows)
        .LimitToRange(LowerEstimationLimit, UpperEstimationLimit);
    }

    public IEnumerable<double> GetEstimatedVariances(IDataset dataset, IEnumerable<int> rows) {
      // must work with a copy because we change tree nodes
      var treeCopy = (ISymbolicExpressionTree)SymbolicExpressionTree.Clone();
      // uses sampling to produce prediction intervals
      alglib.hqrndseed(31415, 926535, out var state);
      var cov = parameterCovariance;
      if (cov == null || cov.Length == 0) return rows.Select(_ => 0.0);
      var n = 30;
      var M = rows.Select(_ => new double[n]).ToArray();
      var paramNodes = new List<ISymbolicExpressionTreeNode>();
      var coeffList = new List<double>();
      // HACK: skip linear scaling parameters because the analyzer doesn't use them (and they are likely correlated with the remaining parameters)
      // only works with linear scaling
      if (!(treeCopy.Root.GetSubtree(0).GetSubtree(0).Symbol is Addition) ||
          !(treeCopy.Root.GetSubtree(0).GetSubtree(0).GetSubtree(0).Symbol is Multiplication)) 
        throw new NotImplementedException("prediction intervals are implemented only for linear scaling");

      foreach (var node in treeCopy.Root.GetSubtree(0).GetSubtree(0).IterateNodesPostfix()) {
        if (node is ConstantTreeNode constNode) {
          paramNodes.Add(constNode);
          coeffList.Add(constNode.Value);
        } else if (node is VariableTreeNode varNode) {
          paramNodes.Add(varNode);
          coeffList.Add(varNode.Weight);
        }
      }
      var coeff = coeffList.ToArray();
      var numParams = coeff.Length;
      if (cov.GetLength(0) != numParams) throw new InvalidProgramException();

      // TODO: probably we do not need to sample but can instead use a first-order or second-order approximation of f
      // see http://sia.webpopix.org/nonlinearRegression.html
      // also see https://rmazing.wordpress.com/2013/08/26/predictnls-part-2-taylor-approximation-confidence-intervals-for-nls-models/
      // https://www.rdocumentation.org/packages/propagate/versions/1.0-4/topics/predictNLS
      double[] p = new double[numParams];
      for (int i = 0; i < 30; i++) {
        // sample and update parameter vector delta is 
        alglib.hqrndnormalv(state, numParams, out var delta);
        alglib.rmatrixmv(numParams, numParams, cov, 0, 0, 0, delta, 0, ref p, 0);
        for (int j = 0; j < numParams; j++) {
          if (paramNodes[j] is ConstantTreeNode constNode) constNode.Value = coeff[j] + p[j];
          else if (paramNodes[j] is VariableTreeNode varNode) varNode.Weight = coeff[j] + p[j];
        }
        var r = 0;
        var estimatedValues = Interpreter.GetSymbolicExpressionTreeValues(treeCopy, dataset, rows).LimitToRange(LowerEstimationLimit, UpperEstimationLimit);

        foreach (var pred in estimatedValues) {
          M[r++][i] = pred;
        }
      }

      // reset parameters
      for (int j = 0; j < numParams; j++) {
        if (paramNodes[j] is ConstantTreeNode constNode) constNode.Value = coeff[j];
        else if (paramNodes[j] is VariableTreeNode varNode) varNode.Weight = coeff[j];
      }
      var sigma2 = sigma * sigma;
      return M.Select(M_i => M_i.Variance() + sigma2).ToArray();
    }

    public ISymbolicRegressionSolution CreateRegressionSolution(IRegressionProblemData problemData) {
      return new SymbolicRegressionSolution(this, new RegressionProblemData(problemData));
    }
    IRegressionSolution IRegressionModel.CreateRegressionSolution(IRegressionProblemData problemData) {
      return CreateRegressionSolution(problemData);
    }

    public void Scale(IRegressionProblemData problemData) {
      Scale(problemData, problemData.TargetVariable);
    }

    public virtual bool IsProblemDataCompatible(IRegressionProblemData problemData, out string errorMessage) {
      return RegressionModel.IsProblemDataCompatible(this, problemData, out errorMessage);
    }

    public override bool IsProblemDataCompatible(IDataAnalysisProblemData problemData, out string errorMessage) {
      if (problemData == null) throw new ArgumentNullException("problemData", "The provided problemData is null.");
      var regressionProblemData = problemData as IRegressionProblemData;
      if (regressionProblemData == null)
        throw new ArgumentException("The problem data is not compatible with this symbolic regression model. Instead a " + problemData.GetType().GetPrettyName() + " was provided.", "problemData");
      return IsProblemDataCompatible(regressionProblemData, out errorMessage);
    }

    #region events
    public event EventHandler TargetVariableChanged;
    private void OnTargetVariableChanged(object sender, EventArgs args) {
      var changed = TargetVariableChanged;
      if (changed != null)
        changed(sender, args);
    }
    #endregion
  }
}

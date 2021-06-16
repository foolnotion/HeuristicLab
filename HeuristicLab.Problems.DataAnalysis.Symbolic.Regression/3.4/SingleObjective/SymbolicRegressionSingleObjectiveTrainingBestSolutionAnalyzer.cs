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

using HeuristicLab.Common;
using HeuristicLab.Core;
using HeuristicLab.Encodings.SymbolicExpressionTreeEncoding;
using HeuristicLab.Parameters;
using HEAL.Attic;
using HeuristicLab.Data;
using System.Collections.Generic;
using System;
using System.Linq;
using HeuristicLab.Analysis.Statistics;

namespace HeuristicLab.Problems.DataAnalysis.Symbolic.Regression {
  /// <summary>
  /// An operator that analyzes the training best symbolic regression solution for single objective symbolic regression problems.
  /// </summary>
  [Item("SymbolicRegressionSingleObjectiveTrainingBestSolutionAnalyzer", "An operator that analyzes the training best symbolic regression solution for single objective symbolic regression problems.")]
  [StorableType("85786F8E-F84D-4909-9A66-620668B0C7FB")]
  public sealed class SymbolicRegressionSingleObjectiveTrainingBestSolutionAnalyzer : SymbolicDataAnalysisSingleObjectiveTrainingBestSolutionAnalyzer<ISymbolicRegressionSolution>,
  ISymbolicDataAnalysisInterpreterOperator, ISymbolicDataAnalysisBoundedOperator {
    private const string ProblemDataParameterName = "ProblemData";
    private const string SymbolicDataAnalysisTreeInterpreterParameterName = "SymbolicDataAnalysisTreeInterpreter";
    private const string EstimationLimitsParameterName = "EstimationLimits";
    #region parameter properties
    public ILookupParameter<IRegressionProblemData> ProblemDataParameter {
      get { return (ILookupParameter<IRegressionProblemData>)Parameters[ProblemDataParameterName]; }
    }
    public ILookupParameter<ISymbolicDataAnalysisExpressionTreeInterpreter> SymbolicDataAnalysisTreeInterpreterParameter {
      get { return (ILookupParameter<ISymbolicDataAnalysisExpressionTreeInterpreter>)Parameters[SymbolicDataAnalysisTreeInterpreterParameterName]; }
    }
    public IValueLookupParameter<DoubleLimit> EstimationLimitsParameter {
      get { return (IValueLookupParameter<DoubleLimit>)Parameters[EstimationLimitsParameterName]; }
    }
    #endregion

    [StorableConstructor]
    private SymbolicRegressionSingleObjectiveTrainingBestSolutionAnalyzer(StorableConstructorFlag _) : base(_) { }
    private SymbolicRegressionSingleObjectiveTrainingBestSolutionAnalyzer(SymbolicRegressionSingleObjectiveTrainingBestSolutionAnalyzer original, Cloner cloner) : base(original, cloner) { }
    public SymbolicRegressionSingleObjectiveTrainingBestSolutionAnalyzer()
      : base() {
      Parameters.Add(new LookupParameter<IRegressionProblemData>(ProblemDataParameterName, "The problem data for the symbolic regression solution."));
      Parameters.Add(new LookupParameter<ISymbolicDataAnalysisExpressionTreeInterpreter>(SymbolicDataAnalysisTreeInterpreterParameterName, "The symbolic data analysis tree interpreter for the symbolic expression tree."));
      Parameters.Add(new ValueLookupParameter<DoubleLimit>(EstimationLimitsParameterName, "The lower and upper limit for the estimated values produced by the symbolic regression model."));
    }
    public override IDeepCloneable Clone(Cloner cloner) {
      return new SymbolicRegressionSingleObjectiveTrainingBestSolutionAnalyzer(this, cloner);
    }

    protected override ISymbolicRegressionSolution CreateSolution(ISymbolicExpressionTree bestTree, double bestQuality) {

      // HACK: create model first for scaling, then calculate statistics and create a new model with prediction intervals
      var model = new SymbolicRegressionModel(ProblemDataParameter.ActualValue.TargetVariable,
        (ISymbolicExpressionTree)bestTree.Clone(),
        SymbolicDataAnalysisTreeInterpreterParameter.ActualValue,
        EstimationLimitsParameter.ActualValue.Lower,
        EstimationLimitsParameter.ActualValue.Upper);
      if (ApplyLinearScalingParameter.ActualValue.Value) model.Scale(ProblemDataParameter.ActualValue);

      // use scaled tree
      CalculateParameterCovariance(model.SymbolicExpressionTree, ProblemDataParameter.ActualValue, SymbolicDataAnalysisTreeInterpreterParameter.ActualValue, out var cov, out var sigma);
      var predIntervalModel = new SymbolicRegressionModel(ProblemDataParameter.ActualValue.TargetVariable,
        (ISymbolicExpressionTree)model.SymbolicExpressionTree.Clone(),
        SymbolicDataAnalysisTreeInterpreterParameter.ActualValue,
        EstimationLimitsParameter.ActualValue.Lower,
        EstimationLimitsParameter.ActualValue.Upper, parameterCovariance: cov, sigma: sigma);

      return new SymbolicRegressionSolution(predIntervalModel, (IRegressionProblemData)ProblemDataParameter.ActualValue.Clone());
    }

    private void CalculateParameterCovariance(ISymbolicExpressionTree tree, IRegressionProblemData problemData, ISymbolicDataAnalysisExpressionTreeInterpreter interpreter, out double[,] cov, out double sigma) {
      var y_pred = interpreter.GetSymbolicExpressionTreeValues(tree, problemData.Dataset, problemData.TrainingIndices).ToArray();
      var residuals = problemData.TargetVariableTrainingValues.Zip(y_pred, (yi, y_pred_i) => yi - y_pred_i).ToArray();

      var paramNodes = new List<ISymbolicExpressionTreeNode>();
      var coeffList = new List<double>();
      foreach (var node in tree.IterateNodesPostfix()) {
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

      var rows = problemData.TrainingIndices.ToArray();
      var dcoeff = new double[rows.Length, numParams];
      TreeToAutoDiffTermConverter.TryConvertToAutoDiff(tree, makeVariableWeightsVariable: true, addLinearScalingTerms: false,
        out var parameters, out var initialConstants, out var func, out var func_grad);
      if (initialConstants.Zip(coeff, (ici, coi) => ici != coi).Any(t => t)) throw new InvalidProgramException();
      var ds = problemData.Dataset;
      var x_r = new double[parameters.Count];
      for (int r = 0; r < rows.Length; r++) {
        // copy row
        for (int c = 0; c < parameters.Count; c++) {
          x_r[c] = ds.GetDoubleValue(parameters[c].variableName, rows[r]);
        }
        var tup = func_grad(coeff, x_r);
        for (int c = 0; c < numParams; c++) {
          dcoeff[r, c] = tup.Item1[c];
        }
      }

      var stats = Statistics.CalculateLinearModelStatistics(dcoeff, coeff, residuals);
      cov = stats.CovMx;
      sigma = stats.sigma;
    }
  }
}

﻿#region License Information
/* HeuristicLab
 * Copyright (C) 2002-2011 Heuristic and Evolutionary Algorithms Laboratory (HEAL)
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
using System.Linq;
using System.Windows.Forms;
using HeuristicLab.Core.Views;
using HeuristicLab.Data;
using HeuristicLab.Data.Views;
using HeuristicLab.MainForm;
using HeuristicLab.MainForm.WindowsForms;

namespace HeuristicLab.Problems.DataAnalysis.Views {
  [View("Estimated Values")]
  [Content(typeof(IRegressionSolution))]
  public partial class RegressionSolutionEstimatedValuesView : ItemView, IRegressionSolutionEvaluationView {
    private const string TARGETVARIABLE_SERIES_NAME = "TargetVariable";
    private const string ESTIMATEDVALUES_SERIES_NAME = "EstimatedValues";

    public new IRegressionSolution Content {
      get { return (IRegressionSolution)base.Content; }
      set {
        base.Content = value;
      }
    }

    private StringConvertibleMatrixView matrixView;

    public RegressionSolutionEstimatedValuesView()
      : base() {
      InitializeComponent();
      matrixView = new StringConvertibleMatrixView();
      matrixView.ShowRowsAndColumnsTextBox = false;
      matrixView.ShowStatisticalInformation = false;
      matrixView.Dock = DockStyle.Fill;
      this.Controls.Add(matrixView);
    }

    #region events
    protected override void RegisterContentEvents() {
      base.RegisterContentEvents();
      Content.ModelChanged += new EventHandler(Content_ModelChanged);
      Content.ProblemDataChanged += new EventHandler(Content_ProblemDataChanged);
    }

    protected override void DeregisterContentEvents() {
      base.DeregisterContentEvents();
      Content.ModelChanged -= new EventHandler(Content_ModelChanged);
      Content.ProblemDataChanged -= new EventHandler(Content_ProblemDataChanged);
    }

    private void Content_ProblemDataChanged(object sender, EventArgs e) {
      OnContentChanged();
    }

    private void Content_ModelChanged(object sender, EventArgs e) {
      OnContentChanged();
    }

    protected override void OnContentChanged() {
      base.OnContentChanged();
      UpdateEstimatedValues();
    }

    private void UpdateEstimatedValues() {
      if (InvokeRequired) Invoke((Action)UpdateEstimatedValues);
      else {
        DoubleMatrix matrix = null;
        if (Content != null) {
          double[,] values = new double[Content.ProblemData.Dataset.Rows, 4];

          double[] target = Content.ProblemData.Dataset.GetVariableValues(Content.ProblemData.TargetVariable);
          double[] estimated = Content.EstimatedValues.ToArray();
          for (int row = 0; row < target.Length; row++) {
            values[row, 0] = target[row];
            values[row, 1] = estimated[row];
            values[row, 2] = Math.Abs(estimated[row] - target[row]);
            values[row, 3] = Math.Abs(values[row, 2] / target[row]);
          }

          matrix = new DoubleMatrix(values);
          matrix.ColumnNames = new string[] { TARGETVARIABLE_SERIES_NAME, ESTIMATEDVALUES_SERIES_NAME, "Absolute Error", "Relative Error" };
        }
        matrixView.Content = matrix;
      }
    }
    #endregion
  }
}

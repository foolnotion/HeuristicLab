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
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using HeuristicLab.Common;
using HeuristicLab.Encodings.SymbolicExpressionTreeEncoding;
using HeuristicLab.MainForm;
using HeuristicLab.MainForm.WindowsForms;

namespace HeuristicLab.Problems.DataAnalysis.Symbolic.Views {
  [View("Response Function View")]
  [Content(typeof(RegressionSolutionBase), false)]
  public partial class SymbolicDataAnalysisSolutionResponseFunctionView : AsynchronousContentView {
    private Dictionary<string, List<ISymbolicExpressionTreeNode>> variableNodes;
    private ISymbolicExpressionTree clonedTree;
    private Dictionary<string, double> medianValues;
    public SymbolicDataAnalysisSolutionResponseFunctionView() {
      InitializeComponent();
      this.variableNodes = new Dictionary<string, List<ISymbolicExpressionTreeNode>>();
      medianValues = new Dictionary<string, double>();
      this.Caption = "Response Function View";
    }

    public new ISymbolicDataAnalysisSolution Content {
      get { return (ISymbolicDataAnalysisSolution)base.Content; }
      set { base.Content = value; }
    }

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

    private void Content_ModelChanged(object sender, EventArgs e) {
      OnModelChanged();
    }
    private void Content_ProblemDataChanged(object sender, EventArgs e) {
      OnProblemDataChanged();
    }

    protected virtual void OnModelChanged() {
      this.UpdateView();
    }

    protected virtual void OnProblemDataChanged() {
      this.UpdateView();
    }

    protected override void OnContentChanged() {
      base.OnContentChanged();
      this.UpdateView();
    }

    private void UpdateView() {
      if (Content != null && Content.Model != null && Content.ProblemData != null) {
        var referencedVariables =
       (from varNode in Content.Model.SymbolicExpressionTree.IterateNodesPrefix().OfType<VariableTreeNode>()
        select varNode.VariableName)
         .Distinct()
         .OrderBy(x => x)
         .ToList();

        medianValues.Clear();
        foreach (var variableName in referencedVariables) {
          medianValues.Add(variableName, Content.ProblemData.Dataset.GetDoubleValues(variableName).Median());
        }

        comboBox.Items.Clear();
        comboBox.Items.AddRange(referencedVariables.ToArray());
        comboBox.SelectedIndex = 0;
      }
    }

    private void CreateSliders(IEnumerable<string> variableNames) {
      flowLayoutPanel.Controls.Clear();

      foreach (var variableName in variableNames) {
        var variableTrackbar = new VariableTrackbar(variableName,
                                                    Content.ProblemData.Dataset.GetDoubleValues(variableName));
        variableTrackbar.Size = new Size(variableTrackbar.Size.Width, flowLayoutPanel.Size.Height - 23);
        variableTrackbar.ValueChanged += TrackBarValueChanged;
        flowLayoutPanel.Controls.Add(variableTrackbar);
      }
    }

    private void TrackBarValueChanged(object sender, EventArgs e) {
      var trackBar = (VariableTrackbar)sender;
      string variableName = trackBar.VariableName;
      ChangeVariableValue(variableName, trackBar.Value);
    }

    private void ChangeVariableValue(string variableName, double value) {
      foreach (var constNode in variableNodes[variableName].Cast<ConstantTreeNode>())
        constNode.Value = value;

      UpdateChart();
    }

    private void UpdateChart() {
      string freeVariable = (string)comboBox.SelectedItem;
      IEnumerable<string> fixedVariables = comboBox.Items.OfType<string>()
        .Except(new string[] { freeVariable });

      var freeVariableValues = Content.ProblemData.Dataset.GetDoubleValues(freeVariable, Content.ProblemData.TrainingIndizes).ToArray();
      var responseValues = Content.Model.Interpreter.GetSymbolicExpressionTreeValues(clonedTree,
                                                                              Content.ProblemData.Dataset,
                                                                              Content.ProblemData.TrainingIndizes).ToArray();
      Array.Sort(freeVariableValues, responseValues);
      responseChart.Series["Model Response"].Points.DataBindXY(freeVariableValues, responseValues);
    }

    private ConstantTreeNode MakeConstantTreeNode(double value) {
      Constant constant = new Constant();
      constant.MinValue = value - 1;
      constant.MaxValue = value + 1;
      ConstantTreeNode constantTreeNode = (ConstantTreeNode)constant.CreateTreeNode();
      constantTreeNode.Value = value;
      return constantTreeNode;
    }

    private void ComboBoxSelectedIndexChanged(object sender, EventArgs e) {
      string freeVariable = (string)comboBox.SelectedItem;
      IEnumerable<string> fixedVariables = comboBox.Items.OfType<string>()
        .Except(new string[] { freeVariable });

      variableNodes.Clear();
      clonedTree = (ISymbolicExpressionTree)Content.Model.SymbolicExpressionTree.Clone();

      foreach (var varNode in clonedTree.IterateNodesPrefix().OfType<VariableTreeNode>()) {
        if (fixedVariables.Contains(varNode.VariableName)) {
          if (!variableNodes.ContainsKey(varNode.VariableName))
            variableNodes.Add(varNode.VariableName, new List<ISymbolicExpressionTreeNode>());

          int childIndex = varNode.Parent.IndexOfSubtree(varNode);
          var replacementNode = MakeConstantTreeNode(medianValues[varNode.VariableName]);
          var parent = varNode.Parent;
          parent.RemoveSubtree(childIndex);
          parent.InsertSubtree(childIndex, replacementNode);
          variableNodes[varNode.VariableName].Add(replacementNode);
        }
      }

      CreateSliders(fixedVariables);
      UpdateChart();
    }
  }
}

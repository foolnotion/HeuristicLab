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

using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using HeuristicLab.Common;
using HeuristicLab.Encodings.SymbolicExpressionTreeEncoding;
using HeuristicLab.Encodings.SymbolicExpressionTreeEncoding.Views;
using HeuristicLab.MainForm.WindowsForms;

namespace HeuristicLab.Problems.DataAnalysis.Symbolic.Views {
  public abstract partial class InteractiveSymbolicDataAnalysisSolutionSimplifierView : AsynchronousContentView {
    private Dictionary<ISymbolicExpressionTreeNode, ISymbolicExpressionTreeNode> foldedNodes;
    private Dictionary<ISymbolicExpressionTreeNode, ISymbolicExpressionTreeNode> changedNodes;
    private Dictionary<ISymbolicExpressionTreeNode, double> nodeImpacts;
    private enum TreeState { Valid, Invalid }
    private TreeState treeState;

    public InteractiveSymbolicDataAnalysisSolutionSimplifierView() {
      InitializeComponent();
      foldedNodes = new Dictionary<ISymbolicExpressionTreeNode, ISymbolicExpressionTreeNode>();
      changedNodes = new Dictionary<ISymbolicExpressionTreeNode, ISymbolicExpressionTreeNode>();
      nodeImpacts = new Dictionary<ISymbolicExpressionTreeNode, double>();
      this.Caption = "Interactive Solution Simplifier";

      // initialize the tree modifier that will be used to perform edit operations over the tree
      treeChart.ModifyTree = Modify;
    }

    /// <summary>
    /// Remove, Replace or Insert subtrees
    /// </summary>
    /// <param name="tree">The symbolic expression tree</param>
    /// <param name="node">The insertion point (the parent node who will receive a new child)</param>
    /// <param name="oldChild">The subtree to be replaced</param>
    /// <param name="newChild">The replacement subtree</param>
    /// <param name="removeSubtree">Flag used to indicate if whole subtrees should be removed (default behavior), or just the subtree root</param>
    private void Modify(ISymbolicExpressionTree tree, ISymbolicExpressionTreeNode node, ISymbolicExpressionTreeNode oldChild, ISymbolicExpressionTreeNode newChild,
                        bool removeSubtree = true) {
      if (oldChild == null && newChild == null) throw new ArgumentException();
      if (oldChild == null) { // insertion operation
        node.AddSubtree(newChild);
        newChild.Parent = node;
        treeChart.Tree = tree; // because a new node is present in the tree, the visualNodes dictionary needs to be updated
      } else if (newChild == null) { // removal operation
        // use switch instead of if/else purely for aesthetical reasons (to avoid nested ifs and elses)
        switch (removeSubtree) {
          case true:
            // remove the whole subtree
            node.RemoveSubtree(node.IndexOfSubtree(oldChild));
            if (oldChild.SubtreeCount > 0)
              foreach (var subtree in oldChild.IterateNodesBreadth()) {
                changedNodes.Remove(subtree);
                foldedNodes.Remove(subtree);
              }
            break;
          case false:
            // only remove the current node and try to preserve its subtrees
            node.RemoveSubtree(node.IndexOfSubtree(oldChild));
            if (oldChild.SubtreeCount > 0)
              for (int i = oldChild.SubtreeCount - 1; i >= 0; --i) {
                var subtree = oldChild.GetSubtree(i);
                oldChild.RemoveSubtree(i);
                node.AddSubtree(subtree);
              }
            break;
        }
        changedNodes.Remove(oldChild);
        foldedNodes.Remove(oldChild);
      } else { // replacement operation
        var replacementIndex = node.IndexOfSubtree(oldChild);
        node.RemoveSubtree(replacementIndex);
        node.InsertSubtree(replacementIndex, newChild);
        newChild.Parent = node;
        if (changedNodes.ContainsKey(oldChild)) {
          changedNodes.Add(newChild, changedNodes[oldChild]); // so that on double click the original node is restored
          changedNodes.Remove(oldChild);
        } else {
          changedNodes.Add(newChild, oldChild);
        }
      }

      if (IsValid(tree)) {
        treeState = TreeState.Valid;
        UpdateModel(Content.Model.SymbolicExpressionTree);
        btnOptimizeConstants.Enabled = true;
        btnSimplify.Enabled = true;
        treeStatusValue.Text = "Valid";
        treeStatusValue.ForeColor = Color.Green;
        this.Refresh();
      } else {
        treeState = TreeState.Invalid;
        btnOptimizeConstants.Enabled = true;
        btnSimplify.Enabled = true;
        treeStatusValue.Text = "Invalid";
        treeStatusValue.ForeColor = Color.Red;
        treeChart.Repaint();
        this.Refresh();
      }
      foreach (var changedNode in changedNodes.Keys) {
        var visualNode = treeChart.GetVisualSymbolicExpressionTreeNode(changedNode);
        visualNode.LineColor = Color.DodgerBlue;
        treeChart.RepaintNode(visualNode);
      }
    }

    private static bool IsValid(ISymbolicExpressionTree tree) {
      return !tree.IterateNodesPostfix().Any(node => node.SubtreeCount < node.Symbol.MinimumArity || node.SubtreeCount > node.Symbol.MaximumArity);
    }

    public new ISymbolicDataAnalysisSolution Content {
      get { return (ISymbolicDataAnalysisSolution)base.Content; }
      set { base.Content = value; }
    }

    protected override void RegisterContentEvents() {
      base.RegisterContentEvents();
      Content.ModelChanged += Content_Changed;
      Content.ProblemDataChanged += Content_Changed;
    }
    protected override void DeregisterContentEvents() {
      base.DeregisterContentEvents();
      Content.ModelChanged -= Content_Changed;
      Content.ProblemDataChanged -= Content_Changed;
    }

    private void Content_Changed(object sender, EventArgs e) {
      UpdateView();
    }

    protected override void OnContentChanged() {
      base.OnContentChanged();
      foldedNodes = new Dictionary<ISymbolicExpressionTreeNode, ISymbolicExpressionTreeNode>();
      UpdateView();
      viewHost.Content = this.Content;
    }

    private void UpdateView() {
      if (Content == null || Content.Model == null || Content.ProblemData == null) return;
      var tree = Content.Model.SymbolicExpressionTree;
      treeChart.Tree = tree.Root.SubtreeCount > 1 ? new SymbolicExpressionTree(tree.Root) : new SymbolicExpressionTree(tree.Root.GetSubtree(0).GetSubtree(0));

      var replacementValues = CalculateReplacementValues(tree);
      foreach (var pair in replacementValues.Where(pair => !(pair.Key is ConstantTreeNode))) {
        foldedNodes[pair.Key] = MakeConstantTreeNode(pair.Value);
      }

      nodeImpacts = CalculateImpactValues(tree);
      PaintNodeImpacts();
    }

    protected abstract Dictionary<ISymbolicExpressionTreeNode, double> CalculateReplacementValues(ISymbolicExpressionTree tree);
    protected abstract Dictionary<ISymbolicExpressionTreeNode, double> CalculateImpactValues(ISymbolicExpressionTree tree);
    protected abstract void UpdateModel(ISymbolicExpressionTree tree);

    private static ConstantTreeNode MakeConstantTreeNode(double value) {
      var constant = new Constant { MinValue = value - 1, MaxValue = value + 1 };
      var constantTreeNode = (ConstantTreeNode)constant.CreateTreeNode();
      constantTreeNode.Value = value;
      return constantTreeNode;
    }

    private void treeChart_SymbolicExpressionTreeNodeClicked(object sender, MouseEventArgs e) {
      var visualNode = (VisualSymbolicExpressionTreeNode)sender;
      if (visualNode == null) return;
      var treeNode = visualNode.SymbolicExpressionTreeNode;
      if (changedNodes.ContainsKey(treeNode)) {
        visualNode.LineColor = Color.DodgerBlue;
      } else if (treeNode is ConstantTreeNode && foldedNodes.ContainsKey(treeNode)) {
        visualNode.LineColor = Color.DarkOrange;
      } else {
        visualNode.LineColor = Color.Black;
      }
      visualNode.TextColor = Color.Black;
      treeChart.RepaintNode(visualNode);
    }

    private void treeChart_SymbolicExpressionTreeNodeDoubleClicked(object sender, MouseEventArgs e) {
      if (treeState == TreeState.Invalid) return;
      var visualNode = (VisualSymbolicExpressionTreeNode)sender;
      var symbExprTreeNode = (SymbolicExpressionTreeNode)visualNode.SymbolicExpressionTreeNode;
      if (symbExprTreeNode == null) return;
      var tree = Content.Model.SymbolicExpressionTree;
      var parent = symbExprTreeNode.Parent;
      int indexOfSubtree = parent.IndexOfSubtree(symbExprTreeNode);
      if (changedNodes.ContainsKey(symbExprTreeNode)) {
        parent.RemoveSubtree(indexOfSubtree);
        ISymbolicExpressionTreeNode originalNode = changedNodes[symbExprTreeNode];
        parent.InsertSubtree(indexOfSubtree, originalNode);
        changedNodes.Remove(symbExprTreeNode);
        UpdateModel(tree);
      } else if (foldedNodes.ContainsKey(symbExprTreeNode)) {
        SwitchNodeWithReplacementNode(parent, indexOfSubtree);
        UpdateModel(tree);
      }
    }

    private void SwitchNodeWithReplacementNode(ISymbolicExpressionTreeNode parent, int subTreeIndex) {
      ISymbolicExpressionTreeNode subTree = parent.GetSubtree(subTreeIndex);
      parent.RemoveSubtree(subTreeIndex);
      if (foldedNodes.ContainsKey(subTree)) {
        var replacementNode = foldedNodes[subTree];
        parent.InsertSubtree(subTreeIndex, replacementNode);
        // exchange key and value 
        foldedNodes.Remove(subTree);
        foldedNodes.Add(replacementNode, subTree);
      }
    }

    private void PaintNodeImpacts() {
      var impacts = nodeImpacts.Values;
      double max = impacts.Max();
      double min = impacts.Min();
      foreach (ISymbolicExpressionTreeNode treeNode in Content.Model.SymbolicExpressionTree.IterateNodesPostfix()) {
        VisualSymbolicExpressionTreeNode visualTree = treeChart.GetVisualSymbolicExpressionTreeNode(treeNode);

        if (!(treeNode is ConstantTreeNode) && nodeImpacts.ContainsKey(treeNode)) {
          double impact = nodeImpacts[treeNode];

          // impact = 0 if no change
          // impact < 0 if new solution is better
          // impact > 0 if new solution is worse
          if (impact < 0.0) {
            // min is guaranteed to be < 0
            visualTree.FillColor = Color.FromArgb((int)(impact / min * 255), Color.Red);
          } else if (impact.IsAlmost(0.0)) {
            visualTree.FillColor = Color.White;
          } else {
            // max is guaranteed to be > 0
            visualTree.FillColor = Color.FromArgb((int)(impact / max * 255), Color.Green);
          }
          visualTree.ToolTip += Environment.NewLine + "Node impact: " + impact;
          var constantReplacementNode = foldedNodes[treeNode] as ConstantTreeNode;
          if (constantReplacementNode != null) {
            visualTree.ToolTip += Environment.NewLine + "Replacement value: " + constantReplacementNode.Value;
          }
        }
        if (visualTree != null && treeNode is ConstantTreeNode && foldedNodes.ContainsKey(treeNode)) {
          visualTree.LineColor = Color.DarkOrange;
        }
      }
      // repaint nodes and refresh
      treeChart.RepaintNodes();
      treeChart.Refresh();
    }

    private void btnSimplify_Click(object sender, EventArgs e) {
      var simplifier = new SymbolicDataAnalysisExpressionTreeSimplifier();
      var simplifiedExpressionTree = simplifier.Simplify(Content.Model.SymbolicExpressionTree);
      UpdateModel(simplifiedExpressionTree);
    }

    protected abstract void btnOptimizeConstants_Click(object sender, EventArgs e);
  }
}

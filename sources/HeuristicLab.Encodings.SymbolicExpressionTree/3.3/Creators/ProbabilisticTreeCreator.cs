#region License Information
/* HeuristicLab
 * Copyright (C) 2002-2008 Heuristic and Evolutionary Algorithms Laboratory (HEAL)
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
using HeuristicLab.Random;
using System;

namespace HeuristicLab.Encodings.SymbolicExpressionTree {
  public class ProbabilisticTreeCreator : OperatorBase {
    private static int MAX_TRIES { get { return 100; } }

    public override string Description {
      get { return @"Generates a new random operator tree."; }
    }

    public ProbabilisticTreeCreator()
      : base() {
      AddVariableInfo(new VariableInfo("Random", "Uniform random number generator", typeof(MersenneTwister), VariableKind.In));
      AddVariableInfo(new VariableInfo("FunctionLibrary", "The function library containing all available functions", typeof(FunctionLibrary), VariableKind.In));
      AddVariableInfo(new VariableInfo("MinTreeSize", "The minimal allowed size of the tree", typeof(IntData), VariableKind.In));
      AddVariableInfo(new VariableInfo("MaxTreeSize", "The maximal allowed size of the tree", typeof(IntData), VariableKind.In));
      AddVariableInfo(new VariableInfo("MaxTreeHeight", "The maximal allowed height of the tree", typeof(IntData), VariableKind.In));
      AddVariableInfo(new VariableInfo("FunctionTree", "The created tree", typeof(IGeneticProgrammingModel), VariableKind.New | VariableKind.Out));
    }

    public override IOperation Apply(IScope scope) {
      IRandom random = GetVariableValue<IRandom>("Random", scope, true);
      FunctionLibrary funLibrary = GetVariableValue<FunctionLibrary>("FunctionLibrary", scope, true);
      int minTreeSize = GetVariableValue<IntData>("MinTreeSize", scope, true).Data;
      int maxTreeSize = GetVariableValue<IntData>("MaxTreeSize", scope, true).Data;
      int maxTreeHeight = GetVariableValue<IntData>("MaxTreeHeight", scope, true).Data;

      IFunctionTree root = Create(random, funLibrary, minTreeSize, maxTreeSize, maxTreeHeight);
      scope.AddVariable(new HeuristicLab.Core.Variable(scope.TranslateName("FunctionTree"), new GeneticProgrammingModel(root)));
      return Util.CreateInitializationOperation(TreeGardener.GetAllSubTrees(root), scope);
    }


    public static IFunctionTree Create(IRandom random, FunctionLibrary funLib, int minSize, int maxSize, int maxHeight) {
      int treeSize = random.Next(minSize, maxSize);
      IFunctionTree root = null;
      int tries = 0;
      TreeGardener gardener = new TreeGardener(random, funLib);
      do {
        try {
          root = gardener.PTC2(treeSize, maxHeight);
        }
        catch (ArgumentException) {
          // try a different size
          treeSize = random.Next(minSize, maxSize);
          tries = 0;
        }
        if (tries++ >= MAX_TRIES) {
          // try a different size
          treeSize = random.Next(minSize, maxSize);
          tries = 0;
        }
      } while (root == null || root.GetSize() > maxSize || root.GetHeight() > maxHeight);
      return root;
    }
  }
}
#region License Information
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

using HeuristicLab.Common;
using HeuristicLab.Core;
using HeuristicLab.Persistence.Default.CompositeSerializers.Storable;
using HeuristicLab.Problems.DataAnalysis;

namespace HeuristicLab.Algorithms.DataAnalysis {
  /// <summary>
  /// Represents a neural network ensemble solution for a regression problem which can be visualized in the GUI.
  /// </summary>
  [Item("NeuralNetworkEnsembleRegressionSolution", "Represents a neural network ensemble solution for a regression problem which can be visualized in the GUI.")]
  [StorableClass]
  public sealed class NeuralNetworkEnsembleRegressionSolution : RegressionSolution, INeuralNetworkEnsembleRegressionSolution {

    public new INeuralNetworkEnsembleModel Model {
      get { return (INeuralNetworkEnsembleModel)base.Model; }
      set { base.Model = value; }
    }

    [StorableConstructor]
    private NeuralNetworkEnsembleRegressionSolution(bool deserializing) : base(deserializing) { }
    private NeuralNetworkEnsembleRegressionSolution(NeuralNetworkEnsembleRegressionSolution original, Cloner cloner)
      : base(original, cloner) {
    }
    public NeuralNetworkEnsembleRegressionSolution(IRegressionProblemData problemData, INeuralNetworkEnsembleModel nnModel)
      : base(nnModel, problemData) {
      RecalculateResults();
    }

    public override IDeepCloneable Clone(Cloner cloner) {
      return new NeuralNetworkEnsembleRegressionSolution(this, cloner);
    }

    protected override void RecalculateResults() {
      CalculateResults();
    }
  }
}

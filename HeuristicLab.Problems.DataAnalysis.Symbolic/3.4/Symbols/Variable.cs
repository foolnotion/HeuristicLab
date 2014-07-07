#region License Information
/* HeuristicLab
 * Copyright (C) 2002-2013 Heuristic and Evolutionary Algorithms Laboratory (HEAL)
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
using HeuristicLab.Persistence.Default.CompositeSerializers.Storable;
namespace HeuristicLab.Problems.DataAnalysis.Symbolic {
  [StorableClass]
  [Item("Variable", "Represents a variable value.")]
  public class Variable : Symbol {
    #region Properties
    [Storable]
    private double weightMu;
    public double WeightMu {
      get { return weightMu; }
      set {
        if (value != weightMu) {
          weightMu = value;
          OnChanged(EventArgs.Empty);
        }
      }
    }
    [Storable]
    private double weightSigma;
    public double WeightSigma {
      get { return weightSigma; }
      set {
        if (weightSigma < 0.0) throw new ArgumentException("Negative sigma is not allowed.");
        if (value != weightSigma) {
          weightSigma = value;
          OnChanged(EventArgs.Empty);
        }
      }
    }
    [Storable]
    private double weightManipulatorMu;
    public double WeightManipulatorMu {
      get { return weightManipulatorMu; }
      set {
        if (value != weightManipulatorMu) {
          weightManipulatorMu = value;
          OnChanged(EventArgs.Empty);
        }
      }
    }
    [Storable]
    private double weightManipulatorSigma;
    public double WeightManipulatorSigma {
      get { return weightManipulatorSigma; }
      set {
        if (weightManipulatorSigma < 0.0) throw new ArgumentException("Negative sigma is not allowed.");
        if (value != weightManipulatorSigma) {
          weightManipulatorSigma = value;
          OnChanged(EventArgs.Empty);
        }
      }
    }
    [Storable(DefaultValue = 0.0)]
    private double multiplicativeWeightManipulatorSigma;
    public double MultiplicativeWeightManipulatorSigma {
      get { return multiplicativeWeightManipulatorSigma; }
      set {
        if (multiplicativeWeightManipulatorSigma < 0.0) throw new ArgumentException("Negative sigma is not allowed.");
        if (value != multiplicativeWeightManipulatorSigma) {
          multiplicativeWeightManipulatorSigma = value;
          OnChanged(EventArgs.Empty);
        }
      }
    }
    private List<string> variableNames;
    [Storable]
    public IEnumerable<string> VariableNames {
      get { return variableNames; }
      set {
        if (value == null) throw new ArgumentNullException();
        variableNames.Clear();
        variableNames.AddRange(value);
        OnChanged(EventArgs.Empty);
      }
    }

    private List<string> allVariableNames;
    [Storable]
    public IEnumerable<string> AllVariableNames {
      get { return allVariableNames; }
      set {
        if (value == null) throw new ArgumentNullException();
        allVariableNames.Clear();
        allVariableNames.AddRange(value);
      }
    }

    public override bool Enabled {
      get {
        if (variableNames.Count == 0) return false;
        return base.Enabled;
      }
      set {
        if (variableNames.Count == 0) base.Enabled = false;
        else base.Enabled = value;
      }
    }

    private const int minimumArity = 0;
    private const int maximumArity = 0;

    public override int MinimumArity {
      get { return minimumArity; }
    }
    public override int MaximumArity {
      get { return maximumArity; }
    }
    #endregion

    [StorableHook(HookType.AfterDeserialization)]
    private void AfterDeserialization() {
      if (allVariableNames == null || (allVariableNames.Count == 0 && variableNames.Count > 0)) {
        allVariableNames = variableNames;
      }
    }

    [StorableConstructor]
    protected Variable(bool deserializing)
      : base(deserializing) {
      variableNames = new List<string>();
      allVariableNames = new List<string>();
    }
    protected Variable(Variable original, Cloner cloner)
      : base(original, cloner) {
      weightMu = original.weightMu;
      weightSigma = original.weightSigma;
      variableNames = new List<string>(original.variableNames);
      allVariableNames = new List<string>(original.allVariableNames);
      weightManipulatorMu = original.weightManipulatorMu;
      weightManipulatorSigma = original.weightManipulatorSigma;
      multiplicativeWeightManipulatorSigma = original.multiplicativeWeightManipulatorSigma;
    }
    public Variable() : this("Variable", "Represents a variable value.") { }
    public Variable(string name, string description)
      : base(name, description) {
      weightMu = 1.0;
      weightSigma = 1.0;
      weightManipulatorMu = 0.0;
      weightManipulatorSigma = 0.05;
      multiplicativeWeightManipulatorSigma = 0.03;
      variableNames = new List<string>();
      allVariableNames = new List<string>();
    }

    public override ISymbolicExpressionTreeNode CreateTreeNode() {
      return new VariableTreeNode(this);
    }

    public override IDeepCloneable Clone(Cloner cloner) {
      return new Variable(this, cloner);
    }
  }
}

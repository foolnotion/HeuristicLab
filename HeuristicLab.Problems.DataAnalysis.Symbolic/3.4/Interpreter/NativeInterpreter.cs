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
using System.Linq;
using System.Runtime.InteropServices;

using HEAL.Attic;

using HeuristicLab.Common;
using HeuristicLab.Core;
using HeuristicLab.Data;
using HeuristicLab.Encodings.SymbolicExpressionTreeEncoding;
using HeuristicLab.Parameters;
using HeuristicLab.Problems.DataAnalysis;

namespace HeuristicLab.Problems.DataAnalysis.Symbolic {
  [StorableType("91723319-8F15-4D33-B277-40AC7C7CF9AE")]
  [Item("NativeInterpreter", "Operator calling into native C++ code for tree interpretation.")]
  public class NativeInterpreter : ParameterizedNamedItem, ISymbolicDataAnalysisExpressionTreeInterpreter {
    private const string EvaluatedSolutionsParameterName = "EvaluatedSolutions";

    #region parameters
    public IFixedValueParameter<IntValue> EvaluatedSolutionsParameter {
      get { return (IFixedValueParameter<IntValue>)Parameters[EvaluatedSolutionsParameterName]; }
    }
    #endregion

    #region properties
    public int EvaluatedSolutions {
      get { return EvaluatedSolutionsParameter.Value.Value; }
      set { EvaluatedSolutionsParameter.Value.Value = value; }
    }
    #endregion

    public NativeInterpreter() {
      Parameters.Add(new FixedValueParameter<IntValue>(EvaluatedSolutionsParameterName, "A counter for the total number of solutions the interpreter has evaluated", new IntValue(0)));
    }

    [StorableConstructor]
    protected NativeInterpreter(StorableConstructorFlag _) : base(_) { }

    protected NativeInterpreter(NativeInterpreter original, Cloner cloner) : base(original, cloner) {
    }

    public override IDeepCloneable Clone(Cloner cloner) {
      return new NativeInterpreter(this, cloner);
    }
    public static NativeInstruction[] Compile(ISymbolicExpressionTree tree, IDataset dataset, Func<ISymbolicExpressionTreeNode, byte> opCodeMapper, out List<ISymbolicExpressionTreeNode> nodes) {
      var root = tree.Root.GetSubtree(0).GetSubtree(0);
      return Compile(root, dataset, opCodeMapper, out nodes);
    }

    public static NativeInstruction[] Compile(ISymbolicExpressionTreeNode root, IDataset dataset, Func<ISymbolicExpressionTreeNode, byte> opCodeMapper, out List<ISymbolicExpressionTreeNode> nodes) {
      if (cachedData == null || cachedDataset != dataset) {
        InitCache(dataset);
      }

      nodes = root.IterateNodesPrefix().ToList(); nodes.Reverse();
      var code = new NativeInstruction[nodes.Count];

      for (int i = 0; i < nodes.Count; ++i) {
        var node = nodes[i];
        code[i] = new NativeInstruction { Arity = (ushort)node.SubtreeCount, OpCode = opCodeMapper(node), Length = (ushort)node.GetLength(), Optimize = true };

        if (node is VariableTreeNode variable) {
          code[i].Value = variable.Weight;
          code[i].Data = cachedData[variable.VariableName].AddrOfPinnedObject();
        } else if (node is ConstantTreeNode constant) {
          code[i].Value = constant.Value;
        }
      }
      return code;
    }

    private readonly object syncRoot = new object();

    [ThreadStatic]
    private static Dictionary<string, GCHandle> cachedData;

    [ThreadStatic]
    private static IDataset cachedDataset;

    protected static readonly HashSet<byte> supportedOpCodes = new HashSet<byte>() {
      (byte)OpCode.Constant,
      (byte)OpCode.Variable,
      (byte)OpCode.Add,
      (byte)OpCode.Sub,
      (byte)OpCode.Mul,
      (byte)OpCode.Div,
      (byte)OpCode.Exp,
      (byte)OpCode.Log,
      (byte)OpCode.Sin,
      (byte)OpCode.Cos,
      (byte)OpCode.Tan,
      (byte)OpCode.Tanh,
      (byte)OpCode.Power,
      (byte)OpCode.Root,
      (byte)OpCode.SquareRoot,
      (byte)OpCode.Square,
      (byte)OpCode.CubeRoot,
      (byte)OpCode.Cube,
      (byte)OpCode.Absolute,
      (byte)OpCode.AnalyticQuotient
    };

    public IEnumerable<double> GetSymbolicExpressionTreeValues(ISymbolicExpressionTree tree, IDataset dataset, IEnumerable<int> rows) {
      return GetSymbolicExpressionTreeValues(tree, dataset, rows.ToArray());
    }
    
    private static void InitCache(IDataset dataset) {
      cachedDataset = dataset;
      // cache new data (but free old data first)
      if (cachedData != null) {
        foreach (var gch in cachedData.Values) {
          gch.Free();
        }
      }
      cachedData = new Dictionary<string, GCHandle>();
      foreach (var v in dataset.DoubleVariables) {
        var values = dataset.GetDoubleValues(v).ToArray();
        var gch = GCHandle.Alloc(values, GCHandleType.Pinned);
        cachedData[v] = gch;
      }
    }

    public void ClearState() {
      if (cachedData != null) {
        foreach (var gch in cachedData.Values) {
          gch.Free();
        }
        cachedData = null;
      }
      cachedDataset = null;
      EvaluatedSolutions = 0;
    }

    public void InitializeState() {
      ClearState();
    }

    public IEnumerable<double> GetSymbolicExpressionTreeValues(ISymbolicExpressionTree tree, IDataset dataset, int[] rows) {
      if (!rows.Any()) return Enumerable.Empty<double>();

      byte mapSupportedSymbols(ISymbolicExpressionTreeNode node) {
        var opCode = OpCodes.MapSymbolToOpCode(node);
        if (supportedOpCodes.Contains(opCode)) return opCode;
        else throw new NotSupportedException($"The native interpreter does not support {node.Symbol.Name}");
      };
      var code = Compile(tree, dataset, mapSupportedSymbols, out List<ISymbolicExpressionTreeNode> nodes);

      var result = new double[rows.Length];
      var options = new SolverOptions { /* not using any options here */ };

      var summary = new OptimizationSummary(); // also not used
      NativeWrapper.GetValues(code, rows, result, null, options, ref summary);

      // when evaluation took place without any error, we can increment the counter
      lock (syncRoot) {
        EvaluatedSolutions++;
      }

      return result;
    }
  }
}
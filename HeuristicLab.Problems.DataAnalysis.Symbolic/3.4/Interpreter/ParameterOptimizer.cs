using System;
using System.Collections.Generic;
using System.Linq;
using HEAL.Attic;
using HeuristicLab.Common;
using HeuristicLab.Core;
using HeuristicLab.Data;
using HeuristicLab.Encodings.SymbolicExpressionTreeEncoding;
using HeuristicLab.NativeInterpreter;
using HeuristicLab.Parameters;

namespace HeuristicLab.Problems.DataAnalysis.Symbolic {
  [StorableType("A624630B-0CEB-4D06-9B26-708987A7AE8F")]
  [Item("ParameterOptimizer", "Operator calling into native C++ code for tree interpretation.")]
  public sealed class ParameterOptimizer : ParameterizedNamedItem {
    private const string UseNonmonotonicStepsParameterName = "UseNonmonotonicSteps";
    private const string OptimizerIterationsParameterName = "OptimizerIterations";

    private const string MinimizerParameterName = "Minimizer";
    private const string LinearSolverParameterName = "LinearSolver";
    private const string TrustRegionStrategyParameterName = "TrustRegionStrategy";
    private const string DogLegParameterName = "DogLeg";
    private const string LineSearchDirectionParameterName = "LineSearchDirection";

    #region parameters
    public IFixedValueParameter<IntValue> OptimizerIterationsParameter {
      get { return (IFixedValueParameter<IntValue>)Parameters[OptimizerIterationsParameterName]; }
    }
    public IFixedValueParameter<BoolValue> UseNonmonotonicStepsParameter {
      get { return (IFixedValueParameter<BoolValue>)Parameters[UseNonmonotonicStepsParameterName]; }
    }
    public IFixedValueParameter<EnumValue<CeresTypes.Minimizer>> MinimizerTypeParameter {
      get { return (IFixedValueParameter<EnumValue<CeresTypes.Minimizer>>)Parameters[MinimizerParameterName]; }
    }
    public IFixedValueParameter<EnumValue<CeresTypes.LinearSolver>> LinearSolverTypeParameter {
      get { return (IFixedValueParameter<EnumValue<CeresTypes.LinearSolver>>)Parameters[LinearSolverParameterName]; }
    }
    public IFixedValueParameter<EnumValue<CeresTypes.TrustRegionStrategy>> TrustRegionStrategyTypeParameter {
      get { return (IFixedValueParameter<EnumValue<CeresTypes.TrustRegionStrategy>>)Parameters[TrustRegionStrategyParameterName]; }
    }
    public IFixedValueParameter<EnumValue<CeresTypes.DogLeg>> DogLegTypeParameter {
      get { return (IFixedValueParameter<EnumValue<CeresTypes.DogLeg>>)Parameters[DogLegParameterName]; }
    }
    public IFixedValueParameter<EnumValue<CeresTypes.LineSearchDirection>> LineSearchDirectionTypeParameter {
      get { return (IFixedValueParameter<EnumValue<CeresTypes.LineSearchDirection>>)Parameters[LineSearchDirectionParameterName]; }
    }
    #endregion

    #region parameter properties
    public int OptimizerIterations {
      get { return OptimizerIterationsParameter.Value.Value; }
      set { OptimizerIterationsParameter.Value.Value = value; }
    }
    public bool UseNonmonotonicSteps {
      get { return UseNonmonotonicStepsParameter.Value.Value; }
      set { UseNonmonotonicStepsParameter.Value.Value = value; }
    }
    private CeresTypes.Minimizer Minimizer {
      get { return MinimizerTypeParameter.Value.Value; }
      set { MinimizerTypeParameter.Value.Value = value; }
    }
    private CeresTypes.LinearSolver LinearSolver {
      get { return LinearSolverTypeParameter.Value.Value; }
      set { LinearSolverTypeParameter.Value.Value = value; }
    }
    private CeresTypes.TrustRegionStrategy TrustRegionStrategy {
      get { return TrustRegionStrategyTypeParameter.Value.Value; }
      set { TrustRegionStrategyTypeParameter.Value.Value = value; }
    }
    private CeresTypes.DogLeg DogLeg {
      get { return DogLegTypeParameter.Value.Value; }
      set { DogLegTypeParameter.Value.Value = value; }
    }
    private CeresTypes.LineSearchDirection LineSearchDirection {
      get { return LineSearchDirectionTypeParameter.Value.Value; }
      set { LineSearchDirectionTypeParameter.Value.Value = value; }
    }
    #endregion

    #region storable ctor and cloning
    [StorableConstructor]
    private ParameterOptimizer(StorableConstructorFlag _) : base(_) { }

    public ParameterOptimizer(ParameterOptimizer original, Cloner cloner) : base(original, cloner) { }

    public override IDeepCloneable Clone(Cloner cloner) {
      return new ParameterOptimizer(this, cloner);
    }
    #endregion

    public ParameterOptimizer() {
      Parameters.Add(new FixedValueParameter<EnumValue<CeresTypes.Minimizer>>(MinimizerParameterName, new EnumValue<CeresTypes.Minimizer>(CeresTypes.Minimizer.TRUST_REGION)));
      Parameters.Add(new FixedValueParameter<EnumValue<CeresTypes.LinearSolver>>(LinearSolverParameterName, new EnumValue<CeresTypes.LinearSolver>(CeresTypes.LinearSolver.DENSE_QR)));
      Parameters.Add(new FixedValueParameter<EnumValue<CeresTypes.TrustRegionStrategy>>(TrustRegionStrategyParameterName, new EnumValue<CeresTypes.TrustRegionStrategy>(CeresTypes.TrustRegionStrategy.LEVENBERG_MARQUARDT)));
      Parameters.Add(new FixedValueParameter<EnumValue<CeresTypes.DogLeg>>(DogLegParameterName, new EnumValue<CeresTypes.DogLeg>(CeresTypes.DogLeg.TRADITIONAL_DOGLEG)));
      Parameters.Add(new FixedValueParameter<EnumValue<CeresTypes.LineSearchDirection>>(LineSearchDirectionParameterName, new EnumValue<CeresTypes.LineSearchDirection>(CeresTypes.LineSearchDirection.STEEPEST_DESCENT)));
      Parameters.Add(new FixedValueParameter<IntValue>(OptimizerIterationsParameterName, "The number of iterations for the nonlinear least squares optimizer.", new IntValue(10)));
      Parameters.Add(new FixedValueParameter<BoolValue>(UseNonmonotonicStepsParameterName, "Allow the non linear least squares optimizer to make steps in parameter space that do not necessarily decrease the error, but might improve overall convergence.", new BoolValue(false)));
    }

    private static byte MapSupportedSymbols(ISymbolicExpressionTreeNode node) {
      var opCode = OpCodes.MapSymbolToOpCode(node);
      if (supportedOpCodes.Contains(opCode)) return opCode;
      else throw new NotSupportedException($"The native interpreter does not support {node.Symbol.Name}");
    }

    public static Dictionary<ISymbolicExpressionTreeNode, double> OptimizeTree(ISymbolicExpressionTree tree, IDataset dataset, IEnumerable<int> rows, string targetVariable, 
      HashSet<ISymbolicExpressionTreeNode> nodesToOptimize, SolverOptions options, ref OptimizationSummary summary) {
      var code = NativeInterpreter.Compile(tree, dataset, MapSupportedSymbols, out List<ISymbolicExpressionTreeNode> nodes);

      for (int i = 0; i < code.Length; ++i) {
        code[i].Optimize = nodesToOptimize.Contains(nodes[i]);
      }

      if (options.Iterations > 0) {
        var target = dataset.GetDoubleValues(targetVariable, rows).ToArray();
        var rowsArray = rows.ToArray();
        var result = new double[rowsArray.Length];

        NativeWrapper.GetValues(code, rowsArray, options, result, target, out summary);
      }
      return Enumerable.Range(0, code.Length).Where(i => nodes[i] is SymbolicExpressionTreeTerminalNode).ToDictionary(i => nodes[i], i => code[i].Value);
    }

    public Dictionary<ISymbolicExpressionTreeNode, double> OptimizeTree(ISymbolicExpressionTree tree, IDataset dataset, IEnumerable<int> rows, string targetVariable, 
      HashSet<ISymbolicExpressionTreeNode> nodesToOptimize = null) {
      var options = new SolverOptions {
        Iterations = OptimizerIterations,
        Minimizer = Minimizer,
        LinearSolver = LinearSolver,
        TrustRegionStrategy = TrustRegionStrategy,
        DogLeg = DogLeg,
        LineSearchDirection = LineSearchDirection,
        UseNonmonotonicSteps = UseNonmonotonicSteps ? 1 : 0
      };

      var summary = new OptimizationSummary();

      // if no nodes are specified, use all the nodes
      if (nodesToOptimize == null) {
        nodesToOptimize = new HashSet<ISymbolicExpressionTreeNode>(tree.IterateNodesPrefix().Where(x => x is SymbolicExpressionTreeTerminalNode));
      }

      return OptimizeTree(tree, dataset, rows, targetVariable, nodesToOptimize, options, ref summary);
    }

    public static Dictionary<ISymbolicExpressionTreeNode, double> OptimizeTree(ISymbolicExpressionTree[] terms, IDataset dataset, IEnumerable<int> rows, string targetVariable, HashSet<ISymbolicExpressionTreeNode> nodesToOptimize, SolverOptions options, double[] coeff, ref OptimizationSummary summary) {
      if (options.Iterations == 0) {
        // throw exception? set iterations to 100? return empty dictionary?
        return new Dictionary<ISymbolicExpressionTreeNode, double>();
      }

      var termIndices = new int[terms.Length];
      var totalCodeSize = 0;
      var totalCode = new List<NativeInstruction>();
      var totalNodes = new List<ISymbolicExpressionTreeNode>();

      // internally the native wrapper takes a single array of NativeInstructions where the indices point to the individual terms
      for (int i = 0; i < terms.Length; ++i) {
        var code = NativeInterpreter.Compile(terms[i], dataset, MapSupportedSymbols, out List<ISymbolicExpressionTreeNode> nodes);
        for (int j = 0; j < code.Length; ++j) {
          code[j].Optimize = nodesToOptimize.Contains(nodes[j]);
        }
        totalCode.AddRange(code);
        totalNodes.AddRange(nodes);

        termIndices[i] = code.Length + totalCodeSize - 1;
        totalCodeSize += code.Length;
      }
      var target = dataset.GetDoubleValues(targetVariable, rows).ToArray();
      var rowsArray = rows.ToArray();
      var result = new double[rowsArray.Length];
      var codeArray = totalCode.ToArray();

      NativeWrapper.GetValuesVarPro(codeArray, termIndices, rowsArray, coeff, options, result, target, out summary);
      return Enumerable.Range(0, totalCodeSize).Where(i => codeArray[i].Optimize).ToDictionary(i => totalNodes[i], i => codeArray[i].Value);
    }

    private static readonly HashSet<byte> supportedOpCodes = new HashSet<byte>() {
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
      // (byte)OpCode.Power, // these symbols are handled differently in the NativeInterpreter than in HL
      // (byte)OpCode.Root,
      (byte)OpCode.SquareRoot,
      (byte)OpCode.Square,
      (byte)OpCode.CubeRoot,
      (byte)OpCode.Cube,
      (byte)OpCode.Absolute,
      (byte)OpCode.AnalyticQuotient
    };
  }
}

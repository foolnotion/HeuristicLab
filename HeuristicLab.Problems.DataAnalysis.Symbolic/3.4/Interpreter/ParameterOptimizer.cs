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
  public class ParameterOptimizer : NativeInterpreter {
    private const string UseNonmonotonicStepsParameterName = "UseNonmonotonicSteps";
    private const string OptimizerIterationsParameterName = "OptimizerIterations";

    private const string MinimizerTypeParameterName = "MinimizerType";
    private const string LinearSolverTypeParameterName = "LinearSolverType";
    private const string TrustRegionStrategyTypeParameterName = "TrustRegionStrategyType";
    private const string DogLegTypeParameterName = "DogLegType";
    private const string LineSearchDirectionTypeParameterName = "LineSearchDirectionType";

    private static readonly string[] MinimizerType = new[] { "LineSearch", "TrustRegion" };
    private static readonly string[] LinerSolverType = new[]
    {
      "DenseNormalCholesky",
      "DenseQR",
      "SparseNormalCholesky",
      "DenseSchur",
      "SparseSchur",
      "IterativeSchur",
      "ConjugateGradients"
    };
    private static readonly string[] TrustRegionStrategyType = new[]
    {
      "LevenbergMarquardt",
      "Dogleg"
    };
    private static readonly string[] DoglegType = new[]
    {
      "Traditional",
      "Subspace"
    };
    private static readonly string[] LinearSearchDirectionType = new[]
    {
      "SteepestDescent",
      "NonlinearConjugateGradient",
      "LBFGS",
      "BFGS"
    };

    #region parameters
    public IFixedValueParameter<IntValue> OptimizerIterationsParameter {
      get { return (IFixedValueParameter<IntValue>)Parameters[OptimizerIterationsParameterName]; }
    }
    public IFixedValueParameter<BoolValue> UseNonmonotonicStepsParameter {
      get { return (IFixedValueParameter<BoolValue>)Parameters[UseNonmonotonicStepsParameterName]; }
    }
    public IConstrainedValueParameter<StringValue> MinimizerTypeParameter {
      get { return (IConstrainedValueParameter<StringValue>)Parameters[MinimizerTypeParameterName]; }
    }
    public IConstrainedValueParameter<StringValue> LinearSolverTypeParameter {
      get { return (IConstrainedValueParameter<StringValue>)Parameters[LinearSolverTypeParameterName]; }
    }
    public IConstrainedValueParameter<StringValue> TrustRegionStrategyTypeParameter {
      get { return (IConstrainedValueParameter<StringValue>)Parameters[TrustRegionStrategyTypeParameterName]; }
    }
    public IConstrainedValueParameter<StringValue> DogLegTypeParameter {
      get { return (IConstrainedValueParameter<StringValue>)Parameters[DogLegTypeParameterName]; }
    }
    public IConstrainedValueParameter<StringValue> LineSearchDirectionTypeParameter {
      get { return (IConstrainedValueParameter<StringValue>)Parameters[LineSearchDirectionTypeParameterName]; }
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
    private CeresTypes.MinimizerType Minimizer {
      get { return (CeresTypes.MinimizerType)Enum.Parse(typeof(CeresTypes.MinimizerType), MinimizerTypeParameter.Value.Value); }
    }
    private CeresTypes.LinearSolverType LinearSolver {
      get { return (CeresTypes.LinearSolverType)Enum.Parse(typeof(CeresTypes.LinearSolverType), LinearSolverTypeParameter.Value.Value); }
    }
    private CeresTypes.TrustRegionStrategyType TrustRegionStrategy {
      get { return (CeresTypes.TrustRegionStrategyType)Enum.Parse(typeof(CeresTypes.TrustRegionStrategyType), TrustRegionStrategyTypeParameter.Value.Value); }
    }
    private CeresTypes.DoglegType Dogleg {
      get { return (CeresTypes.DoglegType)Enum.Parse(typeof(CeresTypes.DoglegType), DogLegTypeParameter.Value.Value); }
    }
    private CeresTypes.LineSearchDirectionType LineSearchDirection {
      get { return (CeresTypes.LineSearchDirectionType)Enum.Parse(typeof(CeresTypes.LineSearchDirectionType), LineSearchDirectionTypeParameter.Value.Value); }
    }
    #endregion

    private static IConstrainedValueParameter<StringValue> InitializeParameter(string name, string[] validValues, string value, bool hidden = true) {
      var parameter = new ConstrainedValueParameter<StringValue>(name, new ItemSet<StringValue>(validValues.Select(x => new StringValue(x))));
      parameter.Value = parameter.ValidValues.Single(x => x.Value == value);
      parameter.Hidden = hidden;
      return parameter;
    }

    [StorableConstructor]
    protected ParameterOptimizer(StorableConstructorFlag _) : base(_) { }

    public ParameterOptimizer() {
      var minimizerTypeParameter = InitializeParameter(MinimizerTypeParameterName, MinimizerType, "TrustRegion");
      var linearSolverTypeParameter = InitializeParameter(LinearSolverTypeParameterName, LinerSolverType, "DenseQR");
      var trustRegionStrategyTypeParameter = InitializeParameter(TrustRegionStrategyTypeParameterName, TrustRegionStrategyType, "LevenbergMarquardt");
      var dogLegTypeParameter = InitializeParameter(DogLegTypeParameterName, DoglegType, "Traditional");
      var lineSearchDirectionTypeParameter = InitializeParameter(LineSearchDirectionTypeParameterName, LinearSearchDirectionType, "SteepestDescent");

      Parameters.Add(new FixedValueParameter<IntValue>(OptimizerIterationsParameterName, "The number of iterations for the nonlinear least squares optimizer.", new IntValue(10)));
      Parameters.Add(new FixedValueParameter<BoolValue>(UseNonmonotonicStepsParameterName, "Allow the non linear least squares optimizer to make steps in parameter space that don't necessarily decrease the error, but might improve overall convergence.", new BoolValue(false)));
      Parameters.AddRange(new[] { minimizerTypeParameter, linearSolverTypeParameter, trustRegionStrategyTypeParameter, dogLegTypeParameter, lineSearchDirectionTypeParameter });
    }

    public ParameterOptimizer(ParameterOptimizer original, Cloner cloner) : base(original, cloner) { }

    public override IDeepCloneable Clone(Cloner cloner) {
      return new ParameterOptimizer(this, cloner);
    }

    private static byte MapSupportedSymbols(ISymbolicExpressionTreeNode node) {
      var opCode = OpCodes.MapSymbolToOpCode(node);
      if (supportedOpCodes.Contains(opCode)) return opCode;
      else throw new NotSupportedException($"The native interpreter does not support {node.Symbol.Name}");
    }

    public static Dictionary<ISymbolicExpressionTreeNode, double> OptimizeTree(ISymbolicExpressionTree tree, IDataset dataset, IEnumerable<int> rows, string targetVariable, HashSet<ISymbolicExpressionTreeNode> nodesToOptimize, SolverOptions options, ref OptimizationSummary summary) {
      var code = Compile(tree, dataset, MapSupportedSymbols, out List<ISymbolicExpressionTreeNode> nodes);

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

    public Dictionary<ISymbolicExpressionTreeNode, double> OptimizeTree(ISymbolicExpressionTree tree, IDataset dataset, IEnumerable<int> rows, string targetVariable, HashSet<ISymbolicExpressionTreeNode> nodesToOptimize = null) {
      var options = new SolverOptions {
        Iterations = OptimizerIterations,
        Minimizer = Minimizer,
        LinearSolver = LinearSolver,
        TrustRegionStrategy = TrustRegionStrategy,
        Dogleg = Dogleg,
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
        var code = Compile(terms[i], dataset, MapSupportedSymbols, out List<ISymbolicExpressionTreeNode> nodes);
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

      NativeWrapper.GetValuesVarPro(codeArray, termIndices,rowsArray, coeff, options, result, target, out summary);
      return Enumerable.Range(0, totalCodeSize).Where(i => codeArray[i].Optimize).ToDictionary(i => totalNodes[i], i => codeArray[i].Value);
    }
  }
}

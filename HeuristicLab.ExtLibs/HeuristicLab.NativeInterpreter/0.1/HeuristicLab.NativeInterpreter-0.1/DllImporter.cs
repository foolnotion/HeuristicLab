using System;
using System.Runtime.InteropServices;

namespace HeuristicLab.NativeInterpreter {
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
  public struct NativeInstruction {
    public byte OpCode;
    public ushort Arity;
    public ushort Length;
    public double Value; // weights for variables, values for parameters
    public IntPtr Data; // used for variables only
    public bool Optimize; // used for parameters only
  }

  public enum Algorithm : int {
    Krogh = 0,
    RuheWedin1 = 1,
    RuheWedin2 = 2,
    RuheWedin3 = 3
  }

  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
  public class SolverOptions {
    public int Iterations = 10;
    public int UseNonmonotonicSteps = 0; // = false
    public CeresTypes.Minimizer Minimizer = CeresTypes.Minimizer.TRUST_REGION;
    public CeresTypes.LinearSolver LinearSolver = CeresTypes.LinearSolver.DENSE_QR;
    public CeresTypes.TrustRegionStrategy TrustRegionStrategy = CeresTypes.TrustRegionStrategy.LEVENBERG_MARQUARDT;
    public CeresTypes.DogLeg DogLeg = CeresTypes.DogLeg.TRADITIONAL_DOGLEG;
    public CeresTypes.LineSearchDirection LineSearchDirection = CeresTypes.LineSearchDirection.LBFGS;
    public Algorithm Algorithm = Algorithm.Krogh;
  }

  // proxy structure to pass information from ceres back to the caller
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
  public class OptimizationSummary {
    public double InitialCost; // value of the objective function before the optimization
    public double FinalCost; // value of the objective function after the optimization
    public int SuccessfulSteps; // number of minimizer iterations in which the step was accepted
    public int UnsuccessfulSteps; // number of minimizer iterations in which the step was rejected
    public int InnerIterationSteps; // number of times inner iterations were performed
    public int ResidualEvaluations; // number of residual only evaluations
    public int JacobianEvaluations; // number of Jacobian (and residual) evaluations
  }

  public static class NativeWrapper {
    private const string x64dll = "hl-native-interpreter-x64.dll";
    private readonly static bool is64;

    static NativeWrapper() {
      is64 = Environment.Is64BitProcess;
    }

    public static void GetValues(NativeInstruction[] code, int[] rows, SolverOptions options, double[] result, double[] target, out OptimizationSummary optSummary) {
      optSummary = new OptimizationSummary();
      if (is64)
        GetValues64(code, code.Length, rows, rows.Length, options, result, target, optSummary);
      else
        throw new NotSupportedException("Native interpreter is only available on x64 builds");
    }
    public static void GetValuesVarPro(NativeInstruction[] code, int[] termIndices, int[] rows, double[] coefficients, SolverOptions options, double[] result, double[] target, out OptimizationSummary optSummary) {
      optSummary = new OptimizationSummary();
      if (is64)
        GetValuesVarPro64(code, code.Length, termIndices, termIndices.Length, rows, rows.Length, coefficients, options, result, target, optSummary);
      else
        throw new NotSupportedException("Native interpreter is only available on x64 builds");
    }

    [DllImport(x64dll, EntryPoint = "GetValues", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void GetValues64(
      [In,Out] NativeInstruction[] code, // parameters are optimized by callee
      [In] int len,
      [In] int[] rows,
      [In] int nRows,
      [In] SolverOptions options,
      [Out] double[] result,
      [In] double[] target,
      [Out] OptimizationSummary optSummary);

    [DllImport(x64dll, EntryPoint = "GetValuesVarPro", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void GetValuesVarPro64(
      [In,Out] NativeInstruction[] code,  // the values fields for non-linear parameters are changed by the callee
      int len,
      int[] termIndices,
      int nTerms,
      int[] rows,
      int nRows,
      [In,Out] double[] coefficients,
      [In] SolverOptions options,
      [In, Out] double[] result,
      [In] double[] target,
      [In,Out] OptimizationSummary optSummary);
  }
}

using System;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;

namespace HeuristicLab.Problems.DataAnalysis.Symbolic {
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
  public struct NativeInstruction {
    public byte OpCode;
    public ushort Arity;
    public ushort Length;
    public double Value;
    public IntPtr Data;
    public bool Optimize;
  }

  // proxy structure to pass information from Ceres back to the caller
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
  public struct OptimizationSummary {
    public double InitialCost; // value of the objective function before the optimization
    public double FinalCost; // value of the objective function after the optimization
    public int SuccessfulSteps; // number of minimizer iterations in which the step was accepted
    public int UnsuccessfulSteps; // number of minimizer iterations in which the step was rejected
    public int InnerIterationSteps; // number of times inner iterations were performed
    public int ResidualEvaluations; // number of residual only evaluations
    public int JacobianEvaluations; // number of Jacobian (and residual) evaluations
  };

  public enum MinimizerType : int {
    LINE_SEARCH = 0,
    TRUST_REGION = 1
  }

  public enum LinearSolverType : int {
    DENSE_NORMAL_CHOLESKY = 0,
    DENSE_QR = 1,
    SPARSE_NORMAL_CHOLESKY = 2,
    DENSE_SCHUR = 3,
    SPARSE_SCHUR = 4,
    ITERATIVE_SCHUR = 5,
    CGNR = 6
  }

  public enum TrustRegionStrategyType : int {
    LEVENBERG_MARQUARDT = 0,
    DOGLEG = 1
  }

  public enum DoglegType : int {
    TRADITIONAL_DOGLEG = 0,
    SUBSPACE_DOGLEG = 1
  }

  public enum LineSearchDirectionType : int {
    STEEPEST_DESCENT = 0,
    NONLINEAR_CONJUGATE_GRADIENT = 1,
    LBFGS = 2,
    BFSG = 3
  }

  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
  public struct SolverOptions {
    public int Iterations;
    public int UseNonmonotonicSteps;
    public int Minimizer; // type of minimizer (trust region or line search)
    public int LinearSolver; // type of linear solver
    public int TrustRegionStrategy; // levenberg-marquardt or dogleg
    public int Dogleg; // type of dogleg (traditional or subspace)
    public int LineSearchDirection; // line search direction type (BFGS, LBFGS, etc)
    public int Algorithm;
  };

  public static class NativeWrapper {
    private const string x86zip = "hl-native-interpreter-x86.zip";
    private const string x64zip = "hl-native-interpreter-x64.zip";

    private const string x86dll = "hl-native-interpreter-x86.dll";
    private const string x64dll = "hl-native-interpreter-x64.dll";

    private readonly static bool is64;

    static NativeWrapper() {
      is64 = Environment.Is64BitProcess;

      //using (var zip = new FileStream(is64 ? x64zip : x86zip, FileMode.Open)) {
      //  using (var archive = new ZipArchive(zip, ZipArchiveMode.Read)) {
      //    foreach (var entry in archive.Entries) {
      //      if (File.Exists(entry.Name)) {
      //        File.Delete(entry.Name);
      //      }
      //    }
      //    ZipFileExtensions.ExtractToDirectory(archive, Environment.CurrentDirectory);
      //  }
      //}
    }

    public static void GetValues(NativeInstruction[] code, int[] rows, double[] result, double[] target, SolverOptions options, ref OptimizationSummary summary) {
      if (is64)
        GetValues64(code, code.Length, rows, rows.Length, options, result, target, ref summary);
      else
        GetValues32(code, code.Length, rows, rows.Length, options, result, target, ref summary);
    }

    public static void GetValuesVarPro(NativeInstruction[] code, int[] termIndices, int[] rows, double[] coefficients,
      double[] result, double[] target, SolverOptions options, ref OptimizationSummary summary) {
      if (is64)
        GetValuesVarPro64(code, code.Length, termIndices, termIndices.Length, rows, rows.Length, coefficients, options, result, target, ref summary);
      else
        GetValuesVarPro32(code, code.Length, termIndices, termIndices.Length, rows, rows.Length, coefficients, options, result, target, ref summary);
    }

    // x86
    [DllImport(x86dll, EntryPoint = "GetValues", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void GetValues32([In, Out] NativeInstruction[] code, int len, int[] rows, int nRows, SolverOptions options, [In, Out] double[] result, double[] target, ref OptimizationSummary summary);

    // x64
    [DllImport(x64dll, EntryPoint = "GetValues", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void GetValues64([In, Out] NativeInstruction[] code, int len, int[] rows, int nRows, SolverOptions options, [In, Out] double[] result, double[] target, ref OptimizationSummary summary);

    [DllImport(x86dll, EntryPoint = "GetValuesVarPro", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void GetValuesVarPro32([In, Out] NativeInstruction[] code, int len, int[] termIndices, int nTerms, int[] rows, int nRows, [In, Out] double[] coefficients, SolverOptions options, [In, Out] double[] result, double[] target, ref OptimizationSummary summary);

    [DllImport(x64dll, EntryPoint = "GetValuesVarPro", CallingConvention = CallingConvention.Cdecl)]
    internal static extern void GetValuesVarPro64([In, Out] NativeInstruction[] code, int len, int[] termIndices, int nTerms, int[] rows, int nRows, [In, Out] double[] coefficients, SolverOptions options, [In, Out] double[] result, double[] target, ref OptimizationSummary summary);
  }
}

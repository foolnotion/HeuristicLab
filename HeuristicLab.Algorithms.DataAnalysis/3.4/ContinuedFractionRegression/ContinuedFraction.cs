using HeuristicLab.Core;

namespace HeuristicLab.Algorithms.DataAnalysis.ContinuedFractionRegression {
  public class ContinuedFraction {
    internal bool[] vars;
    internal Term[] h;

    public ContinuedFraction() { }
    public ContinuedFraction(int nVars, int depth, IRandom rand) {
      this.vars = new bool[nVars];
      for (int i = 0; i < nVars; i++) vars[i] = rand.NextDouble() < 0.3; // page 12 of the preprint. Each input variable then has a probability p = 1/3 to be present in the whitelist

      this.h = new Term[depth * 2 + 1];
      for (int i = 0; i < h.Length; i++) {
        h[i] = new Term();
        var hi = h[i];
        hi.vars = (bool[])vars.Clone();
        hi.coef = new double[nVars];
        for (int vi = 0; vi < nVars; vi++) {
          if (hi.vars[vi])
            hi.coef[vi] = rand.NextDouble() * 6 - 3;
        }
        hi.beta = rand.NextDouble() * 6 - 3;
      }
    }
  }
}
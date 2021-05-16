using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using HEAL.Attic;
using HeuristicLab.Common;
using HeuristicLab.Core;
using HeuristicLab.Data;
using HeuristicLab.Encodings.SymbolicExpressionTreeEncoding;
using HeuristicLab.Parameters;
using HeuristicLab.Problems.DataAnalysis;
using HeuristicLab.Problems.DataAnalysis.Symbolic;
using HeuristicLab.Problems.DataAnalysis.Symbolic.Regression;
using HeuristicLab.Random;

namespace HeuristicLab.Algorithms.DataAnalysis.ContinuedFractionRegression {
  [Item("Continuous Fraction Regression (CFR)", "TODO")]
  [Creatable(CreatableAttribute.Categories.DataAnalysisRegression, Priority = 999)]
  [StorableType("7A375270-EAAF-4AD1-82FF-132318D20E09")]
  public class Algorithm : FixedDataAnalysisAlgorithm<IRegressionProblem> {
    private const string MutationRateParameterName = "MutationRate";
    private const string DepthParameterName = "Depth";
    private const string NumGenerationsParameterName = "Depth";

    #region parameters
    public IFixedValueParameter<PercentValue> MutationRateParameter => (IFixedValueParameter<PercentValue>)Parameters[MutationRateParameterName];
    public double MutationRate {
      get { return MutationRateParameter.Value.Value; }
      set { MutationRateParameter.Value.Value = value; }
    }
    public IFixedValueParameter<IntValue> DepthParameter => (IFixedValueParameter<IntValue>)Parameters[DepthParameterName];
    public int Depth {
      get { return DepthParameter.Value.Value; }
      set { DepthParameter.Value.Value = value; }
    }
    public IFixedValueParameter<IntValue> NumGenerationsParameter => (IFixedValueParameter<IntValue>)Parameters[NumGenerationsParameterName];
    public int NumGenerations {
      get { return NumGenerationsParameter.Value.Value; }
      set { NumGenerationsParameter.Value.Value = value; }
    }
    #endregion

    // storable ctor
    [StorableConstructor]
    public Algorithm(StorableConstructorFlag _) : base(_) { }

    // cloning ctor
    public Algorithm(Algorithm original, Cloner cloner) : base(original, cloner) { }


    // default ctor
    public Algorithm() : base() {
      Parameters.Add(new FixedValueParameter<PercentValue>(MutationRateParameterName, "Mutation rate (default 10%)", new PercentValue(0.1)));
      Parameters.Add(new FixedValueParameter<IntValue>(DepthParameterName, "Depth of the continued fraction representation (default 6)", new IntValue(6)));
      Parameters.Add(new FixedValueParameter<IntValue>(NumGenerationsParameterName, "The maximum number of generations (default 200)", new IntValue(200)));
    }

    public override IDeepCloneable Clone(Cloner cloner) {
      throw new NotImplementedException();
    }

    protected override void Run(CancellationToken cancellationToken) {
      var problemData = Problem.ProblemData;

      var x = problemData.Dataset.ToArray(problemData.AllowedInputVariables.Concat(new[] { problemData.TargetVariable }),
        problemData.TrainingIndices);
      var nVars = x.GetLength(1) - 1;
      var seed = new System.Random().Next();
      var rand = new MersenneTwister((uint)seed);
      CFRAlgorithm(nVars, Depth, MutationRate, x, out var best, out var bestObj, rand, NumGenerations, stagnatingGens: 5, cancellationToken);
    }

    private void CFRAlgorithm(int nVars, int depth, double mutationRate, double[,] trainingData,
      out ContinuedFraction best, out double bestObj,
      IRandom rand, int numGen, int stagnatingGens,
      CancellationToken cancellationToken) {
      /* Algorithm 1 */
      /* Generate initial population by a randomized algorithm */
      var pop = InitialPopulation(nVars, depth, rand, trainingData);
      best = pop.pocket;
      bestObj = pop.pocketObjValue;
      var bestObjGen = 0;
      for (int gen = 1; gen <= numGen && !cancellationToken.IsCancellationRequested; gen++) {
        /* mutate each current solution in the population */
        var pop_mu = Mutate(pop, mutationRate, rand);
        /* generate new population by recombination mechanism */
        var pop_r = RecombinePopulation(pop_mu, rand, nVars);

        /* local search optimization of current solutions */
        foreach (var agent in pop_r.IterateLevels()) {
          LocalSearchSimplex(agent.current, ref agent.currentObjValue, trainingData, rand); // CHECK paper states that pocket might also be optimized. Unclear how / when invariants are maintained.
        }

        foreach (var agent in pop_r.IteratePostOrder()) agent.MaintainInvariant(); // CHECK deviates from Alg1 in paper 

        /* replace old population with evolved population */
        pop = pop_r;

        /* keep track of the best solution */
        if (bestObj > pop.pocketObjValue) { // CHECK: comparison obviously wrong in the paper
          best = pop.pocket;
          bestObj = pop.pocketObjValue;
          bestObjGen = gen;
          // Results.AddOrUpdateResult("MSE (best)", new DoubleValue(bestObj));
          // Results.AddOrUpdateResult("Solution", CreateSymbolicRegressionSolution(best, Problem.ProblemData));
        }


        if (gen > bestObjGen + stagnatingGens) {
          bestObjGen = gen; // CHECK: unspecified in the paper: wait at least stagnatingGens until resetting again
          Reset(pop, nVars, depth, rand, trainingData);
          // InitialPopulation(nVars, depth, rand, trainingData); CHECK reset is not specified in the paper
        }
      }
    }



    private Agent InitialPopulation(int nVars, int depth, IRandom rand, double[,] trainingData) {
      /* instantiate 13 agents in the population */
      var pop = new Agent();
      // see Figure 2
      for (int i = 0; i < 3; i++) {
        pop.children.Add(new Agent());
        for (int j = 0; j < 3; j++) {
          pop.children[i].children.Add(new Agent());
        }
      }

      foreach (var agent in pop.IteratePostOrder()) {
        agent.current = new ContinuedFraction(nVars, depth, rand);
        agent.pocket = new ContinuedFraction(nVars, depth, rand);

        agent.currentObjValue = Evaluate(agent.current, trainingData);
        agent.pocketObjValue = Evaluate(agent.pocket, trainingData);

        /* within each agent, the pocket solution always holds the better value of guiding
         * function than its current solution
         */
        agent.MaintainInvariant();
      }
      return pop;
    }

    // TODO: reset is not described in the paper
    private void Reset(Agent root, int nVars, int depth, IRandom rand, double[,] trainingData) {
      root.pocket = new ContinuedFraction(nVars, depth, rand);
      root.current = new ContinuedFraction(nVars, depth, rand);

      root.currentObjValue = Evaluate(root.current, trainingData);
      root.pocketObjValue = Evaluate(root.pocket, trainingData);

      /* within each agent, the pocket solution always holds the better value of guiding
       * function than its current solution
       */
      root.MaintainInvariant();
    }



    private Agent RecombinePopulation(Agent pop, IRandom rand, int nVars) {
      var l = pop;

      if (pop.children.Count > 0) {
        var s1 = pop.children[0];
        var s2 = pop.children[1];
        var s3 = pop.children[2];

        // CHECK Deviates from paper (recombine all models in the current pop before updating the population)
        var l_current = Recombine(l.pocket, s1.current, SelectRandomOp(rand), rand, nVars);
        var s3_current = Recombine(s3.pocket, l.current, SelectRandomOp(rand), rand, nVars);
        var s1_current = Recombine(s1.pocket, s2.current, SelectRandomOp(rand), rand, nVars);
        var s2_current = Recombine(s2.pocket, s3.current, SelectRandomOp(rand), rand, nVars);

        // recombination works from top to bottom
        // CHECK do we use the new current solutions (s1_current .. s3_current) already in the next levels?
        foreach (var child in pop.children) {
          RecombinePopulation(child, rand, nVars);
        }

        l.current = l_current;
        s3.current = s3_current;
        s1.current = s1_current;
        s2.current = s2_current;
      }
      return pop;
    }

    private Func<bool[], bool[], bool[]> SelectRandomOp(IRandom rand) {
      bool[] union(bool[] a, bool[] b) {
        var res = new bool[a.Length];
        for (int i = 0; i < a.Length; i++) res[i] = a[i] || b[i];
        return res;
      }
      bool[] intersect(bool[] a, bool[] b) {
        var res = new bool[a.Length];
        for (int i = 0; i < a.Length; i++) res[i] = a[i] && b[i];
        return res;
      }
      bool[] symmetricDifference(bool[] a, bool[] b) {
        var res = new bool[a.Length];
        for (int i = 0; i < a.Length; i++) res[i] = a[i] ^ b[i];
        return res;
      }
      switch (rand.Next(3)) {
        case 0: return union;
        case 1: return intersect;
        case 2: return symmetricDifference;
        default: throw new ArgumentException();
      }
    }

    private static ContinuedFraction Recombine(ContinuedFraction p1, ContinuedFraction p2, Func<bool[], bool[], bool[]> op, IRandom rand, int nVars) {
      ContinuedFraction ch = new ContinuedFraction() { h = new Term[p1.h.Length] };
      /* apply a recombination operator chosen uniformly at random on variables of two parents into offspring */
      ch.vars = op(p1.vars, p2.vars);

      /* recombine the coefficients for each term (h) of the continued fraction */
      for (int i = 0; i < p1.h.Length; i++) {
        var coefa = p1.h[i].coef; var varsa = p1.h[i].vars;
        var coefb = p2.h[i].coef; var varsb = p2.h[i].vars;

        /* recombine coefficient values for variables */
        var coefx = new double[nVars];
        var varsx = new bool[nVars]; // CHECK: deviates from paper, probably forgotten in the pseudo-code
        for (int vi = 1; vi < nVars; vi++) {
          if (ch.vars[vi]) {  // CHECK: paper uses featAt()
            if (varsa[vi] && varsb[vi]) {
              coefx[vi] = coefa[vi] + (rand.NextDouble() * 5 - 1) * (coefb[vi] - coefa[vi]) / 3.0;
              varsx[vi] = true;
            } else if (varsa[vi]) {
              coefx[vi] = coefa[vi];
              varsx[vi] = true;
            } else if (varsb[vi]) {
              coefx[vi] = coefb[vi];
              varsx[vi] = true;
            }
          }
        }
        /* update new coefficients of the term in offspring */
        ch.h[i] = new Term() { coef = coefx, vars = varsx };
        /* compute new value of constant (beta) for term hi in the offspring solution ch using 
         * beta of p1.hi and p2.hi */
        ch.h[i].beta = p1.h[i].beta + (rand.NextDouble() * 5 - 1) * (p2.h[i].beta - p1.h[i].beta) / 3.0;
      }
      /* update current solution and apply local search */
      // return LocalSearchSimplex(ch, trainingData); // CHECK: Deviates from paper because Alg1 also has LocalSearch after Recombination
      return ch;
    }

    private Agent Mutate(Agent pop, double mutationRate, IRandom rand) {
      foreach (var agent in pop.IterateLevels()) {
        if (rand.NextDouble() < mutationRate) {
          if (agent.currentObjValue < 1.2 * agent.pocketObjValue ||
             agent.currentObjValue > 2 * agent.pocketObjValue)
            ToggleVariables(agent.current, rand); // major mutation
          else
            ModifyVariable(agent.current, rand); // soft mutation
        }
      }
      return pop;
    }

    private void ToggleVariables(ContinuedFraction cfrac, IRandom rand) {
      double coinToss(double a, double b) {
        return rand.NextDouble() < 0.5 ? a : b;
      }

      /* select a variable index uniformly at random */
      int N = cfrac.vars.Length;
      var vIdx = rand.Next(N);

      /* for each depth of continued fraction, toggle the selection of variables of the term (h) */
      foreach (var h in cfrac.h) {
        /* Case 1: cfrac variable is turned ON: Turn OFF the variable, and either 'Remove' or 
         * 'Remember' the coefficient value at random */
        if (cfrac.vars[vIdx]) {  // CHECK: paper uses varAt()
          h.vars[vIdx] = false;  // CHECK: paper uses varAt()
          h.coef[vIdx] = coinToss(0, h.coef[vIdx]);
        } else {
          /* Case 2: term variable is turned OFF: Turn ON the variable, and either 'Remove' 
           * or 'Replace' the coefficient with a random value between -3 and 3 at random */
          if (!h.vars[vIdx]) {
            h.vars[vIdx] = true;  // CHECK: paper uses varAt()
            h.coef[vIdx] = coinToss(0, rand.NextDouble() * 6 - 3);
          }
        }
      }
      /* toggle the randomly selected variable */
      cfrac.vars[vIdx] = !cfrac.vars[vIdx];  // CHECK: paper uses varAt()
    }

    private void ModifyVariable(ContinuedFraction cfrac, IRandom rand) {
      /* randomly select a variable which is turned ON */
      var candVars = new List<int>();
      for (int i = 0; i < cfrac.vars.Length; i++) if (cfrac.vars[i]) candVars.Add(i);  // CHECK: paper uses varAt()
      if (candVars.Count == 0) return; // no variable active
      var vIdx = candVars[rand.Next(candVars.Count)];

      /* randomly select a term (h) of continued fraction */
      var h = cfrac.h[rand.Next(cfrac.h.Length)];

      /* modify the coefficient value*/
      if (h.vars[vIdx]) {  // CHECK: paper uses varAt()
        h.coef[vIdx] = 0.0;
      } else {
        h.coef[vIdx] = rand.NextDouble() * 6 - 3;
      }
      /* Toggle the randomly selected variable */
      h.vars[vIdx] = !h.vars[vIdx]; // CHECK: paper uses varAt()
    }

    private static double Evaluate(ContinuedFraction cfrac, double[,] trainingData) {
      var dataPoint = new double[trainingData.GetLength(1) - 1];
      var yIdx = trainingData.GetLength(1) - 1;
      double sum = 0.0;
      for (int r = 0; r < trainingData.GetLength(0); r++) {
        for (int c = 0; c < dataPoint.Length; c++) {
          dataPoint[c] = trainingData[r, c];
        }
        var y = trainingData[r, yIdx];
        var pred = Evaluate(cfrac, dataPoint);
        var res = y - pred;
        sum += res * res;
      }
      var delta = 0.1;
      return sum / trainingData.GetLength(0) * (1 + delta * cfrac.vars.Count(vi => vi));
    }

    private static double Evaluate(ContinuedFraction cfrac, double[] dataPoint) {
      var res = 0.0;
      for (int i = cfrac.h.Length - 1; i > 1; i -= 2) {
        var hi = cfrac.h[i];
        var hi1 = cfrac.h[i - 1];
        var denom = hi.beta + dot(hi.vars, hi.coef, dataPoint) + res;
        var numerator = hi1.beta + dot(hi1.vars, hi1.coef, dataPoint);
        res = numerator / denom;
      }
      var h0 = cfrac.h[0];
      res += h0.beta + dot(h0.vars, h0.coef, dataPoint);
      return res;
    }

    private static double dot(bool[] filter, double[] x, double[] y) {
      var s = 0.0;
      for (int i = 0; i < x.Length; i++)
        if (filter[i])
          s += x[i] * y[i];
      return s;
    }


    private static void LocalSearchSimplex(ContinuedFraction ch, ref double quality, double[,] trainingData, IRandom rand) {
      double uniformPeturbation = 1.0;
      double tolerance = 1e-3;
      int maxEvals = 250;
      int numSearches = 4;
      var numRows = trainingData.GetLength(0);
      int numSelectedRows = numRows / 5; // 20% of the training samples

      quality = Evaluate(ch, trainingData); // get quality with origial coefficients

      double[] origCoeff = ExtractCoeff(ch);
      if (origCoeff.Length == 0) return; // no parameters to optimize

      var bestQuality = quality;
      var bestCoeff = origCoeff;

      var fittingData = SelectRandomRows(trainingData, numSelectedRows, rand);

      double objFunc(double[] curCoeff) {
        SetCoeff(ch, curCoeff);
        return Evaluate(ch, fittingData);
      }

      for (int count = 0; count < numSearches; count++) {

        SimplexConstant[] constants = new SimplexConstant[origCoeff.Length];
        for (int i = 0; i < origCoeff.Length; i++) {
          constants[i] = new SimplexConstant(origCoeff[i], uniformPeturbation);
        }

        RegressionResult result = NelderMeadSimplex.Regress(constants, tolerance, maxEvals, objFunc);

        var optimizedCoeff = result.Constants;
        SetCoeff(ch, optimizedCoeff);

        var newQuality = Evaluate(ch, trainingData);

        if (newQuality < bestQuality) {
          bestCoeff = optimizedCoeff;
          bestQuality = newQuality;
        }
      } // reps

      SetCoeff(ch, bestCoeff);
      quality = bestQuality;
    }

    private static double[,] SelectRandomRows(double[,] trainingData, int numSelectedRows, IRandom rand) {
      var numRows = trainingData.GetLength(0);
      var numCols = trainingData.GetLength(1);
      var selectedRows = Enumerable.Range(0, numRows).Shuffle(rand).Take(numSelectedRows).ToArray();
      var subset = new double[numSelectedRows, numCols];
      var i = 0;
      foreach (var r in selectedRows) {
        for (int c = 0; c < numCols; c++) {
          subset[i, c] = trainingData[r, c];
        }
        i++;
      }
      return subset;
    }

    private static double[] ExtractCoeff(ContinuedFraction ch) {
      var coeff = new List<double>();
      foreach (var hi in ch.h) {
        coeff.Add(hi.beta);
        for (int vIdx = 0; vIdx < hi.vars.Length; vIdx++) {
          if (hi.vars[vIdx]) coeff.Add(hi.coef[vIdx]);
        }
      }
      return coeff.ToArray();
    }

    private static void SetCoeff(ContinuedFraction ch, double[] curCoeff) {
      int k = 0;
      foreach (var hi in ch.h) {
        hi.beta = curCoeff[k++];
        for (int vIdx = 0; vIdx < hi.vars.Length; vIdx++) {
          if (hi.vars[vIdx]) hi.coef[vIdx] = curCoeff[k++];
        }
      }
    }

    Symbol addSy = new Addition();
    Symbol mulSy = new Multiplication();
    Symbol divSy = new Division();
    Symbol startSy = new StartSymbol();
    Symbol progSy = new ProgramRootSymbol();
    Symbol constSy = new Constant();
    Symbol varSy = new Problems.DataAnalysis.Symbolic.Variable();

    private ISymbolicRegressionSolution CreateSymbolicRegressionSolution(ContinuedFraction cfrac, IRegressionProblemData problemData) {
      var variables = problemData.AllowedInputVariables.ToArray();
      ISymbolicExpressionTreeNode res = null;
      for (int i = cfrac.h.Length - 1; i > 1; i -= 2) {
        var hi = cfrac.h[i];
        var hi1 = cfrac.h[i - 1];
        var denom = CreateLinearCombination(hi.vars, hi.coef, variables, hi.beta);
        if (res != null) {
          denom.AddSubtree(res);
        }

        var numerator = CreateLinearCombination(hi1.vars, hi1.coef, variables, hi1.beta);

        res = divSy.CreateTreeNode();
        res.AddSubtree(numerator);
        res.AddSubtree(denom);
      }

      var h0 = cfrac.h[0];
      var h0Term = CreateLinearCombination(h0.vars, h0.coef, variables, h0.beta);
      h0Term.AddSubtree(res);

      var progRoot = progSy.CreateTreeNode();
      var start = startSy.CreateTreeNode();
      progRoot.AddSubtree(start);
      start.AddSubtree(h0Term);

      var model = new SymbolicRegressionModel(problemData.TargetVariable, new SymbolicExpressionTree(progRoot), new SymbolicDataAnalysisExpressionTreeBatchInterpreter());
      var sol = new SymbolicRegressionSolution(model, (IRegressionProblemData)problemData.Clone());
      return sol;
    }

    private ISymbolicExpressionTreeNode CreateLinearCombination(bool[] vars, double[] coef, string[] variables, double beta) {
      var sum = addSy.CreateTreeNode();
      for (int i = 0; i < vars.Length; i++) {
        if (vars[i]) {
          var varNode = (VariableTreeNode)varSy.CreateTreeNode();
          varNode.Weight = coef[i];
          varNode.VariableName = variables[i];
          sum.AddSubtree(varNode);
        }
      }
      sum.AddSubtree(CreateConstant(beta));
      return sum;
    }

    private ISymbolicExpressionTreeNode CreateConstant(double value) {
      var constNode = (ConstantTreeNode)constSy.CreateTreeNode();
      constNode.Value = value;
      return constNode;
    }
  }

  public class Agent {
    public ContinuedFraction pocket;
    public double pocketObjValue;
    public ContinuedFraction current;
    public double currentObjValue;

    public IList<Agent> children = new List<Agent>();

    public IEnumerable<Agent> IterateLevels() {
      var agents = new List<Agent>() { this };
      IterateLevelsRec(this, agents);
      return agents;
    }
    public IEnumerable<Agent> IteratePostOrder() {
      var agents = new List<Agent>();
      IteratePostOrderRec(this, agents);
      return agents;
    }

    internal void MaintainInvariant() {
      foreach (var child in children) {
        MaintainInvariant(parent: this, child);
      }
      if (currentObjValue < pocketObjValue) {
        Swap(ref pocket, ref current);
        Swap(ref pocketObjValue, ref currentObjValue);
      }
    }


    private static void MaintainInvariant(Agent parent, Agent child) {
      if (child.pocketObjValue < parent.pocketObjValue) {
        Swap(ref child.pocket, ref parent.pocket);
        Swap(ref child.pocketObjValue, ref parent.pocketObjValue);
      }
    }

    private void IterateLevelsRec(Agent agent, List<Agent> agents) {
      foreach (var child in agent.children) {
        agents.Add(child);
      }
      foreach (var child in agent.children) {
        IterateLevelsRec(child, agents);
      }
    }

    private void IteratePostOrderRec(Agent agent, List<Agent> agents) {
      foreach (var child in agent.children) {
        IteratePostOrderRec(child, agents);
      }
      agents.Add(agent);
    }


    private static void Swap<T>(ref T a, ref T b) {
      var temp = a;
      a = b;
      b = temp;
    }
  }
}

﻿#region License Information
/* HeuristicLab
 * Copyright (C) Heuristic and Evolutionary Algorithms Laboratory (HEAL)
 * and the BEACON Center for the Study of Evolution in Action.
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
using System.Threading;
using HEAL.Attic;
using HeuristicLab.Common;
using HeuristicLab.Core;
using HeuristicLab.Encodings.BinaryVectorEncoding;
using HeuristicLab.Optimization;

namespace HeuristicLab.Algorithms.ParameterlessPopulationPyramid {
  // This code is based off the publication
  // B. W. Goldman and W. F. Punch, "Parameter-less Population Pyramid," GECCO, pp. 785–792, 2014
  // and the original source code in C++11 available from: https://github.com/brianwgoldman/Parameter-less_Population_Pyramid
  [StorableType("D5F1358D-C100-40CF-9BA5-E95F72F64D1A")]
  internal sealed class EvaluationTracker : Item, ISingleObjectiveProblemDefinition<BinaryVectorEncoding, BinaryVector> {
    [Storable]
    private SingleObjectiveProblem<BinaryVectorEncoding, BinaryVector> problem;
    [Storable]
    private int maxEvaluations;

    #region Properties
    [Storable]
    public double BestQuality {
      get;
      private set;
    }
    [Storable]
    public int Evaluations {
      get;
      private set;
    }
    [Storable]
    public int BestFoundOnEvaluation {
      get;
      private set;
    }
    [Storable]
    public BinaryVector BestSolution {
      get;
      private set;
    }

    public BinaryVectorEncoding Encoding {
      get { return problem.Encoding; }
    }
    #endregion


    [StorableConstructor]
    private EvaluationTracker(StorableConstructorFlag _) : base(_) { }

    private EvaluationTracker(EvaluationTracker original, Cloner cloner)
      : base(original, cloner) {
      problem = cloner.Clone(original.problem);
      maxEvaluations = original.maxEvaluations;
      BestQuality = original.BestQuality;
      Evaluations = original.Evaluations;
      BestFoundOnEvaluation = original.BestFoundOnEvaluation;
      BestSolution = cloner.Clone(original.BestSolution);
    }
    public override IDeepCloneable Clone(Cloner cloner) {
      return new EvaluationTracker(this, cloner);
    }

    public EvaluationTracker(SingleObjectiveProblem<BinaryVectorEncoding, BinaryVector> problem, int maxEvaluations) {
      this.problem = problem;
      this.maxEvaluations = maxEvaluations;
      BestSolution = new BinaryVector(problem.Encoding.Length);
      BestQuality = double.NaN;
      Evaluations = 0;
      BestFoundOnEvaluation = 0;
    }

    public double Evaluate(BinaryVector vector, IRandom random) {
      return Evaluate(vector, random, CancellationToken.None);
    }

    public double Evaluate(BinaryVector vector, IRandom random, CancellationToken cancellationToken) {
      if (Evaluations >= maxEvaluations) throw new OperationCanceledException("Maximum Evaluation Limit Reached");
      Evaluations++;
      double fitness = problem.Evaluate(vector, random);
      if (double.IsNaN(BestQuality) || problem.IsBetter(fitness, BestQuality)) {
        BestQuality = fitness;
        BestSolution = (BinaryVector)vector.Clone();
        BestFoundOnEvaluation = Evaluations;
      }
      return fitness;
    }

    public bool Maximization {
      get {
        if (problem == null) return false;
        return problem.Maximization;
      }
    }

    public bool IsBetter(double quality, double bestQuality) {
      return problem.IsBetter(quality, bestQuality);
    }

    public void Analyze(BinaryVector[] individuals, double[] qualities, ResultCollection results, IRandom random) {
      problem.Analyze(individuals, qualities, results, random);
    }

    public IEnumerable<BinaryVector> GetNeighbors(BinaryVector individual, IRandom random) {
      return problem.GetNeighbors(individual, random);
    }
  }
}

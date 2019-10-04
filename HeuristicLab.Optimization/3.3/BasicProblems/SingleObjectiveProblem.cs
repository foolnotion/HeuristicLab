﻿#region License Information
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
using System.Threading;
using HEAL.Attic;
using HeuristicLab.Common;
using HeuristicLab.Core;
using HeuristicLab.Data;
using HeuristicLab.Parameters;

namespace HeuristicLab.Optimization {
  [StorableType("2697320D-0259-44BB-BD71-7EE1B10F664C")]
  public abstract class SingleObjectiveProblem<TEncoding, TEncodedSolution> :
    Problem<TEncoding, TEncodedSolution, SingleObjectiveEvaluator<TEncodedSolution>>,
    ISingleObjectiveProblem<TEncoding, TEncodedSolution>,
    ISingleObjectiveProblemDefinition<TEncoding, TEncodedSolution>
    where TEncoding : class, IEncoding<TEncodedSolution>
    where TEncodedSolution : class, IEncodedSolution {

    [Storable] protected IValueParameter<DoubleValue> BestKnownQualityParameter { get; }
    [Storable] protected IValueParameter<BoolValue> MaximizationParameter { get; }

    public double BestKnownQuality {
      get {
        if (BestKnownQualityParameter.Value == null) return double.NaN;
        return BestKnownQualityParameter.Value.Value;
      }
      set {
        if (double.IsNaN(value)) {
          BestKnownQualityParameter.Value = null;
          return;
        }
        if (BestKnownQualityParameter.Value == null) BestKnownQualityParameter.Value = new DoubleValue(value);
        else BestKnownQualityParameter.Value.Value = value;
      }
    }

    public bool Maximization {
      get { return MaximizationParameter.Value.Value; }
      protected set {
        if (Maximization == value) return;
        MaximizationParameter.ForceValue(new BoolValue(value, @readonly: true));
        OnMaximizationChanged();
      }
    }

    [StorableConstructor]
    protected SingleObjectiveProblem(StorableConstructorFlag _) : base(_) { }

    protected SingleObjectiveProblem(SingleObjectiveProblem<TEncoding, TEncodedSolution> original, Cloner cloner)
      : base(original, cloner) {
      BestKnownQualityParameter = cloner.Clone(original.BestKnownQualityParameter);
      MaximizationParameter = cloner.Clone(original.MaximizationParameter);
      ParameterizeOperators();
    }

    protected SingleObjectiveProblem() : base() {

      MaximizationParameter = new ValueParameter<BoolValue>("Maximization", "Whether the problem should be maximized (True) or minimized (False).", new BoolValue(false).AsReadOnly()) { Hidden = true, ReadOnly = true };
      BestKnownQualityParameter = new OptionalValueParameter<DoubleValue>("BestKnownQuality", "The quality of the best known solution of this problem.");

      Parameters.Add(MaximizationParameter);
      Parameters.Add(BestKnownQualityParameter);

      Operators.Add(Evaluator);
      Operators.Add(new SingleObjectiveAnalyzer<TEncodedSolution>());
      Operators.Add(new SingleObjectiveImprover<TEncodedSolution>());
      Operators.Add(new SingleObjectiveMoveEvaluator<TEncodedSolution>());
      Operators.Add(new SingleObjectiveMoveGenerator<TEncodedSolution>());
      Operators.Add(new SingleObjectiveMoveMaker<TEncodedSolution>());

      ParameterizeOperators();
    }

    protected SingleObjectiveProblem(TEncoding encoding) : base(encoding) {
      Parameters.Add(MaximizationParameter = new ValueParameter<BoolValue>("Maximization", "Set to false if the problem should be minimized.", new BoolValue(false).AsReadOnly()) { Hidden = true, ReadOnly = true });
      Parameters.Add(BestKnownQualityParameter = new OptionalValueParameter<DoubleValue>("BestKnownQuality", "The quality of the best known solution of this problem."));

      Operators.Add(Evaluator);
      Operators.Add(new SingleObjectiveAnalyzer<TEncodedSolution>());
      Operators.Add(new SingleObjectiveImprover<TEncodedSolution>());
      Operators.Add(new SingleObjectiveMoveEvaluator<TEncodedSolution>());
      Operators.Add(new SingleObjectiveMoveGenerator<TEncodedSolution>());
      Operators.Add(new SingleObjectiveMoveMaker<TEncodedSolution>());

      ParameterizeOperators();
    }

    [StorableHook(HookType.AfterDeserialization)]
    private void AfterDeserialization() {
      ParameterizeOperators();
    }

    public virtual double Evaluate(TEncodedSolution solution, IRandom random) {
      return Evaluate(solution, random, CancellationToken.None);
    }
    public abstract double Evaluate(TEncodedSolution solution, IRandom random, CancellationToken cancellationToken);
    public virtual void Analyze(TEncodedSolution[] solutions, double[] qualities, ResultCollection results, IRandom random) { }
    public virtual IEnumerable<TEncodedSolution> GetNeighbors(TEncodedSolution solution, IRandom random) {
      return Enumerable.Empty<TEncodedSolution>();
    }

    public static bool IsBetter(bool maximization, double quality, double bestQuality) {
      return (maximization && quality > bestQuality || !maximization && quality < bestQuality);
    }

    public virtual bool IsBetter(double quality, double bestQuality) {
      return IsBetter(Maximization, quality, bestQuality);
    }

    protected Tuple<TEncodedSolution, double> GetBestSolution(TEncodedSolution[] solutions, double[] qualities) {
      return GetBestSolution(solutions, qualities, Maximization);
    }
    public static Tuple<TEncodedSolution, double> GetBestSolution(TEncodedSolution[] solutions, double[] qualities, bool maximization) {
      var zipped = solutions.Zip(qualities, (s, q) => new { Solution = s, Quality = q });
      var best = (maximization ? zipped.OrderByDescending(z => z.Quality) : zipped.OrderBy(z => z.Quality)).First();
      return Tuple.Create(best.Solution, best.Quality);
    }

    protected override void OnOperatorsChanged() {
      if (Encoding != null) {
        PruneMultiObjectiveOperators(Encoding);
        var combinedEncoding = Encoding as CombinedEncoding;
        if (combinedEncoding != null) {
          foreach (var encoding in combinedEncoding.Encodings.ToList()) {
            PruneMultiObjectiveOperators(encoding);
          }
        }
      }
      base.OnOperatorsChanged();
    }

    private void PruneMultiObjectiveOperators(IEncoding encoding) {
      if (encoding.Operators.Any(x => x is IMultiObjectiveOperator && !(x is ISingleObjectiveOperator)))
        encoding.Operators = encoding.Operators.Where(x => !(x is IMultiObjectiveOperator) || x is ISingleObjectiveOperator).ToList();

      foreach (var multiOp in Encoding.Operators.OfType<IMultiOperator>()) {
        foreach (var moOp in multiOp.Operators.Where(x => x is IMultiObjectiveOperator).ToList()) {
          multiOp.RemoveOperator(moOp);
        }
      }
    }

    protected override void OnEvaluatorChanged() {
      base.OnEvaluatorChanged();
      ParameterizeOperators();
    }

    private void ParameterizeOperators() {
      foreach (var op in Operators.OfType<ISingleObjectiveEvaluationOperator<TEncodedSolution>>())
        op.EvaluateFunc = Evaluate;
      foreach (var op in Operators.OfType<ISingleObjectiveAnalysisOperator<TEncodedSolution>>())
        op.AnalyzeAction = Analyze;
      foreach (var op in Operators.OfType<INeighborBasedOperator<TEncodedSolution>>())
        op.GetNeighborsFunc = GetNeighbors;
    }

    #region ISingleObjectiveHeuristicOptimizationProblem Members
    IParameter ISingleObjectiveHeuristicOptimizationProblem.MaximizationParameter {
      get { return Parameters["Maximization"]; }
    }
    IParameter ISingleObjectiveHeuristicOptimizationProblem.BestKnownQualityParameter {
      get { return Parameters["BestKnownQuality"]; }
    }
    ISingleObjectiveEvaluator ISingleObjectiveHeuristicOptimizationProblem.Evaluator {
      get { return Evaluator; }
    }
    #endregion

    public event EventHandler MaximizationChanged;
    protected void OnMaximizationChanged() {
      MaximizationChanged?.Invoke(this, EventArgs.Empty);
    }
  }
}

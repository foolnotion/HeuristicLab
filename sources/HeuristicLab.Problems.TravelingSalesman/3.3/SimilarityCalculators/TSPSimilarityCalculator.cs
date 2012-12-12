﻿#region License Information
/* HeuristicLab
 * Copyright (C) 2002-2012 Heuristic and Evolutionary Algorithms Laboratory (HEAL)
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
using HeuristicLab.Common;
using HeuristicLab.Core;
using HeuristicLab.Encodings.PermutationEncoding;
using HeuristicLab.Optimization.Operators;

namespace HeuristicLab.Problems.TravelingSalesman {
  /// <summary>
  /// An operator that performs similarity calculation between two traveling salesman solutions.
  /// </summary>
  /// <remarks>
  /// The operator calculates the similarity based on the number of edges the two solutions have in common.
  /// </remarks>
  [Item("TSPSimilarityCalculator", "An operator that performs similarity calculation between two traveling salesman solutions. The operator calculates the similarity based on the number of edges the two solutions have in common.")]
  public sealed class TSPSimilarityCalculator : SingleObjectiveSolutionSimilarityCalculator {
    private TSPSimilarityCalculator(bool deserializing) : base(deserializing) { }
    private TSPSimilarityCalculator(TSPSimilarityCalculator original, Cloner cloner) : base(original, cloner) { }
    public TSPSimilarityCalculator() : base() { }

    public override IDeepCloneable Clone(Cloner cloner) {
      return new TSPSimilarityCalculator(this, cloner);
    }

    public static double CalculateSimilarity(Permutation left, Permutation right) {
      if (left == null || right == null)
        throw new ArgumentException("Cannot calculate similarity because one of the provided solutions or both are null.");
      if (left.PermutationType != right.PermutationType)
        throw new ArgumentException("Cannot calculate similarity because the provided solutions have different types.");
      if (left.Length != right.Length)
        throw new ArgumentException("Cannot calculate similarity because the provided solutions have different lengths.");
      if (object.ReferenceEquals(left, right)) return 1.0;

      switch (left.PermutationType) {
        case PermutationTypes.Absolute:
          return CalculateAbsolute(left, right);
        case PermutationTypes.RelativeDirected:
          return CalculateRelativeDirected(left, right);
        case PermutationTypes.RelativeUndirected:
          return CalculateRelativeUndirected(left, right);
        default:
          throw new InvalidOperationException("unknown permutation type");
      }
    }

    private static double CalculateAbsolute(Permutation left, Permutation right) {
      double similarity = 0.0;
      for (int i = 0; i < left.Length; i++)
        if (left[i] == right[i]) similarity++;

      return similarity / left.Length;
    }

    private static double CalculateRelativeDirected(Permutation left, Permutation right) {
      int[] edges = new int[right.Length];
      for (int i = 0; i < right.Length; i++)
        edges[right[i]] = right[(i + 1) % right.Length];

      double similarity = 0.0;
      for (int i = 0; i < left.Length; i++)
        if (left[(i + 1) % left.Length] == edges[left[i]]) similarity++;

      return similarity / left.Length;
    }

    private static double CalculateRelativeUndirected(Permutation left, Permutation right) {
      int[,] edges = new int[right.Length, 2];
      for (int i = 0; i < right.Length; i++) {
        edges[right[i], 0] = right[(i + 1) % right.Length];
        edges[right[i], 1] = right[(i - 1 + right.Length) % right.Length];
      }

      double similarity = 0.0;
      for (int i = 0; i < left.Length; i++) {
        int targetCity = left[(i + 1) % left.Length];
        if (targetCity == edges[left[i], 0] || targetCity == edges[left[i], 1])
          similarity++;
      }

      return similarity / left.Length;
    }

    public override double CalculateSolutionSimilarity(IScope leftSolution, IScope rightSolution) {
      var sol1 = leftSolution.Variables[SolutionVariableName].Value as Permutation;
      var sol2 = rightSolution.Variables[SolutionVariableName].Value as Permutation;

      return CalculateSimilarity(sol1, sol2);
    }
  }
}

﻿#region License Information
/* HeuristicLab
 * Copyright (C) 2002-2019 Heuristic and Evolutionary Algorithms Laboratory (HEAL)
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

using HeuristicLab.Core;
using HEAL.Fossil;

namespace HeuristicLab.Optimization {
  [StorableType("39eacdb5-80a0-425d-902a-00eb3e1d6610")]
  public interface IMultiObjectiveProblemDefinition : IProblemDefinition {
    bool[] Maximization { get; }
    double[] Evaluate(Individual individual, IRandom random);
    void Analyze(Individual[] individuals, double[][] qualities, ResultCollection results, IRandom random);
  }
}

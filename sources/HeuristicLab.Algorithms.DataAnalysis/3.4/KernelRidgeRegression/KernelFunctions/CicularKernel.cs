﻿#region License Information
/* HeuristicLab
 * Copyright (C) 2002-2016 Heuristic and Evolutionary Algorithms Laboratory (HEAL)
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
using HeuristicLab.Persistence.Default.CompositeSerializers.Storable;

namespace HeuristicLab.Algorithms.DataAnalysis.KernelRidgeRegression {
  [StorableClass]
  [Item("CircularKernel", "A circular kernel function 2*pi*(acos(-d)-d*(1-d²)^(0.5)) where n = ||x-c|| and d = n/beta \n  As described in http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/")]
  public class CircularKernel : KernelBase {

    #region HLConstructors & Boilerplate
    [StorableConstructor]
    protected CircularKernel(bool deserializing) : base(deserializing) { }
    [StorableHook(HookType.AfterDeserialization)]
    private void AfterDeserialization() { }
    protected CircularKernel(CircularKernel original, Cloner cloner) : base(original, cloner) { }
    public CircularKernel() {
    }
    public override IDeepCloneable Clone(Cloner cloner) {
      return new CircularKernel(this, cloner);
    }
    #endregion

    protected override double Get(double norm) {
      var beta = Beta.Value;
      if (Math.Abs(beta) < double.Epsilon) return double.NaN;
      if (norm >= beta) return 0;
      var d = norm / beta;
      return 2 * Math.PI * (Math.Acos(-d) - d * Math.Sqrt(1 - d * d));
    }

    // 4*pi*n^3 / (beta^4 * sqrt(1-n^2/beta^2) 
    protected override double GetGradient(double norm) {
      var beta = Beta.Value;
      if (Math.Abs(beta) < double.Epsilon) return double.NaN;
      if (beta < norm) return 0;
      var d = norm / beta;
      return -4 * Math.PI * d * d * d / beta * Math.Sqrt(1 - d * d);
    }
  }
}

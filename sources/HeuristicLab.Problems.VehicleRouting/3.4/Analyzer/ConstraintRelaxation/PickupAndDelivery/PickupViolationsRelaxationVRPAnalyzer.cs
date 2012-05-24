﻿#region License Information
/* HeuristicLab
 * Copyright (C) 2002-2010 Heuristic and Evolutionary Algorithms Laboratory (HEAL)
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
using System.Text;
using HeuristicLab.Core;
using HeuristicLab.Persistence.Default.CompositeSerializers.Storable;
using HeuristicLab.Operators;
using HeuristicLab.Optimization;
using HeuristicLab.Problems.VehicleRouting.Variants;
using HeuristicLab.Problems.VehicleRouting.Interfaces;
using HeuristicLab.Parameters;
using HeuristicLab.Data;
using HeuristicLab.Common;

namespace HeuristicLab.Problems.VehicleRouting {
  /// <summary>
  /// An operator for adaptive constraint relaxation.
  /// </summary>
  [Item("PickupViolationsRelaxationVRPAnalyzer", "An operator for adaptively relaxing the pickup constraints.")]
  [StorableClass]
  public class PickupViolationsRelaxationVRPAnalyzer: SingleSuccessorOperator, IAnalyzer, IPickupAndDeliveryOperator {
    public ILookupParameter<IVRPProblemInstance> ProblemInstanceParameter {
      get { return (ILookupParameter<IVRPProblemInstance>)Parameters["ProblemInstance"]; }
    }
    public ScopeTreeLookupParameter<IVRPEncoding> VRPToursParameter {
      get { return (ScopeTreeLookupParameter<IVRPEncoding>)Parameters["VRPTours"]; }
    }
    public ScopeTreeLookupParameter<DoubleValue> QualityParameter {
      get { return (ScopeTreeLookupParameter<DoubleValue>)Parameters["Quality"]; }
    }

    public ScopeTreeLookupParameter<IntValue> PickupViolationsParameter {
      get { return (ScopeTreeLookupParameter<IntValue>)Parameters["PickupViolations"]; }
    }

    public IValueParameter<DoubleValue> SigmaParameter {
      get { return (IValueParameter<DoubleValue>)Parameters["Sigma"]; }
    }
    public IValueParameter<DoubleValue> PhiParameter {
      get { return (IValueParameter<DoubleValue>)Parameters["Phi"]; }
    }
    public IValueParameter<DoubleValue> MinPenaltyFactorParameter {
      get { return (IValueParameter<DoubleValue>)Parameters["MinPenaltyFactor"]; }
    }

    public ValueLookupParameter<ResultCollection> ResultsParameter {
      get { return (ValueLookupParameter<ResultCollection>)Parameters["Results"]; }
    }

    public bool EnabledByDefault {
      get { return false; }
    }

    [StorableConstructor]
    private PickupViolationsRelaxationVRPAnalyzer(bool deserializing) : base(deserializing) { }

    public PickupViolationsRelaxationVRPAnalyzer()
      : base() {
      Parameters.Add(new LookupParameter<IVRPProblemInstance>("ProblemInstance", "The problem instance."));
      Parameters.Add(new ScopeTreeLookupParameter<IVRPEncoding>("VRPTours", "The VRP tours which should be evaluated."));
      Parameters.Add(new ScopeTreeLookupParameter<DoubleValue>("Quality", "The qualities of the VRP solutions which should be analyzed."));

      Parameters.Add(new ScopeTreeLookupParameter<IntValue>("PickupViolations", "The pickup violation of the VRP solutions which should be analyzed."));

      Parameters.Add(new ValueParameter<DoubleValue>("Sigma", "The sigma applied to the penalty factor.", new DoubleValue(0.04)));
      Parameters.Add(new ValueParameter<DoubleValue>("Phi", "The phi applied to the penalty factor.", new DoubleValue(0.01)));
      Parameters.Add(new ValueParameter<DoubleValue>("MinPenaltyFactor", "The minimum penalty factor.", new DoubleValue(0.01)));

      Parameters.Add(new ValueLookupParameter<ResultCollection>("Results", "The result collection where the best VRP solution should be stored."));
    }

    public override IDeepCloneable Clone(Cloner cloner) {
      return new PickupViolationsRelaxationVRPAnalyzer(this, cloner);
    }

    private PickupViolationsRelaxationVRPAnalyzer(PickupViolationsRelaxationVRPAnalyzer original, Cloner cloner)
      : base(original, cloner) {
    }

    public override IOperation Apply() {
      IPickupAndDeliveryProblemInstance pdp = ProblemInstanceParameter.ActualValue as IPickupAndDeliveryProblemInstance;
      ResultCollection results = ResultsParameter.ActualValue;
      
      ItemArray<DoubleValue> qualities = QualityParameter.ActualValue;
      ItemArray<IntValue> pickupViolations = PickupViolationsParameter.ActualValue;
      
      double sigma = SigmaParameter.Value.Value;
      double phi = PhiParameter.Value.Value;
      double minPenalty = MinPenaltyFactorParameter.Value.Value;

      for (int j = 0; j < qualities.Length; j++) {
        qualities[j].Value -= pickupViolations[j].Value * pdp.PickupViolationPenalty.Value;
      }

      int validCount = 0;
      for (int j = 0; j < qualities.Length; j++) {
        if (pickupViolations[j].Value == 0)
          validCount++;
      }

      double factor = 1.0 - ((double)validCount / (double)qualities.Length);

      double min = pdp.PickupViolationPenalty.Value / (1 + sigma);
      double max = pdp.PickupViolationPenalty.Value * (1 + phi);

      pdp.PickupViolationPenalty = new DoubleValue(min + (max - min) * factor);
      if (pdp.PickupViolationPenalty.Value < minPenalty)
        pdp.PickupViolationPenalty.Value = minPenalty;

      for (int j = 0; j < qualities.Length; j++) {
        qualities[j].Value += pickupViolations[j].Value * pdp.PickupViolationPenalty.Value;
      }

      if (!results.ContainsKey("Current Pickup Violation Penalty")) {
        results.Add(new Result("Current Pickup Violation Penalty", new DoubleValue(pdp.PickupViolationPenalty.Value)));
      } else {
        (results["Current Pickup Violation Penalty"].Value as DoubleValue).Value = pdp.PickupViolationPenalty.Value;
      }

      return base.Apply();
    }
  }
}

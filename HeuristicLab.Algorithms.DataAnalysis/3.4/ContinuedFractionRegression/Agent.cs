using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace HeuristicLab.Algorithms.DataAnalysis.ContinuedFractionRegression {
  public class Agent {
    public ContinuedFraction pocket;
    public double pocketObjValue;
    public ContinuedFraction current;
    public double currentObjValue;
    public IList<Agent> children = new List<Agent>();

    public void MaintainPocketCurrentInvariant() {
      if (currentObjValue < pocketObjValue) {
        Swap(ref pocket, ref current);
        Swap(ref pocketObjValue, ref currentObjValue);
      }
    }

    public void MaintainInvariant() {
      foreach (var child in children) {
        MaintainParentChildInvariant(parent: this, child);
      }
      MaintainPocketCurrentInvariant();
    }


    private static void MaintainParentChildInvariant(Agent parent, Agent child) {
      if (child.pocketObjValue < parent.pocketObjValue) {
        Swap(ref child.pocket, ref parent.pocket);
        Swap(ref child.pocketObjValue, ref parent.pocketObjValue);
      }
    }

    public IEnumerable<Agent> IterateLevels() {
      var agents = new List<Agent>() { this };
      int i = 0;
      while (i < agents.Count) {
        var p = agents[i];
        foreach (var child in p.children)
          agents.Add(child);
        i++;
      }
      return agents;
    }
    public IEnumerable<Agent> IteratePostOrder() {
      var agents = new List<Agent>();
      IteratePostOrderRec(this, agents);
      return agents;
    }
    public IEnumerable<Agent> IteratePreOrder() {
      var agents = new List<Agent>();
      IteratePreOrderRec(this, agents);
      return agents;
    }

    private void IteratePostOrderRec(Agent agent, List<Agent> agents) {
      foreach (var child in agent.children) {
        IteratePostOrderRec(child, agents);
      }
      agents.Add(agent);
    }

    private void IteratePreOrderRec(Agent agent, List<Agent> agents) {
      agents.Add(agent);
      foreach (var child in agent.children) {
        IteratePreOrderRec(child, agents);
      }
    }


    private static void Swap<T>(ref T a, ref T b) {
      var temp = a;
      a = b;
      b = temp;
    }

    internal void AssertInvariant() {
      Debug.Assert(pocketObjValue <= currentObjValue);
      foreach (var ch in children) {
        Debug.Assert(pocketObjValue <= ch.pocketObjValue);
      }
    }
  }
}

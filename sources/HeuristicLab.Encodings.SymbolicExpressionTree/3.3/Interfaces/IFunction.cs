#region License Information
/* HeuristicLab
 * Copyright (C) 2002-2008 Heuristic and Evolutionary Algorithms Laboratory (HEAL)
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
using System.Text;
using HeuristicLab.Core;

namespace HeuristicLab.Encodings.SymbolicExpressionTree {
  public interface IFunction : IItem {
    string Name { get; }
    IFunctionTree GetTreeNode();
    ICollection<IFunction> GetAllowedSubFunctions(int index);
    void AddAllowedSubFunction(IFunction f, int index);
    void RemoveAllowedSubFunction(IFunction f, int index);
    bool IsAllowedSubFunction(IFunction f, int index);
    int MinSubTrees { get; }
    int MaxSubTrees { get; }
    int MinTreeHeight { get; }
    int MinTreeSize { get; }
    double Tickets { get; }
    IOperator Initializer { get; }
    IOperator Manipulator { get; }
  }
}

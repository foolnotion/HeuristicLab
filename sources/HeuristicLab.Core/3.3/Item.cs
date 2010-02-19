#region License Information
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

using System.Drawing;
using HeuristicLab.Common.Resources;
using HeuristicLab.Persistence.Default.CompositeSerializers.Storable;

namespace HeuristicLab.Core {
  /// <summary>
  /// Represents the base class for all basic item types.
  /// </summary>
  [EmptyStorableClass]
  [Item("Item", "Base class for all HeuristicLab items.")]
  public abstract class Item : DeepCloneable, IItem {
    public virtual string ItemName {
      get { return ItemAttribute.GetName(this.GetType()); }
    }
    public virtual string ItemDescription {
      get { return ItemAttribute.GetDescription(this.GetType()); }
    }
    public virtual Image ItemImage {
      get { return VS2008ImageLibrary.Class; }
    }

    /// <summary>
    /// Gets the string representation of the current instance.
    /// </summary>
    /// <returns>The type name of the current instance.</returns>
    public override string ToString() {
      return ItemName;
    }

    public event ChangedEventHandler Changed;
    protected void OnChanged() {
      OnChanged(new ChangedEventArgs());
    }
    protected virtual void OnChanged(ChangedEventArgs e) {
      if ((e.RegisterChangedObject(this)) && (Changed != null))
        Changed(this, e);
    }
  }
}

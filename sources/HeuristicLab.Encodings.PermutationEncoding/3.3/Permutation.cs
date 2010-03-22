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

using HeuristicLab.Core;
using HeuristicLab.Data;
using HeuristicLab.Persistence.Default.CompositeSerializers.Storable;

namespace HeuristicLab.Encodings.PermutationEncoding {
  [StorableClass]
  [Item("Permutation", "Represents a permutation of integer values.")]
  public class Permutation : IntArray {
    public Permutation() : base() { }
    public Permutation(int length)
      : base(length) {
      for (int i = 0; i < length; i++)
        this[i] = i;
    }
    public Permutation(int length, IRandom random)
      : this(length) {
      Randomize(random);
    }
    public Permutation(int[] elements) : base(elements) { }
    public Permutation(IntArray elements)
      : this(elements.Length) {
      for (int i = 0; i < array.Length; i++)
        array[i] = elements[i];
    }

    public override IDeepCloneable Clone(Cloner cloner) {
      Permutation clone = new Permutation(array);
      cloner.RegisterClonedObject(this, clone);
      return clone;
    }

    public virtual bool Validate() {
      bool[] values = new bool[Length];
      int value;

      for (int i = 0; i < values.Length; i++)
        values[i] = false;
      for (int i = 0; i < Length; i++) {
        value = this[i];
        if ((value < 0) || (value >= values.Length)) return false;
        if (values[value]) return false;
        values[value] = true;
      }
      return true;
    }

    public virtual void Randomize(IRandom random, int startIndex, int length) {
      if (length > 1) {
        // Knuth shuffle
        int index1, index2;
        int val;
        for (int i = length - 1; i > 0; i--) {
          index1 = startIndex + i;
          index2 = startIndex + random.Next(i + 1);
          if (index1 != index2) {
            val = array[index1];
            array[index1] = array[index2];
            array[index2] = val;
          }
        }
        OnReset();
      }
    }
    public void Randomize(IRandom random) {
      Randomize(random, 0, Length);
    }

    public virtual int GetCircular(int position) {
      if (position >= Length) position = position % Length;
      while (position < 0) position += Length;
      return this[position];
    }
  }
}

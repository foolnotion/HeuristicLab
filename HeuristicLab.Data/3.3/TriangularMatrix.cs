#region License Information
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
using HeuristicLab.Common;
using HeuristicLab.Core;
using HEAL.Attic;
using System.Collections.ObjectModel;

namespace HeuristicLab.Data {
  [Item("TriangularMatrix", "Represents a lower triangular matrix.")]
  [StorableType("5C09A4FC-887E-4C40-8926-81325C09FA67")]
  public class TriangularMatrix<T> : ValueTypeArray<T>, IStringConvertibleArray where T : struct {
    [Storable]
    private readonly int dimension;
    public int Dimension { get { return dimension; } }

    public ReadOnlyCollection<T> Data {
      get { return Array.AsReadOnly(array); }
    }

    private TriangularMatrix() { }

    public TriangularMatrix(int dimension) : base(dimension * (dimension + 1) / 2) {
      this.dimension = dimension;
      resizable = false;
    }

    [StorableConstructor]
    protected TriangularMatrix(StorableConstructorFlag _) : base(_) { }

    protected TriangularMatrix(TriangularMatrix<T> original, Cloner cloner) : base(original, cloner) {
      dimension = original.dimension;
      array = (T[])original.array.Clone();
    }

    public override IDeepCloneable Clone(Cloner cloner) {
      return new TriangularMatrix<T>(this, cloner);
    }

    // the indexing rule for the (lower-)triangular matrix is that always i <= j, otherwise an IndexOutOfBounds exception will occur 
    public T this[int rowIndex, int columnIndex] {
      get {
        // provide symmetry of returned values 
        if (columnIndex > rowIndex) return this[columnIndex, rowIndex];
        return array[rowIndex * (rowIndex + 1) / 2 + columnIndex];
      }
      set {
        if (columnIndex > rowIndex) this[columnIndex, rowIndex] = value;
        else array[rowIndex * (rowIndex + 1) / 2 + columnIndex] = value;
      }
    }

    protected virtual string GetValue(int rowIndex, int columnIndex) {
      return this[rowIndex, columnIndex].ToString(); // see above indexing rule
    }

    protected virtual bool SetValue(string value, int rowIndex, int columnIndex) {
      T val;
      if (!TryParse(value, out val))
        return false;
      this[rowIndex, columnIndex] = val;
      return true;
    }

    protected virtual bool Validate(string value, out string errorMessage) {
      T val;
      errorMessage = "";
      if (!TryParse(value, out val)) {
        errorMessage = string.Format("Could not parse string \"{0}\" as {1}.", value, typeof(T));
        return false;
      }
      return true;
    }

    public override IEnumerator<T> GetEnumerator() {
      return array.Cast<T>().GetEnumerator();
    }

    private static bool TryParse(string value, out T val) {
      if (typeof(T) == typeof(sbyte)) {
        sbyte v;
        if (sbyte.TryParse(value, out v)) {
          val = (T)(object)v;
          return true;
        }
      } else if (typeof(T) == typeof(byte)) {
        byte v;
        if (byte.TryParse(value, out v)) {
          val = (T)(object)v;
          return true;
        }
      } else if (typeof(T) == typeof(char)) {
        char v;
        if (char.TryParse(value, out v)) {
          val = (T)(object)v;
          return true;
        }
      } else if (typeof(T) == typeof(short)) {
        short v;
        if (short.TryParse(value, out v)) {
          val = (T)(object)v;
          return true;
        }
      } else if (typeof(T) == typeof(ushort)) {
        ushort v;
        if (ushort.TryParse(value, out v)) {
          val = (T)(object)v;
          return true;
        }
      } else if (typeof(T) == typeof(int)) {
        int v;
        if (int.TryParse(value, out v)) {
          val = (T)(object)v;
          return true;
        }
      } else if (typeof(T) == typeof(uint)) {
        uint v;
        if (uint.TryParse(value, out v)) {
          val = (T)(object)v;
          return true;
        }
      } else if (typeof(T) == typeof(long)) {
        long v;
        if (long.TryParse(value, out v)) {
          val = (T)(object)v;
          return true;
        }
      } else if (typeof(T) == typeof(ulong)) {
        ulong v;
        if (ulong.TryParse(value, out v)) {
          val = (T)(object)v;
          return true;
        }
      }
      val = default(T);
      return false;
    }

    public void GetMatrixCoordinates(int index, out int row, out int col) {
      var root = TriangularRoot(index);
      row = (int)Math.Floor(root);
      col = index - row * (row + 1) / 2;
    }

    private static double TriangularRoot(double x) {
      return (Math.Sqrt(8 * x + 1) - 1) / 2;
    }

    #region IStringConvertibleArray members
    bool IStringConvertibleArray.Validate(string value, out string errorMessage) {
      return Validate(value, out errorMessage);
    }

    public string GetValue(int index) {
      return array[index].ToString();
    }

    public bool SetValue(string value, int index) {
      if (TryParse(value, out T val)) {
        this[index] = val;
        return true;
      }
      throw new ArgumentException("Coult not parse value " + value + " as " + typeof(T));
    }
    #endregion
  }
}

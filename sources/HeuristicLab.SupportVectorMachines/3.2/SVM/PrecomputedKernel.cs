﻿/*
 * SVM.NET Library
 * Copyright (C) 2008 Matthew Johnson
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */


using System;
using System.Collections.Generic;

namespace SVM
{
    /// <remarks>
    /// Class encapsulating a precomputed kernel, where each position indicates the similarity score for two items in the training data.
    /// </remarks>
    [Serializable]
    public class PrecomputedKernel
    {
        private float[,] _similarities;
        private int _rows;
        private int _columns;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="similarities">The similarity scores between all items in the training data</param>
        public PrecomputedKernel(float[,] similarities)
        {
            _similarities = similarities;
            _rows = _similarities.GetLength(0);
            _columns = _similarities.GetLength(1);
        }

        /// <summary>
        /// Constructs a <see cref="Problem"/> object using the labels provided.  If a label is set to "0" that item is ignored.
        /// </summary>
        /// <param name="rowLabels">The labels for the row items</param>
        /// <param name="columnLabels">The labels for the column items</param>
        /// <returns>A <see cref="Problem"/> object</returns>
        public Problem Compute(double[] rowLabels, double[] columnLabels)
        {
            List<Node[]> X = new List<Node[]>();
            List<double> Y = new List<double>();
            int maxIndex = 0;
            for (int i = 0; i < columnLabels.Length; i++)
                if (columnLabels[i] != 0)
                    maxIndex++;
            maxIndex++;
            for (int r = 0; r < _rows; r++)
            {
                if (rowLabels[r] == 0)
                    continue;
                List<Node> nodes = new List<Node>();
                nodes.Add(new Node(0, X.Count + 1));
                for (int c = 0; c < _columns; c++)
                {
                    if (columnLabels[c] == 0)
                        continue;
                    double value = _similarities[r, c];
                    nodes.Add(new Node(nodes.Count, value));
                }
                X.Add(nodes.ToArray());
                Y.Add(rowLabels[r]);
            }
            return new Problem(X.Count, Y.ToArray(), X.ToArray(), maxIndex);
        }

    }
}

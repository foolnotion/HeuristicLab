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
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace HeuristicLab.AdvancedOptimizationFrontend {
  /// <summary>
  /// The about dialog that presents some basic information about the application.
  /// </summary>
  public partial class AboutDialog : Form {
    /// <summary>
    /// Initializes a new instance of <see cref="AboutDialog"/>.
    /// </summary>
    public AboutDialog() {
      InitializeComponent();
      textBox1.Text = @"HeuristicLab
v." + Application.ProductVersion + @"

(c) 2002-2008 HEAL";
    }

    private void okButton_Click(object sender, EventArgs e) {
      this.Close();
    }
  }
}

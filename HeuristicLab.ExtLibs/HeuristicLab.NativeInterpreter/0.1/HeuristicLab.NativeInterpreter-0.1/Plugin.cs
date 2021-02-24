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
using System.IO;

using HeuristicLab.PluginInfrastructure;

namespace HeuristicLab.Problems.DataAnalysis.Symbolic {
  [Plugin("HeuristicLab.NativeInterpreter", "Provides a native (C++) interpreter for symbolic expression trees", "0.0.0.1")]
  [PluginFile("HeuristicLab.Problems.DataAnalysis.Symbolic.NativeInterpreter-0.1.dll", PluginFileType.Assembly)]
  public class HeuristicLabNativeInterpreterPlugin : PluginBase {
    public override void OnLoad() {
      base.OnLoad();
      // add path for native dlls to PATH env variable

      var is64 = Environment.Is64BitProcess;
      string nativeDllPath = Path.Combine(Environment.CurrentDirectory, is64 ? "x64" : "x86");
      var envPath = Environment.GetEnvironmentVariable("PATH");
      if (!envPath.Contains(nativeDllPath))
        Environment.SetEnvironmentVariable("PATH", envPath + ";" + nativeDllPath);
    }
  }
}

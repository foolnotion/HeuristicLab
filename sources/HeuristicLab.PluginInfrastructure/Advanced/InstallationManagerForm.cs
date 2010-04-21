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
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.IO;
using HeuristicLab.PluginInfrastructure.Manager;

namespace HeuristicLab.PluginInfrastructure.Advanced {
  internal partial class InstallationManagerForm : Form {
    private class UpdateOrInstallPluginsBackgroundWorkerArgument {
      public IEnumerable<IPluginDescription> PluginsToUpdate { get; set; }
      public IEnumerable<IPluginDescription> PluginsToInstall { get; set; }
    }

    private class RemovePluginsBackgroundWorkerArgument {
      public IEnumerable<IPluginDescription> PluginsToRemove { get; set; }
    }

    private class RefreshBackgroundWorkerResult {
      public IEnumerable<IPluginDescription> RemotePlugins { get; set; }
      public IEnumerable<DeploymentService.ProductDescription> RemoteProducts { get; set; }
    }

    private InstallationManager installationManager;
    private BackgroundWorker refreshServerPluginsBackgroundWorker;
    private BackgroundWorker updateOrInstallPluginsBackgroundWorker;
    private BackgroundWorker removePluginsBackgroundWorker;
    private BackgroundWorker refreshLocalPluginsBackgroundWorker;
    private PluginManager pluginManager;
    private string pluginDir;

    public InstallationManagerForm(PluginManager pluginManager) {
      InitializeComponent();
      this.pluginManager = pluginManager;
      pluginManager.PluginLoaded += pluginManager_PluginLoaded;
      pluginManager.PluginUnloaded += pluginManager_PluginUnloaded;
      pluginManager.Initializing += pluginManager_Initializing;
      pluginManager.Initialized += pluginManager_Initialized;

      pluginDir = Application.StartupPath;

      #region initialize background workers
      refreshServerPluginsBackgroundWorker = new BackgroundWorker();
      refreshServerPluginsBackgroundWorker.DoWork += new DoWorkEventHandler(refreshServerPluginsBackgroundWorker_DoWork);
      refreshServerPluginsBackgroundWorker.RunWorkerCompleted += new RunWorkerCompletedEventHandler(refreshServerPluginsBackgroundWorker_RunWorkerCompleted);

      updateOrInstallPluginsBackgroundWorker = new BackgroundWorker();
      updateOrInstallPluginsBackgroundWorker.DoWork += new DoWorkEventHandler(updateOrInstallPluginsBackgroundWorker_DoWork);
      updateOrInstallPluginsBackgroundWorker.RunWorkerCompleted += new RunWorkerCompletedEventHandler(updateOrInstallPluginsBackgroundWorker_RunWorkerCompleted);

      removePluginsBackgroundWorker = new BackgroundWorker();
      removePluginsBackgroundWorker.DoWork += new DoWorkEventHandler(removePluginsBackgroundWorker_DoWork);
      removePluginsBackgroundWorker.RunWorkerCompleted += new RunWorkerCompletedEventHandler(removePluginsBackgroundWorker_RunWorkerCompleted);

      refreshLocalPluginsBackgroundWorker = new BackgroundWorker();
      refreshLocalPluginsBackgroundWorker.DoWork += new DoWorkEventHandler(refreshLocalPluginsBackgroundWorker_DoWork);
      refreshLocalPluginsBackgroundWorker.RunWorkerCompleted += new RunWorkerCompletedEventHandler(refreshLocalPluginsBackgroundWorker_RunWorkerCompleted);
      #endregion

      installationManager = new InstallationManager(pluginDir);
      installationManager.PluginInstalled += new EventHandler<PluginInfrastructureEventArgs>(installationManager_PluginInstalled);
      installationManager.PluginRemoved += new EventHandler<PluginInfrastructureEventArgs>(installationManager_PluginRemoved);
      installationManager.PluginUpdated += new EventHandler<PluginInfrastructureEventArgs>(installationManager_PluginUpdated);
      installationManager.PreInstallPlugin += new EventHandler<PluginInfrastructureCancelEventArgs>(installationManager_PreInstallPlugin);
      installationManager.PreRemovePlugin += new EventHandler<PluginInfrastructureCancelEventArgs>(installationManager_PreRemovePlugin);
      installationManager.PreUpdatePlugin += new EventHandler<PluginInfrastructureCancelEventArgs>(installationManager_PreUpdatePlugin);

      RefreshLocalPluginListAsync();
    }

    #region event handlers for refresh local plugin list backgroundworker
    void refreshLocalPluginsBackgroundWorker_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e) {
      if (!e.Cancelled && e.Error == null) {
        UpdateLocalPluginList((IEnumerable<PluginDescription>)e.Result);
        UpdateControlsConnected();
      }
    }

    void refreshLocalPluginsBackgroundWorker_DoWork(object sender, DoWorkEventArgs e) {
      pluginManager.DiscoverAndCheckPlugins();
      e.Result = new List<PluginDescription>(pluginManager.Plugins);
    }
    #endregion

    #region event handlers for plugin removal background worker
    void removePluginsBackgroundWorker_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e) {
      if (!e.Cancelled && e.Error == null) {
        RefreshLocalPluginListAsync();
        UpdateControlsConnected();
      }
    }

    void removePluginsBackgroundWorker_DoWork(object sender, DoWorkEventArgs e) {
      IEnumerable<IPluginDescription> pluginsToRemove = (IEnumerable<IPluginDescription>)e.Argument;
      installationManager.Remove(pluginsToRemove);
    }
    #endregion

    #region event handlers for plugin update background worker
    void updateOrInstallPluginsBackgroundWorker_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e) {
      if (!e.Cancelled && e.Error == null) {
        RefreshLocalPluginListAsync();
        RefreshRemotePluginListAsync();
        UpdateControlsConnected();
      } else {
        UpdateControlsDisconnected();
      }
    }

    void updateOrInstallPluginsBackgroundWorker_DoWork(object sender, DoWorkEventArgs e) {
      UpdateOrInstallPluginsBackgroundWorkerArgument info = (UpdateOrInstallPluginsBackgroundWorkerArgument)e.Argument;
      installationManager.Install(info.PluginsToInstall);
      installationManager.Update(info.PluginsToUpdate);
    }
    #endregion

    #region event handlers for refresh server plugins background worker
    void refreshServerPluginsBackgroundWorker_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e) {
      if (!e.Cancelled && e.Result != null) {
        RefreshBackgroundWorkerResult refreshResult = (RefreshBackgroundWorkerResult)e.Result;
        UpdateRemotePluginList(refreshResult.RemoteProducts, refreshResult.RemotePlugins);
        UpdateControlsConnected();
      } else {
        UpdateControlsDisconnected();
      }
    }

    void refreshServerPluginsBackgroundWorker_DoWork(object sender, DoWorkEventArgs e) {
      RefreshBackgroundWorkerResult result = new RefreshBackgroundWorkerResult();
      result.RemotePlugins = installationManager.GetRemotePluginList();
      result.RemoteProducts = installationManager.GetRemoteProductList();
      e.Cancel = false;
      e.Result = result;
    }



    #endregion

    #region plugin manager event handlers
    void pluginManager_Initialized(object sender, PluginInfrastructureEventArgs e) {
      SetStatusStrip("Initialized PluginInfrastructure");
    }

    void pluginManager_Initializing(object sender, PluginInfrastructureEventArgs e) {
      SetStatusStrip("Initializing PluginInfrastructure");
    }

    void pluginManager_PluginUnloaded(object sender, PluginInfrastructureEventArgs e) {
      SetStatusStrip("Unloaded " + e.Entity);
    }

    void pluginManager_PluginLoaded(object sender, PluginInfrastructureEventArgs e) {
      SetStatusStrip("Loaded " + e.Entity);
    }
    #endregion

    #region installation manager event handlers
    void installationManager_PreUpdatePlugin(object sender, PluginInfrastructureCancelEventArgs e) {
      if (e.Plugins.Count() > 0) {
        e.Cancel = ConfirmUpdateAction(e.Plugins) == false;
      }
    }

    void installationManager_PreRemovePlugin(object sender, PluginInfrastructureCancelEventArgs e) {
      if (e.Plugins.Count() > 0) {
        e.Cancel = ConfirmRemoveAction(e.Plugins) == false;
      }
    }

    void installationManager_PreInstallPlugin(object sender, PluginInfrastructureCancelEventArgs e) {
      if (e.Plugins.Count() > 0)
        if (ConfirmInstallAction(e.Plugins) == true) {
          SetStatusStrip("Installing " + e.Plugins.Aggregate("", (a, b) => a.ToString() + "; " + b.ToString()));
          e.Cancel = false;
        } else {
          e.Cancel = true;
          SetStatusStrip("Install canceled");
        }
    }

    void installationManager_PluginUpdated(object sender, PluginInfrastructureEventArgs e) {
      SetStatusStrip("Updated " + e.Entity);
    }

    void installationManager_PluginRemoved(object sender, PluginInfrastructureEventArgs e) {
      SetStatusStrip("Removed " + e.Entity);
    }

    void installationManager_PluginInstalled(object sender, PluginInfrastructureEventArgs e) {
      SetStatusStrip("Installed " + e.Entity);
    }
    #endregion

    private void SetStatusStrip(string msg) {
      if (InvokeRequired) Invoke((Action<string>)SetStatusStrip, msg);
      else {
        toolStripStatusLabel.Text = msg;
        logTextBox.Text += DateTime.Now + ": " + msg + Environment.NewLine;
      }
    }

    #region button events

    private void refreshButton_Click(object sender, EventArgs e) {
      RefreshRemotePluginListAsync();
    }

    private void updateButton_Click(object sender, EventArgs e) {
      Cursor = Cursors.AppStarting;
      toolStripProgressBar.Visible = true;
      DisableControls();
      var updateOrInstallInfo = new UpdateOrInstallPluginsBackgroundWorkerArgument();
      // if there is a local plugin with same name and same major and minor version then it's an update
      var pluginsToUpdate = from remotePlugin in remotePluginInstaller.CheckedPlugins
                            let matchingLocalPlugins = from localPlugin in localPluginManagerView.Plugins
                                                       where localPlugin.Name == remotePlugin.Name
                                                       where localPlugin.Version.Major == remotePlugin.Version.Major
                                                       where localPlugin.Version.Minor == remotePlugin.Version.Minor
                                                       select localPlugin
                            where matchingLocalPlugins.Count() > 0
                            select remotePlugin;

      // otherwise install a new plugin
      var pluginsToInstall = remotePluginInstaller.CheckedPlugins.Except(pluginsToUpdate);

      updateOrInstallInfo.PluginsToInstall = pluginsToInstall;
      updateOrInstallInfo.PluginsToUpdate = pluginsToUpdate;
      updateOrInstallPluginsBackgroundWorker.RunWorkerAsync(updateOrInstallInfo);
    }

    private void removeButton_Click(object sender, EventArgs e) {
      Cursor = Cursors.AppStarting;
      toolStripProgressBar.Visible = true;
      DisableControls();
      removePluginsBackgroundWorker.RunWorkerAsync(localPluginManagerView.CheckedPlugins);
    }

    #endregion

    #region confirmation dialogs
    private bool ConfirmRemoveAction(IEnumerable<IPluginDescription> plugins) {
      StringBuilder strBuilder = new StringBuilder();
      foreach (var plugin in plugins) {
        foreach (var file in plugin.Files) {
          strBuilder.AppendLine(Path.GetFileName(file.Name));
        }
      }
      return (new ConfirmationDialog("Confirm Delete", "Do you want to delete following files?", strBuilder.ToString())).ShowDialog() == DialogResult.OK;
    }

    private bool ConfirmUpdateAction(IEnumerable<IPluginDescription> plugins) {
      StringBuilder strBuilder = new StringBuilder();
      foreach (var plugin in plugins) {
        strBuilder.AppendLine(plugin.ToString());
      }
      return (new ConfirmationDialog("Confirm Update", "Do you want to update following plugins?", strBuilder.ToString())).ShowDialog() == DialogResult.OK;
    }

    private bool ConfirmInstallAction(IEnumerable<IPluginDescription> plugins) {
      foreach (var plugin in plugins) {
        if (!string.IsNullOrEmpty(plugin.LicenseText)) {
          var licenseConfirmationBox = new LicenseConfirmationBox(plugin);
          if (licenseConfirmationBox.ShowDialog() != DialogResult.OK)
            return false;
        }
      }
      return true;
    }


    #endregion

    #region helper methods

    private void UpdateLocalPluginList(IEnumerable<PluginDescription> plugins) {
      localPluginManagerView.Plugins = plugins;
    }

    private void UpdateRemotePluginList(
      IEnumerable<DeploymentService.ProductDescription> remoteProducts,
      IEnumerable<IPluginDescription> remotePlugins) {

      var mostRecentRemotePlugins = from remote in remotePlugins
                                    where !remotePlugins.Any(x => x.Name == remote.Name && x.Version > remote.Version) // same name and higher version
                                    select remote;

      var newPlugins = from remote in mostRecentRemotePlugins
                       let matchingLocal = (from local in localPluginManagerView.Plugins
                                            where local.Name == remote.Name
                                            where local.Version < remote.Version
                                            select local).FirstOrDefault()
                       where matchingLocal != null
                       select remote;

      remotePluginInstaller.NewPlugins = newPlugins;
      remotePluginInstaller.Products = remoteProducts;
      remotePluginInstaller.AllPlugins = remotePlugins;
    }

    private void RefreshRemotePluginListAsync() {
      Cursor = Cursors.AppStarting;
      toolStripProgressBar.Visible = true;
      refreshButton.Enabled = false;
      refreshServerPluginsBackgroundWorker.RunWorkerAsync();
    }

    private void RefreshLocalPluginListAsync() {
      Cursor = Cursors.AppStarting;
      toolStripProgressBar.Visible = true;
      DisableControls();
      refreshLocalPluginsBackgroundWorker.RunWorkerAsync();
    }

    private void UpdateControlsDisconnected() {
      //localPluginsListView.Enabled = false;
      //ClearPluginsList(remotePluginsListView);
      refreshButton.Enabled = true;
      toolStripProgressBar.Visible = false;
      Cursor = Cursors.Default;
    }

    private void UpdateControlsConnected() {
      refreshButton.Enabled = true;
      toolStripProgressBar.Visible = false;
      Cursor = Cursors.Default;
    }

    private void DisableControls() {
      refreshButton.Enabled = false;
      Cursor = Cursors.Default;
    }
    #endregion

    private void localPluginManager_ItemChecked(object sender, ItemCheckedEventArgs e) {
      removeButton.Enabled = localPluginManagerView.CheckedPlugins.Count() > 0;
    }

    private void remotePluginInstaller_ItemChecked(object sender, ItemCheckedEventArgs e) {
      installButton.Enabled = remotePluginInstaller.CheckedPlugins.Count() > 0;
    }

    private void editConnectionButton_Click(object sender, EventArgs e) {
    }

    protected override void OnClosing(CancelEventArgs e) {
      installationManager.PluginInstalled -= new EventHandler<PluginInfrastructureEventArgs>(installationManager_PluginInstalled);
      installationManager.PluginRemoved -= new EventHandler<PluginInfrastructureEventArgs>(installationManager_PluginRemoved);
      installationManager.PluginUpdated -= new EventHandler<PluginInfrastructureEventArgs>(installationManager_PluginUpdated);
      installationManager.PreInstallPlugin -= new EventHandler<PluginInfrastructureCancelEventArgs>(installationManager_PreInstallPlugin);
      installationManager.PreRemovePlugin -= new EventHandler<PluginInfrastructureCancelEventArgs>(installationManager_PreRemovePlugin);
      installationManager.PreUpdatePlugin -= new EventHandler<PluginInfrastructureCancelEventArgs>(installationManager_PreUpdatePlugin);
      base.OnClosing(e);
    }

    private void connectionSettingsToolStripMenuItem_Click(object sender, EventArgs e) {
      new ConnectionSetupView().ShowDialog();
    }

    private void tabControl_Selected(object sender, TabControlEventArgs e) {
      viewToolStripMenuItem.Enabled = e.TabPage == availablePluginsTabPage;
    }

    private void simpleToolStripMenuItem_Click(object sender, EventArgs e) {

    }

    private void advancedToolStripMenuItem_Click(object sender, EventArgs e) {

    }
  }
}

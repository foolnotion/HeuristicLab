﻿namespace HeuristicLab.Clients.Access.Administration {
  partial class RefreshableRoleSelectionListView {
    /// <summary> 
    /// Required designer variable.
    /// </summary>
    private System.ComponentModel.IContainer components = null;

    /// <summary> 
    /// Clean up any resources being used.
    /// </summary>
    /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
    protected override void Dispose(bool disposing) {
      if (disposing && (components != null)) {
        components.Dispose();
      }
      base.Dispose(disposing);
    }

    #region Component Designer generated code

    /// <summary> 
    /// Required method for Designer support - do not modify 
    /// the contents of this method with the code editor.
    /// </summary>
    private void InitializeComponent() {
      System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(RefreshableRoleSelectionListView));
      this.storeButton = new System.Windows.Forms.Button();
      this.roleSelectionListView = new HeuristicLab.Clients.Access.Administration.RoleSelectionListView();
      this.SuspendLayout();
      // 
      // storeButton
      // 
      this.storeButton.Enabled = false;
      this.storeButton.Image = HeuristicLab.Common.Resources.VSImageLibrary.PublishToWeb;
      this.storeButton.Location = new System.Drawing.Point(33, 3);
      this.storeButton.Name = "storeButton";
      this.storeButton.Size = new System.Drawing.Size(24, 24);
      this.storeButton.TabIndex = 3;
      this.storeButton.UseVisualStyleBackColor = true;
      this.storeButton.Click += new System.EventHandler(this.storeButton_Click);
      // 
      // roleSelectionListView
      // 
      this.roleSelectionListView.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)
                  | System.Windows.Forms.AnchorStyles.Left)
                  | System.Windows.Forms.AnchorStyles.Right)));
      this.roleSelectionListView.Caption = "RoleSelectionList View";
      this.roleSelectionListView.Content = null;
      this.roleSelectionListView.Location = new System.Drawing.Point(3, 33);
      this.roleSelectionListView.Name = "roleSelectionListView";
      this.roleSelectionListView.ReadOnly = false;
      this.roleSelectionListView.Size = new System.Drawing.Size(559, 375);
      this.roleSelectionListView.TabIndex = 4;
      // 
      // RefreshableRoleSelectionListView
      // 
      this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
      this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
      this.Controls.Add(this.storeButton);
      this.Controls.Add(this.roleSelectionListView);
      this.Name = "RefreshableRoleSelectionListView";
      this.Size = new System.Drawing.Size(565, 411);
      this.Controls.SetChildIndex(this.roleSelectionListView, 0);
      this.Controls.SetChildIndex(this.storeButton, 0);
      this.Controls.SetChildIndex(this.refreshButton, 0);
      this.ResumeLayout(false);

    }

    #endregion

    private System.Windows.Forms.Button storeButton;
    private RoleSelectionListView roleSelectionListView;
  }
}

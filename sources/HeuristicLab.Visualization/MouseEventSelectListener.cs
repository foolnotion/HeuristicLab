﻿using System.Windows.Forms;

namespace HeuristicLab.Visualization {
  internal class MouseEventSelectListener : IMouseEventListener {
    #region Interface members

    public event MouseEventHandler OnMouseDown;
    public event MouseEventHandler OnMouseUp;
    public event MouseEventHandler OnMouseMove;

    #endregion

    public MouseEventHandler OnSelectionChanged;
    public MouseEventHandler OnSelectionFinished;
  }
}
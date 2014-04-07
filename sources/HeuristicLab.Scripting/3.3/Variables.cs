﻿using System.Collections;
using System.Collections.Generic;
using System.Dynamic;

namespace HeuristicLab.Scripting {
  public class Variables : DynamicObject, IEnumerable<KeyValuePair<string, object>> {
    private readonly VariableStore variableStore;

    public Variables(VariableStore variableStore) {
      this.variableStore = variableStore;
    }

    public override bool TryGetMember(GetMemberBinder binder, out object result) {
      return variableStore.TryGetValue(binder.Name, out result);
    }

    public override bool TrySetMember(SetMemberBinder binder, object value) {
      variableStore[binder.Name] = value;
      return true;
    }

    public bool Contains(string variableName) {
      return variableStore.ContainsKey(variableName);
    }

    public void Clear() {
      variableStore.Clear();
    }

    public IEnumerator<KeyValuePair<string, object>> GetEnumerator() {
      return variableStore.GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator() {
      return GetEnumerator();
    }
  }
}

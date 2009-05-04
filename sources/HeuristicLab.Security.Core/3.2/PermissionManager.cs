﻿using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using HeuristicLab.Security.Contracts.Interfaces;
using HeuristicLab.Security.Contracts.BusinessObjects;
using HeuristicLab.Security.DataAccess;
using HeuristicLab.DataAccess.Interfaces;
using HeuristicLab.PluginInfrastructure;
using System.Security.Cryptography;

namespace HeuristicLab.Security.Core {
  public class PermissionManager : IPermissionManager{

    private static ISessionFactory factory = ServiceLocator.GetSessionFactory();

    private static ISession session;
    
    private static IDictionary<Guid,string> currentSessions = new Dictionary<Guid, string>();
    Object locker = new Object();

    private static string getMd5Hash(string input) {
      // Create a new instance of the MD5CryptoServiceProvider object.
      MD5 md5Hasher = MD5.Create();

      // Convert the input string to a byte array and compute the hash.
      byte[] data = md5Hasher.ComputeHash(Encoding.Default.GetBytes(input));

      // Create a new Stringbuilder to collect the bytes
      // and create a string.
      StringBuilder sBuilder = new StringBuilder();

      // Loop through each byte of the hashed data 
      // and format each one as a hexadecimal string.
      for (int i = 0; i < data.Length; i++) {
        sBuilder.Append(data[i].ToString("x2"));
      }

      // Return the hexadecimal string.
      return sBuilder.ToString();
    }

   /// <summary>
   /// If a session exists for this userName then it is returned, otherwise the given password
   /// is checked and a new session is created.
   /// </summary>
   /// <param name="userName"></param>
   /// <param name="password"></param>
   /// <returns></returns>
    public Guid Authenticate(String userName, String password) {
      lock (locker)
        if (currentSessions.Values.Contains(userName))
          return GetGuid(userName);
      try {
        session = factory.GetSessionForCurrentThread();

        password = getMd5Hash(password);

        IUserAdapter userAdapter = session.GetDataAdapter<User, IUserAdapter>();
        User user = userAdapter.GetByLogin(userName);

        if (user != null && 
            user.Password.Equals(password)) {
          Guid newSessionId = Guid.NewGuid();
          lock (locker)
            currentSessions.Add(newSessionId, userName);
          return newSessionId;
        } else return Guid.Empty;
      }
      finally {
        if (session != null)
          session.EndSession();
      }
    }

    /// <summary>
    /// Checks if the owner of the given session has the given permission.
    /// </summary>
    /// <param name="sessionId"></param>
    /// <param name="permissionId"></param>
    /// <param name="entityId"></param>
    /// <returns></returns>
    public bool CheckPermission(Guid sessionId, Guid permissionId, Guid entityId) {
      string userName;
      bool existsSession;
      lock (locker)
        existsSession = currentSessions.TryGetValue(sessionId, out userName);
      if (existsSession) {
        try {
          session = factory.GetSessionForCurrentThread();
          
          IPermissionOwnerAdapter permOwnerAdapter = session.GetDataAdapter<PermissionOwner, IPermissionOwnerAdapter>();
          PermissionOwner permOwner = permOwnerAdapter.GetByName(userName);

          IPermissionAdapter permissionAdapter = session.GetDataAdapter<Permission, IPermissionAdapter>();
          Permission permission = permissionAdapter.GetById(permissionId);
          
          if ((permission != null) && (permOwner != null))
            return (permissionAdapter.getPermission(permOwner.Id, permission.Id, entityId) != null);
          else return false;
        }
        finally {
          if (session != null)
            session.EndSession();
        }
      } else return false;
    }

    /// <summary>
    /// Removes the given session.
    /// </summary>
    /// <param name="sessionId"></param>
    public void EndSession(Guid sessionId) {
      lock (locker) {
        if (currentSessions.Keys.Contains(sessionId))
          currentSessions.Remove(sessionId);
      }
    }

    /// <summary>
    /// Gets the sessionId for a user.
    /// </summary>
    /// <param name="userName"></param>
    /// <returns></returns>
    public Guid GetGuid(string userName) {
      foreach (Guid guid in currentSessions.Keys)
        if (currentSessions[guid].CompareTo(userName) == 0)
          return guid;
      return Guid.Empty;
    }
  } 
}

﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using HeuristicLab.Hive.Contracts.Interfaces;
using HeuristicLab.Hive.Contracts.BusinessObjects;
using HeuristicLab.Hive.Server.Core.InternalInterfaces.DataAccess;
using System.Resources;
using System.Reflection;
using HeuristicLab.Hive.Contracts;

namespace HeuristicLab.Hive.Server.Core {
  class UserRoleManager: IUserRoleManager {

    IUserAdapter userAdapter;
    IUserGroupAdapter userGroupAdapter;

    public UserRoleManager() {
      userAdapter = ServiceLocator.GetUserAdapter();
      userGroupAdapter = ServiceLocator.GetUserGroupAdapter();
    }

    #region IUserRoleManager Members

    public ResponseList<User> GetAllUsers() {
      ResponseList<User> response = new ResponseList<User>();

      List<User> allUsers = new List<User>(userAdapter.GetAllUsers());
      response.List = allUsers;
      response.Success = true;
      response.StatusMessage = ApplicationConstants.RESPONSE_USERROLE_GET_ALL_USERS;

      return response;
    }

    public Response AddNewUser(User user) {
      Response response = new Response();

      if (user.PermissionOwnerId != 0) {
        response.Success = false;
        response.StatusMessage = ApplicationConstants.RESPONSE_USERROLE_ID_MUST_NOT_BE_SET;
        return response;
      }
      userAdapter.UpdateUser(user);
      response.Success = true;
      response.StatusMessage = ApplicationConstants.RESPONSE_USERROLE_USER_SUCCESSFULLY_ADDED;
      
      return response;
    }

    public ResponseList<UserGroup> GetAllUserGroups() {
      ResponseList<UserGroup> response = new ResponseList<UserGroup>();

      response.List = new List<UserGroup>(userGroupAdapter.GetAllUserGroups());
      response.Success = true;
      response.StatusMessage = ApplicationConstants.RESPONSE_USERROLE_GET_ALL_USERGROUPS;

      return response;
    }

    public Response RemoveUser(long userId) {
      Response response = new Response();
      User user = userAdapter.GetUserById(userId);
      if (user == null) {
        response.Success = false;
        response.StatusMessage = ApplicationConstants.RESPONSE_USERROLE_USER_DOESNT_EXIST;
        return response;
      }
      userAdapter.DeleteUser(user);
      response.Success = true;
      response.StatusMessage = ApplicationConstants.RESPONSE_USERROLE_USER_REMOVED;
                         
      return response;
    }

    public Response AddNewUserGroup(UserGroup userGroup) {
      Response response = new Response();
      
      if (userGroup.PermissionOwnerId != 0) {
        response.Success = false;
        response.StatusMessage = ApplicationConstants.RESPONSE_USERROLE_ID_MUST_NOT_BE_SET;
        return response;
      }
      userGroupAdapter.UpdateUserGroup(userGroup);
      response.Success = false;
      response.StatusMessage = ApplicationConstants.RESPONSE_USERROLE_USERGROUP_ADDED;

      return response;
    }

    public Response RemoveUserGroup(long groupId) {
      Response response = new Response();

      UserGroup userGroupFromDb = userGroupAdapter.GetUserGroupById(groupId);
      if (userGroupFromDb == null) {
        response.Success = false;
        response.StatusMessage = ApplicationConstants.RESPONSE_USERROLE_USERGROUP_DOESNT_EXIST;
        return response;
      }
      userGroupAdapter.DeleteUserGroup(userGroupFromDb);
      response.Success = false;
      response.StatusMessage = ApplicationConstants.RESPONSE_USERROLE_USERGROUP_ADDED;

      return response;
    }

    public Response AddPermissionOwnerToGroup(long groupId, PermissionOwner permissionOwner) {
      Response response = new Response();

      if (permissionOwner.PermissionOwnerId != 0) {
        response.Success = false;
        response.StatusMessage = ApplicationConstants.RESPONSE_USERROLE_ID_MUST_NOT_BE_SET;
        return response;
      }

      UserGroup userGroup = userGroupAdapter.GetUserGroupById(groupId);
      if (userGroup == null) {
        response.Success = false;
        response.StatusMessage = ApplicationConstants.RESPONSE_USERROLE_USERGROUP_DOESNT_EXIST;
        return response;
      }
      userGroup.Members.Add(permissionOwner);
      userGroupAdapter.UpdateUserGroup(userGroup);

      response.Success = true;
      response.StatusMessage = ApplicationConstants.RESPONSE_USERROLE_PERMISSIONOWNER_ADDED;

      return response;
    }

    public Response RemovePermissionOwnerFromGroup(long groupId, long permissionOwnerId) {
      Response response = new Response();

      UserGroup userGroup = userGroupAdapter.GetUserGroupById(groupId);
      if (userGroup == null) {
        response.Success = false;
        response.StatusMessage = ApplicationConstants.RESPONSE_USERROLE_USERGROUP_DOESNT_EXIST;
        return response;
      }
      User user = userAdapter.GetUserById(permissionOwnerId);
      if (user == null) {
        response.Success = false;
        response.StatusMessage = ApplicationConstants.RESPONSE_USERROLE_PERMISSIONOWNER_DOESNT_EXIST;
        return response;
      }
      foreach (PermissionOwner permissionOwner in userGroup.Members) {
        if (permissionOwner.PermissionOwnerId == permissionOwnerId) {
          userGroup.Members.Remove(permissionOwner);
          userGroupAdapter.UpdateUserGroup(userGroup);
          response.Success = true;
          response.StatusMessage = ApplicationConstants.RESPONSE_USERROLE_PERMISSIONOWNER_REMOVED;
          return response;
        }
      }
      response.Success = false;
      response.StatusMessage = ApplicationConstants.RESPONSE_USERROLE_PERMISSIONOWNER_DOESNT_EXIST;
      
      return response;
    }

    #endregion
  }
}

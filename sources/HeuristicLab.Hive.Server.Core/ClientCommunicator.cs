﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using HeuristicLab.Hive.Contracts.BusinessObjects;
using HeuristicLab.Hive.Contracts.Interfaces;
using HeuristicLab.Hive.Contracts;
using HeuristicLab.Core;
using HeuristicLab.Hive.Server.Core.InternalInterfaces.DataAccess;
using System.Resources;
using System.Reflection;

namespace HeuristicLab.Hive.Server.Core {
  /// <summary>
  /// The ClientCommunicator manages the whole communication with the client
  /// </summary>
  public class ClientCommunicator: IClientCommunicator {
    LinkedList<long> jobs;
    int nrOfJobs = 1;

    IClientAdapter clientAdapter;

    public ClientCommunicator() {
      clientAdapter = ServiceLocator.GetClientAdapter(); 

      jobs = new LinkedList<long>();
      for (long i = 0; i < nrOfJobs; i++) {
        jobs.AddFirst(i);
      }
    }

    #region IClientCommunicator Members

    public Response Login(ClientInfo clientInfo) {
      Response response = new Response();
      response.Success = true;

      ICollection<ClientInfo> allClients = clientAdapter.GetAllClients();
      ClientInfo client = clientAdapter.GetClientById(clientInfo.ClientId);
      if (client != null) {
        if (client.State != State.offline) {
          response.Success = false;
          response.StatusMessage = ApplicationConstants.RESPONSE_COMMUNICATOR_LOGIN_USER_ALLREADY_ONLINE;
        }
      } 

      if (response.Success) {
        clientAdapter.UpdateClient(clientInfo);
        response.StatusMessage = ApplicationConstants.RESPONSE_COMMUNICATOR_LOGIN_SUCCESS;
      }

      return response;
    }

    public ResponseHB SendHeartBeat(HeartBeatData hbData) {
      ResponseHB response = new ResponseHB();

      response.Success = true;
      response.StatusMessage = ApplicationConstants.RESPONSE_COMMUNICATOR_HARDBEAT_RECEIVED;
      response.ActionRequest = new List<MessageContainer>();
      if (jobs.Count > 0) 
        response.ActionRequest.Add(new MessageContainer(MessageContainer.MessageType.FetchJob));
      else
        response.ActionRequest.Add(new MessageContainer(MessageContainer.MessageType.NoMessage));

      return response;
    }

    public ResponseJob PullJob(Guid clientId) {
      ResponseJob response = new ResponseJob();
      lock (this) {
        response.JobId = jobs.Last.Value;
        jobs.RemoveLast();
        response.SerializedJob = PersistenceManager.SaveToGZip(new TestJob());
      }
      
      response.Success = true;
      response.StatusMessage = ApplicationConstants.RESPONSE_COMMUNICATOR_JOB_PULLED;
      return response;
    }

    public ResponseResultReceived SendJobResult(JobResult Result, bool finished) {
      ResponseResultReceived response = new ResponseResultReceived();
      response.Success = true;
      response.StatusMessage = ApplicationConstants.RESPONSE_COMMUNICATOR_JOBRESULT_RECEIVED;
      response.JobId = Result.JobId;

      return response;
    }
                           
    public Response Logout(Guid clientId) {
      Response response = new Response();
      
      ClientInfo client = clientAdapter.GetClientById(clientId);
      if (client == null) {
        response.Success = false;
        response.StatusMessage = ApplicationConstants.RESPONSE_COMMUNICATOR_LOGOUT_CLIENT_NOT_REGISTERED;
        return response;
      }
      client.State = State.offline;
      clientAdapter.UpdateClient(client);

      response.Success = true;
      response.StatusMessage = ApplicationConstants.RESPONSE_COMMUNICATOR_LOGOUT_SUCCESS;
      
      return response;
    }

    #endregion
  }
}

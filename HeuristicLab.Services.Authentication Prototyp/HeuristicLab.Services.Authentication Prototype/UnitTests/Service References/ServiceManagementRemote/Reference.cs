﻿//------------------------------------------------------------------------------
// <auto-generated>
//     Dieser Code wurde von einem Tool generiert.
//     Laufzeitversion:2.0.50727.4927
//
//     Änderungen an dieser Datei können falsches Verhalten verursachen und gehen verloren, wenn
//     der Code erneut generiert wird.
// </auto-generated>
//------------------------------------------------------------------------------

namespace UnitTests.ServiceManagementRemote {
    
    
    [System.CodeDom.Compiler.GeneratedCodeAttribute("System.ServiceModel", "3.0.0.0")]
    [System.ServiceModel.ServiceContractAttribute(ConfigurationName="ServiceManagementRemote.IAuthorizationManagementService")]
    public interface IAuthorizationManagementService {
        
        [System.ServiceModel.OperationContractAttribute(Action="http://tempuri.org/IAuthorizationManagementService/CreateRole", ReplyAction="http://tempuri.org/IAuthorizationManagementService/CreateRoleResponse")]
        void CreateRole(string roleName, bool isPermission);
    }
    
    [System.CodeDom.Compiler.GeneratedCodeAttribute("System.ServiceModel", "3.0.0.0")]
    public interface IAuthorizationManagementServiceChannel : UnitTests.ServiceManagementRemote.IAuthorizationManagementService, System.ServiceModel.IClientChannel {
    }
    
    [System.Diagnostics.DebuggerStepThroughAttribute()]
    [System.CodeDom.Compiler.GeneratedCodeAttribute("System.ServiceModel", "3.0.0.0")]
    public partial class AuthorizationManagementServiceClient : System.ServiceModel.ClientBase<UnitTests.ServiceManagementRemote.IAuthorizationManagementService>, UnitTests.ServiceManagementRemote.IAuthorizationManagementService {
        
        public AuthorizationManagementServiceClient() {
        }
        
        public AuthorizationManagementServiceClient(string endpointConfigurationName) : 
                base(endpointConfigurationName) {
        }
        
        public AuthorizationManagementServiceClient(string endpointConfigurationName, string remoteAddress) : 
                base(endpointConfigurationName, remoteAddress) {
        }
        
        public AuthorizationManagementServiceClient(string endpointConfigurationName, System.ServiceModel.EndpointAddress remoteAddress) : 
                base(endpointConfigurationName, remoteAddress) {
        }
        
        public AuthorizationManagementServiceClient(System.ServiceModel.Channels.Binding binding, System.ServiceModel.EndpointAddress remoteAddress) : 
                base(binding, remoteAddress) {
        }
        
        public void CreateRole(string roleName, bool isPermission) {
            base.Channel.CreateRole(roleName, isPermission);
        }
    }
}

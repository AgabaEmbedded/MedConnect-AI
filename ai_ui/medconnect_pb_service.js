// package: medconnect
// file: medconnect.proto

var medconnect_pb = require("./medconnect_pb");
var grpc = require("@improbable-eng/grpc-web").grpc;

var MedConnectService = (function () {
  function MedConnectService() {}
  MedConnectService.serviceName = "medconnect.MedConnectService";
  return MedConnectService;
}());

MedConnectService.HealthCheck = {
  methodName: "HealthCheck",
  service: MedConnectService,
  requestStream: false,
  responseStream: false,
  requestType: medconnect_pb.HealthCheckRequest,
  responseType: medconnect_pb.HealthCheckResponse
};

MedConnectService.Conversation = {
  methodName: "Conversation",
  service: MedConnectService,
  requestStream: false,
  responseStream: false,
  requestType: medconnect_pb.ConversationRequest,
  responseType: medconnect_pb.ConversationResponse
};

MedConnectService.Reset = {
  methodName: "Reset",
  service: MedConnectService,
  requestStream: false,
  responseStream: false,
  requestType: medconnect_pb.ResetRequest,
  responseType: medconnect_pb.ResetResponse
};

MedConnectService.Translate = {
  methodName: "Translate",
  service: MedConnectService,
  requestStream: false,
  responseStream: false,
  requestType: medconnect_pb.TranslateRequest,
  responseType: medconnect_pb.TranslateResponse
};

exports.MedConnectService = MedConnectService;

function MedConnectServiceClient(serviceHost, options) {
  this.serviceHost = serviceHost;
  this.options = options || {};
}

MedConnectServiceClient.prototype.healthCheck = function healthCheck(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(MedConnectService.HealthCheck, {
    request: requestMessage,
    host: this.serviceHost,
    metadata: metadata,
    transport: this.options.transport,
    debug: this.options.debug,
    onEnd: function (response) {
      if (callback) {
        if (response.status !== grpc.Code.OK) {
          var err = new Error(response.statusMessage);
          err.code = response.status;
          err.metadata = response.trailers;
          callback(err, null);
        } else {
          callback(null, response.message);
        }
      }
    }
  });
  return {
    cancel: function () {
      callback = null;
      client.close();
    }
  };
};

MedConnectServiceClient.prototype.conversation = function conversation(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(MedConnectService.Conversation, {
    request: requestMessage,
    host: this.serviceHost,
    metadata: metadata,
    transport: this.options.transport,
    debug: this.options.debug,
    onEnd: function (response) {
      if (callback) {
        if (response.status !== grpc.Code.OK) {
          var err = new Error(response.statusMessage);
          err.code = response.status;
          err.metadata = response.trailers;
          callback(err, null);
        } else {
          callback(null, response.message);
        }
      }
    }
  });
  return {
    cancel: function () {
      callback = null;
      client.close();
    }
  };
};

MedConnectServiceClient.prototype.reset = function reset(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(MedConnectService.Reset, {
    request: requestMessage,
    host: this.serviceHost,
    metadata: metadata,
    transport: this.options.transport,
    debug: this.options.debug,
    onEnd: function (response) {
      if (callback) {
        if (response.status !== grpc.Code.OK) {
          var err = new Error(response.statusMessage);
          err.code = response.status;
          err.metadata = response.trailers;
          callback(err, null);
        } else {
          callback(null, response.message);
        }
      }
    }
  });
  return {
    cancel: function () {
      callback = null;
      client.close();
    }
  };
};

MedConnectServiceClient.prototype.translate = function translate(requestMessage, metadata, callback) {
  if (arguments.length === 2) {
    callback = arguments[1];
  }
  var client = grpc.unary(MedConnectService.Translate, {
    request: requestMessage,
    host: this.serviceHost,
    metadata: metadata,
    transport: this.options.transport,
    debug: this.options.debug,
    onEnd: function (response) {
      if (callback) {
        if (response.status !== grpc.Code.OK) {
          var err = new Error(response.statusMessage);
          err.code = response.status;
          err.metadata = response.trailers;
          callback(err, null);
        } else {
          callback(null, response.message);
        }
      }
    }
  });
  return {
    cancel: function () {
      callback = null;
      client.close();
    }
  };
};

exports.MedConnectServiceClient = MedConnectServiceClient;


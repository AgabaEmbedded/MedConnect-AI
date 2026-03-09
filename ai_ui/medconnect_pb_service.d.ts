// package: medconnect
// file: medconnect.proto

import * as medconnect_pb from "./medconnect_pb";
import {grpc} from "@improbable-eng/grpc-web";

type MedConnectServiceHealthCheck = {
  readonly methodName: string;
  readonly service: typeof MedConnectService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof medconnect_pb.HealthCheckRequest;
  readonly responseType: typeof medconnect_pb.HealthCheckResponse;
};

type MedConnectServiceConversation = {
  readonly methodName: string;
  readonly service: typeof MedConnectService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof medconnect_pb.ConversationRequest;
  readonly responseType: typeof medconnect_pb.ConversationResponse;
};

type MedConnectServiceReset = {
  readonly methodName: string;
  readonly service: typeof MedConnectService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof medconnect_pb.ResetRequest;
  readonly responseType: typeof medconnect_pb.ResetResponse;
};

type MedConnectServiceTranslate = {
  readonly methodName: string;
  readonly service: typeof MedConnectService;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof medconnect_pb.TranslateRequest;
  readonly responseType: typeof medconnect_pb.TranslateResponse;
};

export class MedConnectService {
  static readonly serviceName: string;
  static readonly HealthCheck: MedConnectServiceHealthCheck;
  static readonly Conversation: MedConnectServiceConversation;
  static readonly Reset: MedConnectServiceReset;
  static readonly Translate: MedConnectServiceTranslate;
}

export type ServiceError = { message: string, code: number; metadata: grpc.Metadata }
export type Status = { details: string, code: number; metadata: grpc.Metadata }

interface UnaryResponse {
  cancel(): void;
}
interface ResponseStream<T> {
  cancel(): void;
  on(type: 'data', handler: (message: T) => void): ResponseStream<T>;
  on(type: 'end', handler: (status?: Status) => void): ResponseStream<T>;
  on(type: 'status', handler: (status: Status) => void): ResponseStream<T>;
}
interface RequestStream<T> {
  write(message: T): RequestStream<T>;
  end(): void;
  cancel(): void;
  on(type: 'end', handler: (status?: Status) => void): RequestStream<T>;
  on(type: 'status', handler: (status: Status) => void): RequestStream<T>;
}
interface BidirectionalStream<ReqT, ResT> {
  write(message: ReqT): BidirectionalStream<ReqT, ResT>;
  end(): void;
  cancel(): void;
  on(type: 'data', handler: (message: ResT) => void): BidirectionalStream<ReqT, ResT>;
  on(type: 'end', handler: (status?: Status) => void): BidirectionalStream<ReqT, ResT>;
  on(type: 'status', handler: (status: Status) => void): BidirectionalStream<ReqT, ResT>;
}

export class MedConnectServiceClient {
  readonly serviceHost: string;

  constructor(serviceHost: string, options?: grpc.RpcOptions);
  healthCheck(
    requestMessage: medconnect_pb.HealthCheckRequest,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: medconnect_pb.HealthCheckResponse|null) => void
  ): UnaryResponse;
  healthCheck(
    requestMessage: medconnect_pb.HealthCheckRequest,
    callback: (error: ServiceError|null, responseMessage: medconnect_pb.HealthCheckResponse|null) => void
  ): UnaryResponse;
  conversation(
    requestMessage: medconnect_pb.ConversationRequest,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: medconnect_pb.ConversationResponse|null) => void
  ): UnaryResponse;
  conversation(
    requestMessage: medconnect_pb.ConversationRequest,
    callback: (error: ServiceError|null, responseMessage: medconnect_pb.ConversationResponse|null) => void
  ): UnaryResponse;
  reset(
    requestMessage: medconnect_pb.ResetRequest,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: medconnect_pb.ResetResponse|null) => void
  ): UnaryResponse;
  reset(
    requestMessage: medconnect_pb.ResetRequest,
    callback: (error: ServiceError|null, responseMessage: medconnect_pb.ResetResponse|null) => void
  ): UnaryResponse;
  translate(
    requestMessage: medconnect_pb.TranslateRequest,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: medconnect_pb.TranslateResponse|null) => void
  ): UnaryResponse;
  translate(
    requestMessage: medconnect_pb.TranslateRequest,
    callback: (error: ServiceError|null, responseMessage: medconnect_pb.TranslateResponse|null) => void
  ): UnaryResponse;
}


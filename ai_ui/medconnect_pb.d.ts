// package: medconnect
// file: medconnect.proto

import * as jspb from "google-protobuf";

export class HealthCheckRequest extends jspb.Message {
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): HealthCheckRequest.AsObject;
  static toObject(includeInstance: boolean, msg: HealthCheckRequest): HealthCheckRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: HealthCheckRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): HealthCheckRequest;
  static deserializeBinaryFromReader(message: HealthCheckRequest, reader: jspb.BinaryReader): HealthCheckRequest;
}

export namespace HealthCheckRequest {
  export type AsObject = {
  }
}

export class HealthCheckResponse extends jspb.Message {
  getStatus(): string;
  setStatus(value: string): void;

  getMessage(): string;
  setMessage(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): HealthCheckResponse.AsObject;
  static toObject(includeInstance: boolean, msg: HealthCheckResponse): HealthCheckResponse.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: HealthCheckResponse, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): HealthCheckResponse;
  static deserializeBinaryFromReader(message: HealthCheckResponse, reader: jspb.BinaryReader): HealthCheckResponse;
}

export namespace HealthCheckResponse {
  export type AsObject = {
    status: string,
    message: string,
  }
}

export class ConversationRequest extends jspb.Message {
  getMessage(): string;
  setMessage(value: string): void;

  getAudio(): string;
  setAudio(value: string): void;

  getPremium(): boolean;
  setPremium(value: boolean): void;

  getLanguage(): string;
  setLanguage(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ConversationRequest.AsObject;
  static toObject(includeInstance: boolean, msg: ConversationRequest): ConversationRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ConversationRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ConversationRequest;
  static deserializeBinaryFromReader(message: ConversationRequest, reader: jspb.BinaryReader): ConversationRequest;
}

export namespace ConversationRequest {
  export type AsObject = {
    message: string,
    audio: string,
    premium: boolean,
    language: string,
  }
}

export class ConversationResponse extends jspb.Message {
  getMessage(): string;
  setMessage(value: string): void;

  getAudio(): string;
  setAudio(value: string): void;

  getDoctorId(): string;
  setDoctorId(value: string): void;

  getMedicalSummary(): string;
  setMedicalSummary(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ConversationResponse.AsObject;
  static toObject(includeInstance: boolean, msg: ConversationResponse): ConversationResponse.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ConversationResponse, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ConversationResponse;
  static deserializeBinaryFromReader(message: ConversationResponse, reader: jspb.BinaryReader): ConversationResponse;
}

export namespace ConversationResponse {
  export type AsObject = {
    message: string,
    audio: string,
    doctorId: string,
    medicalSummary: string,
  }
}

export class ResetRequest extends jspb.Message {
  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ResetRequest.AsObject;
  static toObject(includeInstance: boolean, msg: ResetRequest): ResetRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ResetRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ResetRequest;
  static deserializeBinaryFromReader(message: ResetRequest, reader: jspb.BinaryReader): ResetRequest;
}

export namespace ResetRequest {
  export type AsObject = {
  }
}

export class ResetResponse extends jspb.Message {
  getStatus(): string;
  setStatus(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ResetResponse.AsObject;
  static toObject(includeInstance: boolean, msg: ResetResponse): ResetResponse.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ResetResponse, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ResetResponse;
  static deserializeBinaryFromReader(message: ResetResponse, reader: jspb.BinaryReader): ResetResponse;
}

export namespace ResetResponse {
  export type AsObject = {
    status: string,
  }
}

export class TranslateRequest extends jspb.Message {
  getMessage(): string;
  setMessage(value: string): void;

  getSourceLanguage(): string;
  setSourceLanguage(value: string): void;

  getTargetLanguage(): string;
  setTargetLanguage(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): TranslateRequest.AsObject;
  static toObject(includeInstance: boolean, msg: TranslateRequest): TranslateRequest.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: TranslateRequest, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): TranslateRequest;
  static deserializeBinaryFromReader(message: TranslateRequest, reader: jspb.BinaryReader): TranslateRequest;
}

export namespace TranslateRequest {
  export type AsObject = {
    message: string,
    sourceLanguage: string,
    targetLanguage: string,
  }
}

export class TranslateResponse extends jspb.Message {
  getMessage(): string;
  setMessage(value: string): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): TranslateResponse.AsObject;
  static toObject(includeInstance: boolean, msg: TranslateResponse): TranslateResponse.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: TranslateResponse, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): TranslateResponse;
  static deserializeBinaryFromReader(message: TranslateResponse, reader: jspb.BinaryReader): TranslateResponse;
}

export namespace TranslateResponse {
  export type AsObject = {
    message: string,
  }
}


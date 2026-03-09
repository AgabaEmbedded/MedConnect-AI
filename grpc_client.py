"""
Test client for MedConnect gRPC server (same idea as CKD grpc_client).
Run the server first: python grpc_server.py
Then: python grpc_client.py
"""
import grpc
import medconnect_pb2
import medconnect_pb2_grpc


def test_health(channel):
    stub = medconnect_pb2_grpc.MedConnectServiceStub(channel)
    req = medconnect_pb2.HealthCheckRequest()
    resp = stub.HealthCheck(req)
    print(f"HealthCheck: status={resp.status} message={resp.message}")


def test_conversation(channel, message="I have a headache", language="english", premium=False):
    stub = medconnect_pb2_grpc.MedConnectServiceStub(channel)
    req = medconnect_pb2.ConversationRequest(
        message=message,
        audio="",
        premium=premium,
        language=language,
    )
    resp = stub.Conversation(req)
    print(f"Conversation: message={resp.message[:80]}...")
    if resp.doctor_id:
        print(f"  doctor_id={resp.doctor_id}")
    if resp.medical_summary:
        print(f"  medical_summary={resp.medical_summary[:80]}...")
    return resp


def test_reset(channel):
    stub = medconnect_pb2_grpc.MedConnectServiceStub(channel)
    req = medconnect_pb2.ResetRequest()
    resp = stub.Reset(req)
    print(f"Reset: status={resp.status}")


def test_translate(channel, message="Hello", source="english", target="hausa"):
    stub = medconnect_pb2_grpc.MedConnectServiceStub(channel)
    req = medconnect_pb2.TranslateRequest(
        message=message,
        source_language=source,
        target_language=target,
    )
    resp = stub.Translate(req)
    print(f"Translate ({source} -> {target}): {resp.message!r}")
    return resp


def main():
    import sys
    # Usage: python grpc_client.py [host] [port]  (default: localhost 50052)
    # From laptop to VPS: use VPS public IP and port 50052, or use SSH tunnel (see README).
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = sys.argv[2] if len(sys.argv) > 2 else "50052"
    address = f"{host}:{port}"

    print(f"Connecting to {address}...")
    with grpc.insecure_channel(address) as channel:
        test_health(channel)
        print()
        test_conversation(channel, message="Hello, I need medical help.")
        print()
        test_translate(channel, message="Hello", source="english", target="hausa")
        print()
        test_reset(channel)


if __name__ == "__main__":
    main()

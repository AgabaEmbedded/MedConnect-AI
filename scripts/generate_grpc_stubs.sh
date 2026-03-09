#!/usr/bin/env bash
# Generate Python gRPC stubs from medconnect.proto (same pattern as ckd_ai_tools_api)
set -e
cd "$(dirname "$0")/.."
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. medconnect.proto
echo "Generated medconnect_pb2.py and medconnect_pb2_grpc.py"

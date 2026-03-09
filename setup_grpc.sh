#!/bin/bash
set -e
[ -f "medconnect.proto" ] || { echo "Run from medconnect_ai"; exit 1; }
[ -d ".venv" ] && source .venv/bin/activate
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. medconnect.proto
echo "Generated: medconnect_pb2.py, medconnect_pb2_grpc.py"

#!/usr/bin/env bash
# Generate JavaScript stub files from medconnect.proto for the AI UI Constructor.
# Official guide: https://dev.singularitynet.io (Generating Stubs for JS)
#
# Prerequisites:
#   1. Install protoc (https://github.com/protocolbuffers/protobuf/releases)
#   2. Run from medconnect_ai: npm install
#
# Output: ai_ui/medconnect_pb.js, ai_ui/medconnect_pb_service.js (and possibly .ts)

set -e
cd "$(dirname "$0")/.."
PROTO_FILE="medconnect.proto"
OUT_DIR="./ai_ui"
PACKAGE_NAME="medconnect"
ORG_ID="AJ_dev_outreach_test_1"
SERVICE_ID="medconnect"
NAMESPACE_PREFIX="${PACKAGE_NAME}_${ORG_ID}_${SERVICE_ID}"

if [ ! -f "$PROTO_FILE" ]; then
  echo "Error: $PROTO_FILE not found. Run this script from medconnect_ai."
  exit 1
fi
if [ ! -d "node_modules" ] || [ ! -d "node_modules/ts-protoc-gen" ]; then
  echo "Run: npm install"
  exit 1
fi
if ! command -v protoc &>/dev/null; then
  echo "Error: protoc not found. Install from https://github.com/protocolbuffers/protobuf/releases"
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "Generating stubs into $OUT_DIR (namespace_prefix=$NAMESPACE_PREFIX) ..."
export PATH="./node_modules/.bin:$PATH"
protoc \
  --plugin=protoc-gen-ts=./node_modules/.bin/protoc-gen-ts \
  --js_out="import_style=commonjs,binary,namespace_prefix=${NAMESPACE_PREFIX}:${OUT_DIR}" \
  --ts_out="service=grpc-web:${OUT_DIR}" \
  "$PROTO_FILE"
echo "Done. Upload medconnect_pb.js and medconnect_pb_service.js (or .ts) to the AI UI Constructor."
ls -la "$OUT_DIR"/medconnect_pb* 2>/dev/null || true

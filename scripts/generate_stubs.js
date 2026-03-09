/**
 * Generate JS/TS stub files from medconnect.proto for the AI UI Constructor.
 * Run from medconnect_ai: npm run generate-stubs
 * Prerequisites: protoc on PATH, npm install done.
 */
const { execSync } = require("child_process");
const path = require("path");
const fs = require("fs");

const root = path.join(__dirname, "..");
const protoFile = "medconnect.proto";
const outDir = path.join(root, "ai_ui");
const packageName = "medconnect";
const orgId = "AJ_dev_outreach_test_1";
const serviceId = "medconnect";
const namespacePrefix = `${packageName}_${orgId}_${serviceId}`;

const protoPath = path.join(root, protoFile);
const pluginPath = path.join(root, "node_modules", ".bin", "protoc-gen-ts" + (process.platform === "win32" ? ".cmd" : ""));

if (!fs.existsSync(protoPath)) {
  console.error("Error: medconnect.proto not found. Run from medconnect_ai.");
  process.exit(1);
}
if (!fs.existsSync(path.join(root, "node_modules", "ts-protoc-gen"))) {
  console.error("Error: Run npm install first (ts-protoc-gen not found).");
  process.exit(1);
}
if (!fs.existsSync(pluginPath) && !fs.existsSync(pluginPath.replace(".cmd", ""))) {
  console.error("Error: Run npm install first (protoc-gen-ts not found).");
  process.exit(1);
}

if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

const jsOut = `import_style=commonjs,binary,namespace_prefix=${namespacePrefix}:${outDir}`;
const tsOut = `service=grpc-web:${outDir}`;
const cmd = [
  "protoc",
  `--plugin=protoc-gen-ts=${pluginPath}`,
  `--js_out=${jsOut}`,
  `--ts_out=${tsOut}`,
  protoFile
].join(" ");

const binDir = path.join(root, "node_modules", ".bin");
const pathSep = process.platform === "win32" ? ";" : ":";
const env = { ...process.env, PATH: binDir + pathSep + (process.env.PATH || "") };

console.log("Generating stubs into ai_ui/ ...");
process.chdir(root);
execSync(cmd, { stdio: "inherit", env });
console.log("Done. Upload medconnect_pb.js and medconnect_pb_service.js (or .ts) to the AI UI Constructor.");

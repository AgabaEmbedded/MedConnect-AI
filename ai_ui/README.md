# MedConnect AI UI (Constructor / Sandbox v2)

Use the **AI UI Constructor** to build and publish the DEMO UI for the MedConnect service. Stub files must be **generated from medconnect.proto**; do not create them by hand.

**Flow:** Generate stubs → Upload stubs to Constructor → Use this **index.js** and **style.css** → Set OrganisationID, ServiceId, Endpoint → Compile → Export → Publish.

## 1. Generate stub files

From the **medconnect_ai** repo root: install protoc, run `npm install`, then `npm run generate-stubs`. You get **medconnect_pb.js** and **medconnect_pb_service.js** in ai_ui/.

## 2. AI UI Constructor workflow

1. Open AI UI Constructor, sign in with Marketplace account (MetaMask + Sepolia).
2. New Project → Upload **medconnect_pb.js** and **medconnect_pb_service.js**.
3. Paste **index.js** and **style.css** from this folder.
4. Set OrganisationID, ServiceId (`medconnect`), Endpoint (e.g. `https://<vps-host>:10000`).
5. Compile, test, Export, Publish via Publisher Portal.

## 3. What index.js does

HealthCheck, Conversation (with language), Translate, Reset. Uses Sandbox `serviceClient.unary()` pattern.

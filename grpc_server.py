"""
gRPC Server for MedConnect-AI
Reads the project (graph, run_conversation_turn, conversation_states) and exposes
HealthCheck, Conversation, Reset, Translate for SingularityNET – same pattern as CKD.
"""
import os
import sys
import base64
import logging

# Load env before importing main (main requires GEMINI_API_KEY)
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    logging.warning("GEMINI_API_KEY not set; main will raise on import.")

import grpc
from concurrent import futures

# Add project root
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import medconnect_pb2
import medconnect_pb2_grpc

# Import project logic (same as FastAPI app uses)
import main
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


class MedConnectServiceServicer(medconnect_pb2_grpc.MedConnectServiceServicer):
    """Implements MedConnectService by calling into main.graph and run_conversation_turn."""

    def HealthCheck(self, request, context):
        try:
            graph_ok = main.graph is not None
            openai_ok = main.client_manager.openai_client is not None
            trans_ok = main.client_manager.translate_client is not None
            msg = f"graph={graph_ok}, openai={openai_ok}, translation={trans_ok}"
            return medconnect_pb2.HealthCheckResponse(status="healthy", message=msg)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return medconnect_pb2.HealthCheckResponse(status="unhealthy", message=str(e))

    def Conversation(self, request, context):
        try:
            user_input = main.UserMessage(
                message=request.message or "",
                audio=request.audio or "",
                premium=request.premium,
                language=request.language or "english",
            )
            state = main.conversation_states.get("default")
            state = main.run_conversation_turn(main.graph, user_input, state)
            main.conversation_states["default"] = state

            if not state.get("messages"):
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("No response generated")
                return medconnect_pb2.ConversationResponse()

            last_message = state["messages"][-1]
            if not isinstance(last_message, AIMessage):
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("No agent response in state")
                return medconnect_pb2.ConversationResponse()

            message = last_message.content
            base64_audio = ""
            if user_input.audio and main.spitch_client is not None:
                resp = main.spitch_client.speech.generate(
                    text=message,
                    language=state["language"][:2],
                    voice=main.client_manager.voice_dict.get(state["language"], "comfort"),
                    format="mp3",
                )
                base64_audio = base64.b64encode(resp.read()).decode("utf-8")

            doctor_id = ""
            medical_summary = ""
            if state.get("is_doctor_id"):
                state["is_doctor_id"] = False
                doctor_id = state.get("selected_doctor", "")
                medical_summary = state.get("soap_summary") or ""

            return medconnect_pb2.ConversationResponse(
                message=message,
                audio=base64_audio,
                doctor_id=doctor_id,
                medical_summary=medical_summary,
            )
        except Exception as e:
            logger.exception("Conversation failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return medconnect_pb2.ConversationResponse()

    def Reset(self, request, context):
        try:
            main.conversation_states.clear()
            return medconnect_pb2.ResetResponse(status="reset successful")
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return medconnect_pb2.ResetResponse(status="error")

    def Translate(self, request, context):
        try:
            text = main.client_manager.translate_text(
                request.message or "",
                request.source_language or "english",
                request.target_language or "english",
            )
            return medconnect_pb2.TranslateResponse(message=text)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return medconnect_pb2.TranslateResponse(message="")


def serve():
    port = os.environ.get("GRPC_PORT", "50052")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    medconnect_pb2_grpc.add_MedConnectServiceServicer_to_server(
        MedConnectServiceServicer(), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("MedConnect gRPC server started on port %s", port)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()

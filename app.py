# streamlit_demo.py
import streamlit as st
import requests
import base64
from datetime import datetime

# ========================
# CONFIGURATION
# ========================
st.set_page_config(
    page_title="MedConnect AI - Remote Medical Assistant",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 MedConnect AI")
st.markdown("*Your intelligent medical assistant – connecting you to the right doctor, in your language.*")

API_URL = "https://medconnect-ai-4fnj.onrender.com/conversation"
RESET_URL = "https://medconnect-ai-4fnj.onrender.com/reset"

# Supported languages
LANGUAGES = {
    "English": "english",
    "Hausa": "hausa",
    "Yoruba": "yoruba",
    "Igbo": "igbo"
}

# ========================
# SESSION STATE
# ========================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_active" not in st.session_state:
    st.session_state.conversation_active = True

# ========================
# SIDEBAR CONTROLS
# ========================
with st.sidebar:
    st.header("⚙️ Settings")
    
    selected_lang_name = st.selectbox("Language", options=list(LANGUAGES.keys()), index=0)
    language = LANGUAGES[selected_lang_name]
    
    premium = st.toggle("Premium User", value=False)
    
    st.markdown("---")
    if st.button("🗑️ New Conversation"):
        requests.post(RESET_URL)
        st.session_state.messages = []
        st.success("Conversation reset!")
        st.rerun()

    st.markdown("---")
    st.caption("MedConnect AI V3 • LangGraph Multi-Agent System")

# ========================
# CHAT INTERFACE
# ========================
# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"])
            
            # Show doctor match if available
            if message.get("doctorid"):
                st.success(f"✅ Connected to Doctor ID: **{message['doctorid']}**")
            if message.get("medical_summary"):
                with st.expander("📋 View Medical Summary (SOAP Note)"):
                    st.text(message["medical_summary"])

# User input
if prompt := st.chat_input("Describe your symptoms or ask a medical question..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare payload
    payload = {
        "message": prompt,
        "audio": "",  # No audio in Streamlit demo
        "premium": premium,
        "language": language
    }
    
    # Show thinking spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    agent_message = data.get("message", "Sorry, I didn't understand that.")
                    doctorid = data.get("doctorid", "")
                    medical_summary = data.get("medical_summary", "")

                    st.markdown(agent_message)

                    # Store assistant message
                    msg_data = {
                        "role": "assistant",
                        "content": agent_message,
                        "doctorid": doctorid,
                        "medical_summary": medical_summary
                    }
                    st.session_state.messages.append(msg_data)

                    # Highlight doctor match
                    if doctorid:
                        st.success(f"✅ You have been matched with Doctor ID: **{doctorid}**")
                        st.balloons()

                    if medical_summary:
                        with st.expander("📋 View Generated SOAP Note"):
                            st.text(medical_summary)

                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("🚨 Cannot connect to the agent server. Make sure `uvicorn main:app --reload` is running.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# ========================
# FOOTER
# ========================
st.markdown("---")
st.caption("Powered by Gemini Flash • Fine-tuned MedConnectAI • LangGraph • Built for African Healthcare 🌍")
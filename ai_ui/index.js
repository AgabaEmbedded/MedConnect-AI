
import React, { useState } from "react";
import StyledButton from "@integratedComponents/StyledButton";
import OutlinedTextArea from "@commonComponents/OutlinedTextArea";
import { MedConnectService } from "./medconnect_pb_service";
import "./style.css";

const LANGUAGES = ["english", "hausa", "yoruba", "igbo"];

const MedConnect_UI = ({ serviceClient, isComplete }) => {
  const [healthMessage, setHealthMessage] = useState(null);
  const [conversationMessage, setConversationMessage] = useState("");
  const [conversationLanguage, setConversationLanguage] = useState("english");
  const [conversationReply, setConversationReply] = useState(null);
  const [conversationError, setConversationError] = useState(null);
  const [conversationLoading, setConversationLoading] = useState(false);
  const [translateSource, setTranslateSource] = useState("english");
  const [translateTarget, setTranslateTarget] = useState("hausa");
  const [translateInput, setTranslateInput] = useState("Hello");
  const [translateOutput, setTranslateOutput] = useState(null);
  const [translateLoading, setTranslateLoading] = useState(false);
  const [resetStatus, setResetStatus] = useState(null);

  const onHealthEnd = (response) => {
    const { message, status, statusMessage } = response;
    if (status !== 0) {
      setHealthMessage("Health check failed: " + (statusMessage || "unknown"));
      return;
    }
    setHealthMessage(message && message.getMessage ? message.getMessage() : (message && message.getStatus ? message.getStatus() : "OK"));
  };

  const runHealthCheck = () => {
    setHealthMessage(null);
    const methodDescriptor = MedConnectService.HealthCheck;
    const request = new methodDescriptor.requestType();
    serviceClient.unary(methodDescriptor, {
      request,
      preventCloseServiceOnEnd: false,
      onEnd: onHealthEnd
    });
  };

  const onConversationEnd = (response) => {
    const { message, status, statusMessage } = response;
    setConversationLoading(false);
    if (status !== 0) {
      setConversationError(statusMessage || "Request failed");
      setConversationReply(null);
      return;
    }
    setConversationError(null);
    setConversationReply(message);
  };

  const sendConversation = () => {
    setConversationLoading(true);
    setConversationError(null);
    setConversationReply(null);
    const methodDescriptor = MedConnectService.Conversation;
    const request = new methodDescriptor.requestType();
    request.setMessage(conversationMessage || "");
    request.setLanguage(conversationLanguage || "english");
    request.setPremium(false);
    serviceClient.unary(methodDescriptor, {
      request,
      preventCloseServiceOnEnd: false,
      onEnd: onConversationEnd
    });
  };

  const onTranslateEnd = (response) => {
    const { message, status, statusMessage } = response;
    setTranslateLoading(false);
    if (status !== 0) {
      setTranslateOutput("Error: " + (statusMessage || "Request failed"));
      return;
    }
    setTranslateOutput(message && message.getMessage ? message.getMessage() : "");
  };

  const runTranslate = () => {
    setTranslateLoading(true);
    setTranslateOutput(null);
    const methodDescriptor = MedConnectService.Translate;
    const request = new methodDescriptor.requestType();
    request.setMessage(translateInput || "");
    request.setSourceLanguage(translateSource || "english");
    request.setTargetLanguage(translateTarget || "hausa");
    serviceClient.unary(methodDescriptor, {
      request,
      preventCloseServiceOnEnd: false,
      onEnd: onTranslateEnd
    });
  };

  const onResetEnd = (response) => {
    const { message, status, statusMessage } = response;
    if (status !== 0) {
      setResetStatus("Reset failed: " + (statusMessage || "unknown"));
      return;
    }
    setResetStatus(message && message.getStatus ? message.getStatus() : "OK");
    setConversationReply(null);
    setConversationError(null);
  };

  const runReset = () => {
    setResetStatus(null);
    const methodDescriptor = MedConnectService.Reset;
    const request = new methodDescriptor.requestType();
    serviceClient.unary(methodDescriptor, {
      request,
      preventCloseServiceOnEnd: false,
      onEnd: onResetEnd
    });
  };

  return (
    <div className="medconnect-container">
      <section className="service-group">
        <div className="input-block input-block--compact">
          <h4>Health</h4>
          <div className="health-row">
            <StyledButton btnText="Ping" variant="outlined" onClick={runHealthCheck} />
            {healthMessage != null && (
              <span className={"health-status " + (healthMessage.toLowerCase().includes("failed") ? "health-status--error" : "health-status--ok")}>
                <span className="health-dot" />
                {healthMessage}
              </span>
            )}
          </div>
        </div>
        <div className="input-block input-block--compact">
          <h4>Reset</h4>
          <p className="hint">Reset conversation state.</p>
          <div className="health-row">
            <StyledButton btnText="Reset" variant="outlined" onClick={runReset} />
            {resetStatus != null && (
              <span className={"health-status " + ((resetStatus + "").toLowerCase().includes("fail") ? "health-status--error" : "health-status--ok")}>
                <span className="health-dot" />
                {resetStatus}
              </span>
            )}
          </div>
        </div>
      </section>

      <div className="input-block input-block--hero">
        <h4>Conversation</h4>
        <p className="hint">Send a message to the MedConnect assistant. Choose language.</p>
        <OutlinedTextArea
          value={conversationMessage}
          onChange={(e) => setConversationMessage(e.target.value)}
          placeholder="e.g. I have a headache and fever"
          rows={3}
        />
        <div className="row">
          <label>Language:</label>
          <select value={conversationLanguage} onChange={(e) => setConversationLanguage(e.target.value)}>
            {LANGUAGES.map((l) => (
              <option key={l} value={l}>{l}</option>
            ))}
          </select>
        </div>
        <div className="cta-row">
          <StyledButton
            btnText={conversationLoading ? "Sending…" : "Send"}
            variant="contained"
            onClick={sendConversation}
            disabled={conversationLoading}
          />
        </div>
        {conversationError && <div className="error-msg">{conversationError}</div>}
        {conversationReply && (
          <div className="reply-bubble">
            <p className="reply-text">{conversationReply.getMessage ? conversationReply.getMessage() : ""}</p>
            {conversationReply.getMedicalSummary && conversationReply.getMedicalSummary() && (
              <p className="summary">Summary: {conversationReply.getMedicalSummary()}</p>
            )}
            {conversationReply.getDoctorId && conversationReply.getDoctorId() && (
              <p className="doctor">Doctor ID: {conversationReply.getDoctorId()}</p>
            )}
          </div>
        )}
      </div>

      <div className="input-block">
        <h4>Translate</h4>
        <p className="hint">Translate text between English, Hausa, Yoruba, and Igbo.</p>
        <div className="row">
          <input type="text" value={translateInput} onChange={(e) => setTranslateInput(e.target.value)} placeholder="Text to translate" className="translate-input" />
        </div>
        <div className="row row--gap">
          <label>From:</label>
          <select value={translateSource} onChange={(e) => setTranslateSource(e.target.value)}>
            {LANGUAGES.map((l) => (
              <option key={l} value={l}>{l}</option>
            ))}
          </select>
          <label>To:</label>
          <select value={translateTarget} onChange={(e) => setTranslateTarget(e.target.value)}>
            {LANGUAGES.map((l) => (
              <option key={l} value={l}>{l}</option>
            ))}
          </select>
        </div>
        <div className="cta-row">
          <StyledButton btnText={translateLoading ? "Translating…" : "Translate"} variant="outlined" onClick={runTranslate} disabled={translateLoading} />
        </div>
        {translateOutput != null && (
          <div className="translate-result">
            <span className="translate-result-label">Translated ({translateTarget})</span>
            <p className="translate-result-text">{translateOutput}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default MedConnect_UI;

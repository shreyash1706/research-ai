import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Research Agent", page_icon="🔬", layout="wide")

# --- INITIALIZATION ---
if "current_session_id" not in st.session_state:
    # Ask backend to create a new session in DB
    res = requests.post(f"{API_BASE}/sessions").json()
    st.session_state.current_session_id = res["session_id"]
    st.session_state.messages = []

current_sid = st.session_state.current_session_id

# --- SIDEBAR ---
with st.sidebar:
    st.header("💬 Chat History")
    
    if st.button("➕ New Chat", use_container_width=True):
        res = requests.post(f"{API_BASE}/sessions").json()
        st.session_state.current_session_id = res["session_id"]
        st.session_state.messages = []
        st.rerun()

    st.divider()
    
    # Fetch all sessions from backend
    try:
        all_sessions = requests.get(f"{API_BASE}/sessions").json()
        for idx, session in enumerate(all_sessions):
            friendly_name = f"Chat {len(all_sessions) - idx} ({session['date'][:16]})"
            
            if st.button(friendly_name, key=session["id"], use_container_width=True):
                st.session_state.current_session_id = session["id"]
                # Fetch history for this specific chat from backend
                st.session_state.messages = requests.get(f"{API_BASE}/chat/{session['id']}").json()
                st.rerun()
    except requests.exceptions.ConnectionError:
        st.error("Backend offline.")

# --- MAIN UI ---
st.title("🔬 Local Research Agent")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about a machine learning paper..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Agent is thinking..."):
            # We ONLY send the prompt now! Backend handles the history.
            response = requests.post(
                f"{API_BASE}/chat/{current_sid}",
                json={"prompt": prompt}, 
                stream=True
            )
            
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
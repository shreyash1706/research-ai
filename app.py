import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="Research Agent", page_icon="🔬")
st.title("🔬 Local Research Agent")

# Initialize session state for memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input Block
if prompt := st.chat_input("Ask about a machine learning paper..."):
    # 1. Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Setup Assistant UI
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # We don't send the system prompt from here, the backend handles it.
            # We just send the conversational history.
            with st.spinner("Agent is thinking..."):
                response = requests.post(
                    API_URL,
                    json={"messages": st.session_state.messages},
                    stream=True
                )
                response.raise_for_status()
                
            # Stream the text into the UI
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            
            # Finalize rendering
            message_placeholder.markdown(full_response)
            
        except requests.exceptions.ConnectionError:
            st.error("Backend connection failed. Is FastAPI running on port 8000?")
            st.stop()
            
    # 3. Save assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
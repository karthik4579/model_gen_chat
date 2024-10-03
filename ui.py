from utils import generate_chat_response
import streamlit as st

st.set_page_config(page_title="Ketu | LLM Finetuning Assistant", page_icon="🤖")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.chat_message("assistant"):
    message_placeholder = st.empty()
    response = generate_chat_response(st.session_state.messages)
    message_placeholder.markdown(response)
    
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = generate_chat_response(st.session_state.messages)
        message_placeholder.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
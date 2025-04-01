import warnings
import pandas as pd
import streamlit as st
import plotly.express as px  # Ensure this import matches the actual file where ChatBot is defined
from ..test import ChatBot  # Update with actual path
# Initialize the ChatBot
bot = ChatBot()

# Set Streamlit app configuration
st.set_page_config(page_title="BRTS Navigation Bot")

# Sidebar design
with st.sidebar:
    st.image("assets/img/omdena.png", use_column_width=True)  # Update with actual path
    st.title('BRTS Navigation Bot')
    st.write("Your personal assistant for navigating the BRTS system.")

# Function for generating LLM response
def generate_response(input):
    result = bot.ask(input)  # Adjusted to use bot.ask
    return result

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! How can I assist you with BRTS navigation today?"}]

# Main function for the Streamlit app
def main(chatbot):
    chatbot.write("""<h3><i class="fa-regular fa-message"></i>&nbsp;&nbsp;Chatbot</h3>""", unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        role_class = "assistant" if message["role"] == "assistant" else "user"
        with chatbot.container():
            chatbot.markdown(f"""
                <div class='message-container'>
                    <span class='{role_class}'>{message["role"]}: </span>{message["content"]}
                </div>
            """, unsafe_allow_html=True)
    
    # User-provided prompt
    if input := chatbot.text_input("Your message: "):
        st.session_state.messages.append({"role": "user", "content": input})
        with chatbot.container():
            chatbot.markdown(f"""
                <div class='message-container'>
                    <span class='user'>user: </span>{input}
                </div>
            """, unsafe_allow_html=True)
        
        # Generate a new response if last message is not from assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with chatbot.spinner("Finding the best route for you..."):
                response = generate_response(input)
                chatbot.markdown(f"""
                    <div class='message-container'>
                        <span class='assistant'>assistant: </span>{response}
                    </div>
                """, unsafe_allow_html=True)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)

# Apply custom styling
st.markdown("""
    <style>
        .message-container {
            background: #f0f0f0;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
        }
        .assistant {
            color: #0366d6;
            font-weight: bold;
        }
        .user {
            color: #d73a49;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

if __name__ == '__main__':
    main(st)

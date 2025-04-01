import streamlit as st
from streamlit_lottie import st_lottie
from constants import PROJECT_DESC

def main(home):    
    home.write("<br>"*1, unsafe_allow_html=True)
    
    cols = home.columns([2,0.1,1.2,0.1])
    with cols[0]:
        st.write("<br>"*3, unsafe_allow_html=True)
        
        with st.columns(3)[1]:
            st.image("assets/img/omdena.png", use_column_width=True)
            
        st.write("<br>", unsafe_allow_html=True)
        st.write("""
            <center>
                <h1 class='poppins-light'>Developing an AI Chatbot for Bhopal's Bus Rapid Transit System (BRTS) Navigation</h1>
            </center>
        """, unsafe_allow_html=True)
        
    with cols[-2]:
        st_lottie("https://lottie.host/ba9653f3-b074-4514-81d5-234c057dba21/eOERHmJJ4W.json")

    home.markdown(PROJECT_DESC, unsafe_allow_html=True)

if __name__ == "__main__":
    main(home)
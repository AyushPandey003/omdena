import streamlit as st

def main(team):
    # team.write("<h3>👥 Team</h3>", unsafe_allow_html=True)
    team.write("<br>", unsafe_allow_html=True)
    team.write("""
    <center>
        <h2>Meet the Team</h2>
    </center>
    """, unsafe_allow_html=True)
    team.write("<br>", unsafe_allow_html=True)

    with team.container():
        st.write("<br>"*2, unsafe_allow_html=True)
        columns = st.columns(3)
        for col in columns:
            with col:
                st.write(f"<center><p>{'John Doe<br>'*10}</p></center>", unsafe_allow_html=True)
        st.write("<br>"*2, unsafe_allow_html=True)
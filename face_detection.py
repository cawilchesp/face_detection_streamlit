import streamlit as st


def main():
    # setting page layout
    st.set_page_config(
        page_title="Interactive Interface for Face Detection",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Main Page ------------------
    st.title('Face Detection')
    

if __name__ == '__main__':
    main()
                
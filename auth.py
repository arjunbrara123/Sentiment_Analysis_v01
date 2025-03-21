"""
Authentication Module (auth.py)
================================

Purpose:
--------
This module manages user authentication for the Competitor Intelligence Dashboard.
It displays a secure login form using Streamlit and verifies credentials via SHA-256 hashing.
Only authenticated users are allowed to access the dashboard.

Function Overview:
------------------
1. hash_password: Accepts a plain-text password and returns its SHA-256 hash.
2. login_form: Displays the login form, validates user credentials, updates session state upon success, and halts execution if authentication fails.

How It Works:
-------------
- Compares the entered username and hashed password against pre-configured valid credentials.
- Updates the session state to allow access if credentials are correct; otherwise, stops further execution.
"""

import hashlib
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# --- Configuration for authentication ---
VALID_USERNAME = "123"
VALID_HASHED_PASSWORD = "a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3"

def hash_password(password: str) -> str:
    # Hashes the given password using SHA-256.
    return hashlib.sha256(password.encode()).hexdigest()

def login_form() -> bool:
    # Purpose: Displays a login form and authenticates the user.
    # Returns: True if authenticated, otherwise stops execution.

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if st.session_state["authenticated"]:
        return True

    # Layout for login screen: display logos and the form in the center column
    col1, col2, col3, col4, col5 = st.columns([3,1,3,1,3])
    with col1:
        st.image("images/bgil_competitors_transparent.png")
    with col5:
        st.write("")
        st.write("")
        st.write("")
        st.image("images/ai_logo_transparent.png")
    with col3:
        st.image("images/bgil-alice-logo1.png")
        with st.form("login_form"):
            username = st.text_input(
                "👤 **Username**",
                placeholder="Your Username Here...",
                help="💡 Please enter your username here. If you do not have one, please contact the underwriting team for access to this tool."
            )
            password = st.text_input(
                "🤐 **Password**",
                type="password",
                placeholder="Your Password Here...",
                help="💡 If you have forgotten your password, please contact the underwriting team to reset this."
            )
            submitted = st.form_submit_button(" 🔐 Login")

        if submitted:
            if username == VALID_USERNAME and hash_password(password) == VALID_HASHED_PASSWORD:
                st.session_state["authenticated"] = True
                st.success("Login successful!")
                st.rerun()  # Refresh to remove the login form
            else:
                st.error("Incorrect username or password")
                st.stop()

    st.stop()  # Stop execution until login is complete
    return False

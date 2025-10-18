import streamlit as st
import smtplib
import random
import time
import requests
from urllib.parse import urlencode

# --- CONFIG ---
st.set_page_config(page_title="2FA Login", page_icon="🔐")

# --- STATE INIT ---
if "step" not in st.session_state:
    st.session_state.step = "login"
if "otp" not in st.session_state:
    st.session_state.otp = None
if "otp_time" not in st.session_state:
    st.session_state.otp_time = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None

# --- EMAIL FUNCTION ---
def send_email(to_email, otp):
    try:
        email_address = st.secrets["email_address"]
        email_password = st.secrets["email_password"]

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(email_address, email_password)
            message = f"Subject: Your 2FA OTP Code\n\nYour OTP is: {otp}"
            server.sendmail(email_address, to_email, message)
        return True
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False

# --- GITHUB OAUTH CONFIG ---
GITHUB_CLIENT_ID = st.secrets.get("github_client_id", "")
GITHUB_CLIENT_SECRET = st.secrets.get("github_client_secret", "")
APP_BASE_URL = st.secrets.get("app_base_url", "http://localhost:8501")
REDIRECT_URI = APP_BASE_URL.rstrip("/") + "/.auth"

def get_github_auth_url():
    params = {
        "client_id": GITHUB_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": "user:email",
    }
    return "https://github.com/login/oauth/authorize?" + urlencode(params)

def exchange_github_code_for_token(code):
    resp = requests.post(
        "https://github.com/login/oauth/access_token",
        data={
            "client_id": GITHUB_CLIENT_ID,
            "client_secret": GITHUB_CLIENT_SECRET,
            "code": code,
            "redirect_uri": REDIRECT_URI,
        },
        headers={"Accept": "application/json"},
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()

def fetch_github_profile(access_token):
    headers = {"Authorization": f"token {access_token}"}
    user = requests.get("https://api.github.com/user", headers=headers, timeout=10).json()
    emails = requests.get("https://api.github.com/user/emails", headers=headers, timeout=10).json()
    if isinstance(emails, list) and len(emails) > 0:
        for e in emails:
            if e.get("primary"):
                user["email"] = e["email"]
                break
        if "email" not in user:
            user["email"] = emails[0].get("email")
    return user

# --- HANDLE GITHUB CALLBACK ---
params = st.query_params()
if "code" in params:
    code = params["code"][0]
    try:
        token = exchange_github_code_for_token(code)
        access_token = token.get("access_token")
        if access_token:
            profile = fetch_github_profile(access_token)
            st.session_state.step = "dashboard"
            st.session_state.user_email = profile.get("email", profile.get("login", "unknown"))
            st.query_params()  # clear URL params
            st.rerun()
    except Exception as e:
        st.error(f"GitHub login failed: {e}")

# --- LOGIN PAGE ---
if st.session_state.step == "login":
    st.title("🔐 Secure Login System")
    st.subheader("Choose your login method")

    tab1, tab2 = st.tabs(["📧 Email OTP", "🐙 GitHub OAuth"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        user_email = st.text_input("Your Email")

        if st.button("Login with Email"):
            if username == "swagato" and password == "test123":
                if not user_email or "@" not in user_email:
                    st.warning("⚠️ Please enter a valid email address.")
                else:
                    otp = random.randint(100000, 999999)
                    st.session_state.otp = str(otp)
                    st.session_state.otp_time = time.time()
                    st.session_state.user_email = user_email

                    if send_email(user_email, otp):
                        st.success(f"✅ OTP sent to {user_email}. Check your inbox.")
                        st.session_state.step = "verify"
            else:
                st.error("Invalid username or password ❌")

    with tab2:
        st.write("Login using your GitHub account:")
        if st.button("Login with GitHub"):
            if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
                st.error("GitHub OAuth not configured in secrets.")
            else:
                auth_url = get_github_auth_url()
                st.markdown(f"[Click here to authorize via GitHub]({auth_url})")

# --- OTP VERIFY ---
elif st.session_state.step == "verify":
    st.title("📩 Verify Your OTP")
    otp_input = st.text_input("Enter the OTP sent to your email")
    if st.button("Verify"):
        if time.time() - st.session_state.otp_time > 180:
            st.warning("⚠️ OTP expired. Please login again.")
            st.session_state.step = "login"
        elif otp_input == st.session_state.otp:
            st.success("✅ Login successful!")
            st.session_state.step = "dashboard"
        else:
            st.error("❌ Invalid OTP. Try again.")

# --- DASHBOARD ---
elif st.session_state.step == "dashboard":
    st.title("🎉 Welcome to Your Dashboard!")
    st.write(f"You are securely logged in as **{st.session_state.user_email}**")
    if st.button("Logout"):
        for key in ["step", "otp", "otp_time", "user_email"]:
            st.session_state[key] = None
        st.session_state.step = "login"
        st.rerun()

import streamlit as st
import smtplib
import random
import time

# --- CONFIG ---
st.set_page_config(page_title="2FA Login", page_icon="🔐")

# --- STATE INIT ---
if "step" not in st.session_state:
    st.session_state.step = "login"
if "otp" not in st.session_state:
    st.session_state.otp = None
if "otp_time" not in st.session_state:
    st.session_state.otp_time = None

# --- EMAIL SENDER FUNCTION ---
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


# --- STEP 1: LOGIN ---
if st.session_state.step == "login":
    st.title("🔐 Secure Login with Email-based 2FA")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Dummy credentials (replace with DB validation)
        if username == "swagato" and password == "test123":
            otp = random.randint(100000, 999999)
            st.session_state.otp = str(otp)
            st.session_state.otp_time = time.time()

            # replace this with the user's real email
            recipient_email = "duttpriya70@gmail.com"

            if send_email(recipient_email, otp):
                st.success("✅ OTP sent to your email. Check your inbox.")
                st.session_state.step = "verify"
        else:
            st.error("Invalid username or password ❌")

# --- STEP 2: OTP VERIFY ---
elif st.session_state.step == "verify":
    st.title("📩 Verify Your OTP")
    otp_input = st.text_input("Enter the OTP sent to your email")

    if st.button("Verify"):
        if time.time() - st.session_state.otp_time > 180:  # 3 min expiry
            st.warning("⚠️ OTP expired. Please login again.")
            st.session_state.step = "login"
        elif otp_input == st.session_state.otp:
            st.success("✅ Login successful!")
            st.session_state.step = "dashboard"
        else:
            st.error("❌ Invalid OTP. Try again.")

# --- STEP 3: DASHBOARD ---
elif st.session_state.step == "dashboard":
    st.title("🎉 Welcome to Your Dashboard!")
    st.write("You are now securely logged in using 2FA via email.")
    if st.button("Logout"):
        st.session_state.step = "login"
        st.session_state.otp = None
        st.session_state.otp_time = None

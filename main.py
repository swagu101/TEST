import streamlit as st
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="Profile Info Extractor", page_icon="👤")

st.title("👤 LinkedIn & GitHub Profile Info Extractor")
st.write("Paste a profile link to extract basic information")

url = st.text_input("Paste GitHub or LinkedIn profile link here")

# -------------------------
# GitHub Extraction
# -------------------------
def extract_github_info(username):
    api_url = f"https://api.github.com/users/{username}"
    response = requests.get(api_url)

    if response.status_code != 200:
        return None

    data = response.json()

    return {
        "Name": data.get("name"),
        "Username": data.get("login"),
        "Bio": data.get("bio"),
        "Followers": data.get("followers"),
        "Following": data.get("following"),
        "Public Repos": data.get("public_repos"),
        "Profile URL": data.get("html_url"),
        "Location": data.get("location")
    }

# -------------------------
# LinkedIn Extraction (basic)
# -------------------------
def extract_linkedin_info(profile_url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(profile_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    name = soup.find("h1")
    headline = soup.find("div", class_="text-body-medium")

    return {
        "Name": name.get_text(strip=True) if name else "Not Found",
        "Headline": headline.get_text(strip=True) if headline else "Not Found",
        "Profile URL": profile_url
    }

# -------------------------
# Button click
# -------------------------
if st.button("Extract"):

    if "github.com" in url:
        username = url.rstrip("/").split("/")[-1]
        data = extract_github_info(username)

        if data:
            st.success("✅ GitHub Profile Extracted")

            for key, value in data.items():
                st.write(f"**{key}:** {value}")
        else:
            st.error("❌ Invalid GitHub profile or API limit reached")

    elif "linkedin.com" in url:
        data = extract_linkedin_info(url)

        st.success("✅ LinkedIn Profile Extracted")

        for key, value in data.items():
            st.write(f"**{key}:** {value}")

    else:
        st.warning("❗ Please enter a valid GitHub or LinkedIn URL")

print("loading utils.py")

#import libraries
import os
import ast
from bs4 import BeautifulSoup
import chardet
from collections import Counter
from collections import defaultdict

#!pip install google-play-scraper
from google_play_scraper import search, permissions, app

from itertools import chain
import joblib #save models
import json
import logging
import numpy as np
import openpyxl
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd
from pathlib import Path
import pickle #save jobs
import random
import re
import requests
from bs4 import BeautifulSoup
import shutil
import sys
import time
from tqdm import tqdm  # Import tqdm for the progress bar
from tqdm.auto import tqdm
tqdm.pandas()

#!pip install nltk
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils.class_weight import compute_class_weight

#!pip install joblib scikit-learn nltk --quiet
#!pip install transformers datasets scikit-learn tqdm joblib --quiet
#!pip install streamlit pyngrok
import streamlit as st
import requests
import time

import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def fetch_info(app_id):
    try:
        info = app(app_id)
        perm = permissions(app_id)
        return {
            "APP_ID": app_id,
            "App Name": info["title"],
            "Icon URL": info["icon"],
            "Category": info["genreId"],
            "Age Group": info["contentRating"],
            "Policy Link": info['privacyPolicy'],
            "Perm": perm
        }
    except Exception as e:
        print(f"❌ Error fetching info for {app_id}: {e}")
        return None



def search_apps_starting_with(query, n_results=10):
    query = query.strip()
    query = re.sub(r'\s{2,}', ' ', query)
    raw_results = search(query, lang='en', country='us')
    filtered = [r for r in raw_results if r["title"].lower().startswith(query.lower())]
    return filtered[:n_results]

#perm label
perm_label = ['CAMERA', 'MICROPHONE', 'PHONE_CALL', 'SENSOR', 'SMS',
              'CALENDAR', 'CONTACTS', 'LOCATION', 'STORAGE', 'PERSISTENTID']


#Perm normalize
perm_map = {
    'location': 'LOCATION',
    'photos/media/files': 'STORAGE',
    'storage': 'STORAGE',
    'camera': 'CAMERA',
    'microphone': 'MICROPHONE',
    #'phone': 'PHONE_CALL',
    'device id & call information': 'PERSISTENTID',
    'cellular data settings': 'PERSISTENTID',
    'persistentid': 'PERSISTENTID',
    #'device & app history': 'PERSISTENTID',
    'sms': 'SMS',
    'calendar': 'CALENDAR',
    'contacts': 'CONTACTS',
    'identity': 'CONTACTS',
    'sensors': 'SENSOR',
    'wearable sensors/activity data': 'SENSOR',
    'wi-fi connection information': ['LOCATION', 'PERSISTENTID'],  # for ACCESS_WIFI_STATE, ACCESS_NETWORK_STATE
    'uncategorized': None,
    'other': None
}

def map_permissions_list(raw_perms):
    mapped_perms = []

    if not isinstance(raw_perms, dict):
        return mapped_perms

    for cat in raw_perms:
        cat_lower = cat.lower()

        # Special case for 'other' + 'view network connection'
        if cat_lower == 'other':
            # Check if 'view network connection' is in the values for this key
            perms_in_other = raw_perms[cat]
            if any('view network connection' in p.lower() for p in perms_in_other):
                if 'PERSISTENTID' not in mapped_perms:
                    mapped_perms.append('PERSISTENTID')
                continue

        # Special case for 'phone' key with detailed checks
        if cat_lower == 'phone':
            perms_in_phone = raw_perms[cat]
            for p in perms_in_phone:
                p_lower = p.lower()
                if 'phone status' in p_lower:
                    if 'PERSISTENTID' not in mapped_perms:
                        mapped_perms.append('PERSISTENTID')
                if 'phone number' in p_lower or 'call phone' in p_lower:
                    if 'PHONE_CALL' not in mapped_perms:
                        mapped_perms.append('PHONE_CALL')
            continue

        std_cat = perm_map.get(cat_lower)

        if isinstance(std_cat, list):
            for item in std_cat:
                if item not in mapped_perms:
                    mapped_perms.append(item)
        elif isinstance(std_cat, str):
            if std_cat not in mapped_perms:
                mapped_perms.append(std_cat)

    return mapped_perms

# risk summary
# Define risk descriptions
risk_map = {
    "CAMERA": "🎥 CAMERA: Could record your child’s face or surroundings.",
    "MICROPHONE": "🎙️ MICROPHONE: Could listen to your child talk or play.",
    "LOCATION": "📍 LOCATION: Can follow your child’s location.",
    "CONTACTS": "👥 CONTACTS: Could share your child’s contacts with others.",
    "STORAGE": "📁 STORAGE: Could save or read your child’s photos or downloads.",
    "PHONE_CALL": "📞 PHONE CALL: Could make calls without your permission.",
    "SMS": "💬 SMS: Could send or read text messages without your knowledge.",
    "CALENDAR": "🗓️ CALENDAR: May access your child’s schedule or reminders.",
    "SENSOR": "🧭 SENSOR: May track your child's physical movement or activity.",
    "PERSISTENTID": "🆔 DEVICE ID: Could track your child across apps or over time."
}

def generate_risk_summary(permissions_used):
    if not permissions_used:
        return "No permissions used."

    risks = []
    for perm in permissions_used:
        perm_upper = perm.upper()
        if perm_upper in risk_map:
            risks.append(risk_map[perm_upper])

    return "\n".join(risks) if risks else "No risky permissions found."


#fetch policy text

def fetch_policy_text(url):
    # Try standard fetch first
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        if len(response.text) > 1000:  # crude quality check
            print("✅ Standard fetch succeeded.")
            return response.text
        else:
            print("⚠️ Response too short. Trying Playwright fallback...")
    except Exception as e:
        print(f"❌ Standard fetch failed: {e}")

    # Playwright fallback
    try:
        from playwright.sync_api import sync_playwright
        # from playwright_stealth import stealth_sync  # Optional: uncomment if you really need stealth
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            # stealth_sync(page)  # Optional: use if site uses bot detection
            page.goto(url, wait_until="networkidle", timeout=30000)
            time.sleep(1)  # Let JS finish loading
            html = page.content()
            browser.close()
            print("✅ Playwright fetch succeeded.")
            return html
    except Exception as e:
        print(f"⚠️ Playwright fetch failed: {e}")
        return None


# Preprocess text
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=' ', strip=True)
    return text


def extract_segments(html):
    soup = BeautifulSoup(html, "html.parser")

    # Remove unwanted elements
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Insert newline after block-level elements (to imitate paragraph breaks)
    for tag in soup.find_all(["p", "li", "br", "div", "section", "h1", "h2", "h3", "h4"]):
        tag.insert_after("\n")

    # Get text
    text = soup.get_text()

    # Now split into segments
    segments = [seg.strip() for seg in text.split("\n") if seg.strip()]
    return segments


def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(cleaned_tokens)


# Keyword - rule based for mismatch
# --- Helper to check if keywords appear ---
def keyword_check(policy_text, keywords):
    if not isinstance(policy_text, str):
        return False
    policy_text = policy_text.lower()
    return any(keyword in policy_text for keyword in keywords)

# --- Check if permissions are mentioned in policy ---
def check_permissions(row):
    policy_text = row['CleanText']
    permissions_used = row['Permissions Used']

    mismatches = []
    found_perms = []

    for perm in permissions_used:
        keywords = final_keywords.get(perm, [])
        if keyword_check(policy_text, keywords):
            found_perms.append(perm)

    # If nothing found, then all are mismatches
    if not found_perms:
        mismatches = permissions_used
    else:
        mismatches = [perm for perm in permissions_used if perm not in found_perms]

    return pd.Series({
        "Permissions Found": found_perms,
        "Policy Mismatches": mismatches
    })


# --- Count matched keyword frequencies and phrases ---
def find_keywords_and_counts(policy_text, keywords):
    if not isinstance(policy_text, str):
        return 0, {}

    policy_text = policy_text.lower()
    match_counter = Counter()

    for kw in keywords:
        kw_clean = kw.lower().strip()
        matches = re.findall(re.escape(kw_clean), policy_text)
        if matches:
            match_counter[kw_clean] += len(matches)

    total_count = sum(match_counter.values())
    return total_count, dict(match_counter)

# --- Match permissions to keyword frequencies ---
with open('PARENT_App/data/final_keywords.json', 'r', encoding='utf-8') as f:
    final_keywords = json.load(f)

def extract_permission_keyword_matches(row):
    policy_text = row['CleanText']
    permissions_used = row['Permissions Used']

    freq_dict = {}
    match_dict = {}
    for perm in permissions_used:
        keywords = final_keywords.get(perm, [])
        freq, matches = find_keywords_and_counts(policy_text, keywords)
        freq_dict[perm] = freq
        match_dict[perm] = matches

    return pd.Series({'Keyword Frequencies': freq_dict, 'Keyword Matches': match_dict})



# Policy verdict
def classify_app_risk(row):
    clean_text = row.get('CleanText', '')
    if len(clean_text.split()) < 10:
        # Verdict for short policy text
        verdict = "⚠️ Policy text too short for analysis; results may be incomplete."
        recommendation = (
            "Consider visiting the app's website for a more detailed privacy policy."
        )
        overview = "Extracted policy text length too short to perform analysis."
        legal_issues = []
        return pd.Series({
            "Verdict": verdict,
            "Legal Concerns": legal_issues,
            "Recommendations": recommendation,
            "Overview": overview
        })
    
    policy_mismatches = row['Policy Mismatches']

    legal_flags = {
        "CAMERA": "This app can use the camera, but doesn’t clearly explain why. Children’s images need strong protection.",
        "MICROPHONE": "The app might listen through the microphone, but gives no clear reason. This could be risky for younger users.",
        "LOCATION": "The app can see your child’s location, but the privacy policy doesn’t clearly explain why. This might go against child safety guidelines.",
        "CONTACTS": "The app asks to read contacts, but it doesn’t say how it uses them. Apps should explain how they use personal info like names or phone numbers.",
        "PERSISTENTID": "The app might track your child over time but doesn’t say why. This may not follow GDPR rules for profiling children (Art. 5, 8).",
        "STORAGE": "The app can access your child’s files or photos but doesn’t explain the reason. This might breach UK child data protection.",
        "PHONE_CALL": "This app can make calls but doesn’t explain why. Apps aimed at kids should be clear about this.",
        "SMS": "This app may read or send texts without explaining why. That’s a privacy risk, especially for children.",
        "CALENDAR": "This app can read your child’s schedule or reminders, but doesn’t say how it uses that information.",
        "SENSOR": "The app can monitor motion or activity but doesn't say why. That might go against privacy principles."
    }
    
    legal_issues = [legal_flags[perm] for perm in policy_mismatches if perm in legal_flags]


    # Generate friendly explanation
    if policy_mismatches:
        if len(policy_mismatches) == 1:
            overview = (
                f"This app requests access to {policy_mismatches[0].lower()}, "
                "but doesn’t provide a clear explanation for why it needs this permission.\n "
                " This lack of clarity could raise concerns, especially for children's data."
            )
        else:
            perms_str = ", ".join([perm.lower() for perm in policy_mismatches[:-1]]) + " and " + policy_mismatches[-1].lower()
            overview = (
                f"This app requests access to several features, such as {perms_str}, "
                "but doesn’t provide clear explanations for why it needs these permissions.\n "
                " This lack of clarity could raise concerns, particularly for children's data."
            )
    else:
        overview = "This app's permissions are clearly explained in the policy."

    if len(policy_mismatches) >= 2:
        mismatch_str = (lambda items: items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1])([perm.lower() for perm in policy_mismatches])
        verdict = (
            f"❌ This app uses several permissions (like {mismatch_str}), "
            "but doesn’t explain how or why.\n"
            " That may not meet UK child privacy standards:\n"
            "- GDPR Article 5 – apps must be clear about data use\n"
            "- GDPR Article 8 – parental consent required for children's data\n"
            "- UK Age Appropriate Design Code – extra protections like prioritizing children’s privacy by default, "
            "only collecting what’s necessary, and using clear language for data use."
        )
        recommendation = (
            "- Try exploring other apps with clearer privacy policies.\n "
            "- Talk to your child about what information apps can access.\n "
            "- You can also edit app permissions in your device settings."
        )
    elif len(policy_mismatches) == 1:
        perm_name = policy_mismatches[0].lower()
        verdict = (
            f"⚠️ This app can access your child’s {perm_name}, but doesn’t clearly explain why.\n"
            " Children’s data needs strong protection under UK privacy laws:\n"
            "- GDPR Article 5 – apps must be clear about data use\n"
            "- GDPR Article 8 – parental consent is required for children's data\n"
            "- UK Age Appropriate Design Code – apps should prioritize children’s privacy by default, "
            "only collect what’s necessary, and use clear language about data use."
        )
        recommendation = (
            f"- Consider checking if the app is really needed.\n "
            f"- You can limit access to {perm_name} in your phone settings.\n "
            f"- Explain to your child why it's important to ask before giving apps permission."
        )
    else:
        verdict = (
            "✅ This app clearly explains how it uses your child’s data.\n "
            " That’s a good sign — it supports your child’s privacy and follows safety rules like the UK GDPR (Article 5 & 8) and the Children’s Code."
        )
        recommendation = (
            "- Encourage your child to ask questions whenever they give an app permission to access their data.\n "
            "- Teach them to be cautious and help them say ‘No’ to unnecessary requests.\n"
            "- Consider reviewing app permissions regularly and discussing with your child how apps use their information."
        )

    return pd.Series({
        "Verdict": verdict,
        "Legal Concerns": legal_issues,
        "Recommendations": recommendation,
        "Overview": overview
    })


# save
EXCEL_PATH = "PARENT_App/data/app_analysis_results.xlsx"


def check_if_app_exists(app_id_or_name, path=EXCEL_PATH):
    if not os.path.exists(path):
        return False
    df = pd.read_excel(path)
    return any(
        df["APP_ID"].astype(str).str.lower() == str(app_id_or_name).lower()
    )

def append_to_excel(new_df, path=EXCEL_PATH):
    if os.path.exists(path):
        old_df = pd.read_excel(path)
        combined_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    combined_df.to_excel(path, index=False)


print("utils.py loaded")

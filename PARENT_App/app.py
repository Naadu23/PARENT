print('loading app.py')

import streamlit as st
import pandas as pd
import time
import joblib
import os

from utils import (
    search_apps_starting_with,
    fetch_info,
    classify_app_risk,
    map_permissions_list,
    generate_risk_summary,
    fetch_policy_text,
    extract_segments,
    clean_html,
    clean_text,
    check_permissions,
    extract_permission_keyword_matches
)
from bert import get_tokenizer_and_models, process_policy_segments, predict_labels, summarize_predicted_labels_paragraph
from gdpr import run_gdpr_processing, fallback_extract_words_from_entities, build_summary, final_keyword,perm_label
from logistic import generate_sharing_summary
from ui import display_app_header, display_app_analysis
from tqdm.auto import tqdm

tqdm.pandas()

@st.cache_resource
def load_models():
    tokenizer, models = get_tokenizer_and_models()
    logistic_model = joblib.load("PARENT_App/best_logistic_pipeline.joblib")
    return tokenizer, models, logistic_model

bert_tokenizer, bert_models, model = load_models()

# Load Excel files from repo folder
EXCEL_PATH_ANALYSIS = "PARENT_App/data/app_analysis_results.xlsx"
EXCEL_PATH_SECONDARY = "PARENT_App/data/processed_output.xlsx"

# --- Caching Heavy Functions --- #
@st.cache_data(show_spinner=False)
def cached_fetch_info(app_id):
    return fetch_info(app_id)

@st.cache_data(show_spinner=False)
def cached_fetch_policy_text(url):
    return fetch_policy_text(url)

@st.cache_data(show_spinner=False)
def cached_extract_segments(text):
    return extract_segments(text)

@st.cache_data(show_spinner=False)
def cached_clean_policy_text(text):
    return clean_text(clean_html(text))

@st.cache_data(show_spinner=False)
def cached_process_policy_segments(app_df):
    return process_policy_segments(
        app_df=app_df,
        tokenizer=bert_tokenizer,
        predict_labels=predict_labels,
        summarize_predicted_labels_paragraph=summarize_predicted_labels_paragraph,
        min_tokens=10,
        max_segments=70
    )


# --- UI Layout --- #
st.markdown("### 📱PARENT : *Privacy App REview for Non-Technical users*")

st.markdown(
    '<p style="font-style:italic; color:#555; font-size:18px; margin-top:10px; margin-bottom:30px;">'
    '“Understand What Your Child’s Apps Are Really Doing — Know the Risks, Protect Their Clicks.”'
    '</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p style="margin-bottom:20px;">This app helps parents understand the privacy risks in apps their children and themselves use.</p>'
    '<p style="margin-bottom:20px;">It looks at app permissions and data use, then explains what the apps do in easy terms.</p>'
    '<p style="margin-bottom:30px;">This way, you can protect your child’s privacy, make safer choices, and keep their digital world secure without any tech confusion.</p>',
    unsafe_allow_html=True
)

col1, col2 = st.columns([2, 1])
with col1:
    st.write("")
with col2:
    if st.button("🔄 Clear & Search Again", help='Click to clear and restart search'):
        st.session_state.clear()
        st.rerun()

# --- Session State Initialization --- #
if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False
if "selected_app_id" not in st.session_state:
    st.session_state.selected_app_id = None
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "selected_option" not in st.session_state:
    st.session_state.selected_option = None
if "search_triggered" not in st.session_state:
    st.session_state.search_triggered = False
if "analysis_row" not in st.session_state:
    st.session_state.analysis_row = None

# --- Helpers --- #
def append_to_excel(df, path):
    try:
        existing_df = pd.read_excel(path)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_excel(path, index=False)
    except FileNotFoundError:
        df.to_excel(path, index=False)


@st.cache_data
def load_excel_if_exists(path):
    if os.path.exists(path):
        df = pd.read_excel(path)
        df = df.fillna("")
        return df
    return pd.DataFrame()
def app_exists_in_any_excel(app_id):
    df_analysis = load_excel_if_exists(EXCEL_PATH_ANALYSIS)
    df_secondary = load_excel_if_exists(EXCEL_PATH_SECONDARY)

    def safe_check(df):
        return not df.empty and "APP_ID" in df.columns and not df[df["APP_ID"].str.lower() == app_id.lower()].empty

    return (
        safe_check(df_analysis),
        safe_check(df_secondary),
        df_analysis,
        df_secondary,
    )

# --- App Search UI --- #
if not st.session_state.analysis_started and not st.session_state.search_triggered:
    st.markdown('<h2 style="font-weight:700; font-size:22px; margin-bottom:10px;">Search for an App</h2>', unsafe_allow_html=True)
    user_input = st.text_input("", placeholder="Type an app name to search. Eg: \"TikTok\"", key="user_input")

    if user_input:
        matches = search_apps_starting_with(user_input)
        if not matches:
            st.error("❌ No apps found starting with that name.")
        else:
            st.session_state.search_results = matches
            st.write("Here are the matching apps:")
            options = [f"{m['title']} — {m['appId']}" for m in matches]
            selected_option = st.selectbox("Select an app:", options)

            if selected_option:
                selected_index = options.index(selected_option)
                selected_app_id = matches[selected_index]["appId"]
                st.session_state.selected_option = selected_option
                st.session_state.selected_app_id = selected_app_id

                if st.button("🔍 Search Selected App"):
                    st.session_state.search_triggered = True
                    st.rerun()

# --- Handle Cached Analysis (load from Excel) --- #
if st.session_state.search_triggered and st.session_state.selected_app_id and not st.session_state.analysis_started:
    app_id = st.session_state.selected_app_id
    in_analysis, in_secondary, df_analysis, df_secondary = app_exists_in_any_excel(app_id)

    if in_analysis or in_secondary:
        df_combined = pd.concat([
            df_analysis[df_analysis["APP_ID"].str.lower() == app_id.lower()],
            df_secondary[df_secondary["APP_ID"].str.lower() == app_id.lower()]
        ])
        if not df_combined.empty:
            row = df_combined.iloc[0]
            display_app_header(row)
            display_app_analysis(row)
            feedback_url = "https://forms.office.com/Pages/ResponsePage.aspx?id=nKagUU8OPUu2QhLgExmGNdZ3ApRDp8pFkU4wg8uTI3ZUM0U4TEdDQTg2OFREVVdGMU5OV1NBSVRFSi4u&origin=Invitation&channel=1"
            st.info(f"📢 [Click here to submit feedback]({feedback_url})")
            st.session_state.analysis_row = row.to_dict()
            st.session_state.analysis_started = True

    else:
        # Run full analysis
        st.session_state.analysis_started = True
        start_time = time.time()
        progress_container = st.empty()
        info_container = st.empty()

        progress_bar = progress_container.progress(0)
        info_container.info(f"🔍 Checking privacy details for **{st.session_state.selected_option}**...")

        progress_bar.progress(5)
        app_data = cached_fetch_info(app_id)
        progress_bar.progress(10)

        if app_data:
            app_df = pd.DataFrame([app_data])
            app_df["Permissions Used"] = app_df["Perm"].apply(map_permissions_list)
            app_df["Risk Summary"] = app_df["Permissions Used"].apply(generate_risk_summary)
            progress_bar.progress(20)

            policy_url = app_data.get("Policy Link")
            if policy_url:
                policy_text = cached_fetch_policy_text(policy_url)
                progress_bar.progress(30)

                if policy_text:
                    segments = cached_extract_segments(policy_text)
                    cleaned = "\n".join(segments)
                    app_df["MergedPolicyText"] = [cleaned]
                    app_df["Policy Segments"] = [segments]

                    app_df['CleanText'] = app_df['MergedPolicyText'].apply(cached_clean_policy_text)
                    progress_bar.progress(45)

                    app_df[['Permissions Found', 'Policy Mismatches']] = app_df.apply(check_permissions, axis=1)
                    progress_bar.progress(55)

                    app_df[['Keyword Frequencies', 'Keyword Matches']] = app_df.apply(extract_permission_keyword_matches, axis=1)
                    progress_bar.progress(65)
                    
                    # Only run GDPR and BERT if CleanText tokens >= 10
                    if len(app_df.at[0, 'CleanText'].split()) >= 10:

                        app_df, segment_df = cached_process_policy_segments(app_df)
                        progress_bar.progress(70)

                        app_df = run_gdpr_processing(app_df)
                        progress_bar.progress(80)
                        
                        ##### FALLBACK SECTION STARTS HERE #####
                        # Get entities for the current app (assuming row index 0 here)
                        entities = app_df.at[0, 'entities']

                        # Extract corrected PD/NPD words *and* detection flags with confidence filtering
                        pd_words, npd_words, pd_detected, npd_detected = fallback_extract_words_from_entities(entities, confidence_threshold=0.65)

                        # Build the summary, passing detection flags too
                        summary = build_summary(pd_words, npd_words, pd_detected, npd_detected)

                        # Check if fallback needed based on existing summary and detection flags
                        existing_summary = app_df.at[0, "Data Collection Summary"].strip().lower()
                        should_fallback = (not existing_summary or "no specific data collection details" in existing_summary)

                        if should_fallback and (pd_detected or npd_detected):
                            print("🔁 Fallback activated: updating summary and mismatches using keyword traces.")
                            app_df.at[0, "Data Collection Summary"] = summary

                            found_perms = set()
                            for perm in perm_label:
                                terms = final_keyword.get(perm, [])
                                if terms:
                                    found_perms.add(perm)

                            current_mismatches = set(app_df.at[0, "Policy Mismatches"])
                            updated_mismatches = list(current_mismatches - found_perms)
                            app_df.at[0, "Policy Mismatches"] = updated_mismatches

                        ##### FALLBACK SECTION ENDS HERE #####

                        app_df[['Verdict', 'Legal Concerns', 'Recommendations', 'Overview']] = app_df.apply(classify_app_risk, axis=1)
                        progress_bar.progress(85)

                        app_df['Third_Party_Prediction'] = model.predict(app_df['CleanText'])
                        if hasattr(model, "predict_proba"):
                            app_df['Third_Party_Probability'] = model.predict_proba(app_df['CleanText'])[:, 1]
                        else:
                            app_df['Third_Party_Probability'] = None
                        progress_bar.progress(90)

                        app_df['Sharing_Summary'] = app_df.apply(generate_sharing_summary, axis=1)
                        progress_bar.progress(95)
                    else:
                        st.error("❌ Could not fetch this app's privacy policy.")
                    progress_bar.progress(100)
                    
                else:
                    st.warning("⚠️ Policy text too short for text analysis; skipping to prevent misinformation.")
                    app_df[['Verdict', 'Legal Concerns', 'Recommendations', 'Overview']] = app_df.apply(classify_app_risk, axis=1)
                    progress_bar.progress(85)
                    app_df['Third_Party_Prediction'] = None
                    app_df['Third_Party_Probability'] = None
                    app_df['Sharing_Summary'] = None
                    progress_bar.progress(95)   
            else:
                st.error("❌ No privacy policy link provided.")
                progress_bar.progress(100)

            append_to_excel(app_df, EXCEL_PATH_ANALYSIS)
            row = app_df.iloc[0]
            display_app_header(row)
            display_app_analysis(row)
            feedback_url = "https://forms.office.com/Pages/ResponsePage.aspx?id=nKagUU8OPUu2QhLgExmGNdZ3ApRDp8pFkU4wg8uTI3ZUM0U4TEdDQTg2OFREVVdGMU5OV1NBSVRFSi4u&origin=Invitation&channel=1"
            st.info(f"📢 [Click here to submit feedback]({feedback_url})")
            st.session_state.analysis_row = row.to_dict()
        else:
            st.error("❌ Failed to fetch app metadata.")
            progress_bar.progress(100)

        progress_container.empty()
        info_container.empty()
        elapsed = time.time() - start_time
        st.success(f"✅ Analysis completed in {int(elapsed // 60)}m {int(elapsed % 60)}s.")

# --- Recover View on Rerun (for download_button, etc.) --- #
elif st.session_state.analysis_started and st.session_state.analysis_row:
    row = pd.Series(st.session_state.analysis_row)
    display_app_header(row)
    display_app_analysis(row)
    feedback_url = "https://forms.office.com/Pages/ResponsePage.aspx?id=nKagUU8OPUu2QhLgExmGNdZ3ApRDp8pFkU4wg8uTI3ZUM0U4TEdDQTg2OFREVVdGMU5OV1NBSVRFSi4u&origin=Invitation&channel=1"
    st.info(f"📢 [Click here to submit feedback]({feedback_url})")

print('app.py loaded')


import ast
import base64
import os
import pandas as pd
import re
import streamlit as st
import streamlit.components.v1 as components
from bert import label_explanations


def display_app_header(selected):
    app_name = selected.get("App Name", "Unknown App")
    icon_url = selected.get("Icon URL", "").strip() or "https://www.svgrepo.com/show/533118/mobile-bolt.svg"
    policy_url = str(selected.get("Policy Link", "")).strip()
    category = selected.get("Category", "Unknown")
    age_group = selected.get("Age Group", "N/A")
    friendly_explanation = selected.get("Overview", "No explanation available.")

    


    col1, col2 = st.columns([1, 6])
    with col1:
        st.image(icon_url, width=80)
    with col2:
        st.header(app_name)
        st.markdown(f"**Found in genre _{category}_**, recommended for _{age_group}_.")
        if policy_url and policy_url.lower() != "nan":
            st.markdown(f'<a href="{policy_url}" target="_blank">🔗 Privacy Policy</a>', unsafe_allow_html=True)
        else:
            st.info("Privacy Policy URL not available.")

    # Normalize for matching
    text_lower = friendly_explanation.lower()

    if "too short" in text_lower:
        st.warning(friendly_explanation)   # yellow
    elif "concern" in text_lower:
        st.error(friendly_explanation)     # red
    elif "explained" in text_lower or "clearly" in text_lower:
        st.success(friendly_explanation)   # green
    else:
        st.info(friendly_explanation)      # blue (default)
        
    #st.markdown("---")

def format_bullet_points(text, emoji=""):
    if isinstance(text, str):
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        for sentence in parts:
            if sentence:
                st.markdown(f"- {sentence.strip()}")
    elif isinstance(text, list):
        for item in text:
            st.markdown(f"- {item}")
    else:
        st.info("None found.")

def display_app_analysis(row):
    st.subheader("🧐 App Permissions and What They Mean")
    format_bullet_points(row.get("Risk Summary", ""))

    st.subheader("📜 Legal Concerns (Detailed)")
    legal_concerns = row.get("Legal Concerns", [])
    if isinstance(legal_concerns, str):
        try:
            legal_concerns = ast.literal_eval(legal_concerns)
        except Exception:
            legal_concerns = []

    if isinstance(legal_concerns, list) and legal_concerns:
        for concern in legal_concerns:
            st.markdown(f"❗ {concern}")
    else:
        st.info("No legal concerns were found.")

        # --- Show GDPR Summary directly ---
    st.subheader("📋 GDPR Summary")
    gdpr_summary = row.get("GDPR Summary", "")
    if gdpr_summary:
        st.markdown(gdpr_summary)
    else:
        st.info("No GDPR summary available.")

    # --- Expander for more details ---
    with st.expander("Click to see how confident we are about this app's GDPR alignment based on its privacy policy"):
        # 1. PDF Download
        st.markdown("**Download GDPR Summary PDF**")
        pdf_path = row.get('PDF Path', None)

        # Make sure pdf_path is a non-empty string
        if isinstance(pdf_path, str) and pdf_path.strip() and os.path.exists(pdf_path):
            file_name = os.path.basename(pdf_path)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label=f"⬇️ Download GDPR Summary ({file_name})",
                data=pdf_bytes,
                file_name=file_name,
                mime="application/pdf",
            )
        else:
            st.info("No PDF available for download.")

        # 2. Top Labels
        st.markdown("**GDPR Labels**")
        max_labels = row.get("Top Labels", [])
        if isinstance(max_labels, str):
            try:
                max_labels = ast.literal_eval(max_labels)
            except Exception:
                max_labels = [max_labels]

        if max_labels and isinstance(max_labels, (list, tuple)):
            for label in max_labels:
                st.markdown(f"- {label}")
        else:
            st.info("No labels predicted. **Empty PDF** might have been generated.")

    # Subheader for data collection summary
    st.subheader("📦 Data Collection Summary")
    summary = (row.get("Data Collection Summary") or "").strip()
    st.markdown(summary if summary else "No summary of data collection was provided.")

    # Confidence in data collection predictions
    avg_probs = row.get("Average Label Probabilities", {})
    prediction_summary = str(row.get("Prediction Summary (Freq/Max)") or "").strip()

    # Group ordering for display
    group_order = [
        ("Information Type", "Information types detected:"),
        ("Collection Process", "Collection process:"),
        ("Purpose", "Purposes for collecting data:")
    ]

    # Only show if there are predictions or summaries
    if avg_probs or prediction_summary:
        with st.expander("Click to see how confident we are about the data this app collects based on its privacy policy"):

            # Show average confidence levels in grouped, plain-English format
            if avg_probs:
                if isinstance(avg_probs, str):
                    try:
                        avg_probs = ast.literal_eval(avg_probs)
                    except Exception:
                        avg_probs = {}

                if isinstance(avg_probs, dict):
                    if len(avg_probs) == 0:
                        st.markdown("Nothing to show")
                    else:
                        # Add main heading here
                        st.markdown("**Average Confidence Levels for Different Data Types:**")

                        # Loop through groups in defined order
                        for group_key, group_title in group_order:
                            st.markdown(f"**{group_title}**")
                            
                            # Filter and sort labels by highest probability first
                            group_items = [(label, prob) for label, prob in avg_probs.items() if label.startswith(group_key)]
                            group_items.sort(key=lambda x: x[1], reverse=True)

                            # Show each label with percentage and plain-English explanation
                            for label, prob in group_items:
                                short_label = label.split("_", 1)[1]  # Remove category prefix
                                explanation = label_explanations.get(label, label)
                                st.markdown(f"- **{short_label} ({prob:.0%})**: {explanation}")

                            st.markdown("")  # Blank line between groups
                            
            st.markdown("---")
                
            # Show prediction summary if available
            if prediction_summary:
                if (
                    prediction_summary is None or
                    (isinstance(prediction_summary, float) and pd.isna(prediction_summary)) or
                    (isinstance(prediction_summary, str) and prediction_summary.strip().lower() == "nan")
                ):
                    st.markdown("**Summary of Prediction Frequency and Confidence:**")
                    st.markdown("Nothing to show")
                else:
                    st.markdown("**Summary of Prediction Frequency and Confidence:**")
                    st.markdown(prediction_summary)
    else:
        st.info("This prediction is based on the GDPR policy analysis, no specific data collection predictions were made.")

    st.subheader("🔗 Third Party Sharing")
    sharing_summary = (row.get("Sharing_Summary") or "").strip()
    if sharing_summary:
        st.markdown("_Note: This prediction is based on privacy policy text._")
        st.markdown(sharing_summary)
    else:
        st.markdown("No specific third-party sharing prediction available.")

    st.subheader("🚦 Verdict")
    verdict = (row.get("Verdict", "") or "").strip()
    if verdict:
        verdict_lines = re.split(r'(?<=[.!?]) +', verdict)
        for line in verdict_lines:
            clean = line.strip()
            if clean:
                if "❌" in clean:
                    st.error(clean)
                elif "⚠️" in clean:
                    st.warning(clean)
                else:
                    st.success(clean)
    else:
        st.info("No verdict provided.")

    st.subheader("🜭 What You Can Do")
    st.markdown(row.get("Recommendations", ""))

    with st.expander("🛡️ How to Turn Off App Permissions"):
        st.markdown("""#### 📱 For Android:
1. Open **Settings** > **Apps**
2. Select the app
3. Tap **Permissions**
4. Toggle off what you don’t need

#### 🍏 For iPhone:
1. Open **Settings**
2. Scroll to the app
3. Toggle off permissions like Camera, Photos, etc.

""")
        


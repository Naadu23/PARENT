print("loading GDPR processing module")

# gdpr.py
import os
import re
import json
import pandas as pd
import unicodedata
from tqdm import tqdm
from transformers import pipeline
from fpdf import FPDF

tqdm.pandas()

# Initialize NER pipeline once
ner_pipe = pipeline("token-classification", model="PaDaS-Lab/gdpr-privacy-policy-ner", aggregation_strategy="simple")

# Load correction map and label/category mappings
# Label mappings for GDPR entities
label_mapping_privacy = {
    "DC": "Data Controller", "DP": "Data Processor", "DPO": "Data Protection Officer",
    "R": "Recipient", "TP": "Third Party", "A": "Authority", "DS": "Data Subject",
    "DSO": "Data Source", "RP": "Required Purpose", "NRP": "Not Required Purpose",
    "P": "Processing", "PD": "Personal Data", "NPD": "Non-Personal Data",
    "OM": "Organisational Measure", "TM": "Technical Measure", "LB": "Legal Basis",
    "CONS": "Consent", "CONT": "Contract", "LI": "Legitimate Interest",
    "ADM": "Automated Decision Making", "RET": "Retention", "SEU": "Scale EU",
    "SNEU": "Scale Non-EU", "RI": "Right", "LC": "Lodge Complaint",
    "DSR15": "Art. 15 Right to access by the data subject", "DSR16": "Art. 16 Right to rectification",
    "DSR17": "Art. 17 Right to erasure (‘right to be forgotten’)", "DSR18": "Art. 18 Right to restriction of processing",
    "DSR19": "Art. 19 Notification obligations", "DSR20": "Art. 20 Right to data portability",
    "DSR21": "Art. 21 Right to object", "DSR22": "Art. 22 Automated individual decision-making, including profiling"
}

# Concept-to-entity label grouping
category_mapping = {
    "Mentions Personal Data": {"PD", "NPD"},
    "Mentions Purpose of Use": {"P", "RP", "NRP", "RET", "OM", "TM", "ADM"},
    "Mentions Legal Basis": {"LB", "CONS", "CONT", "LI"},
    "Mentions Your Rights": {"RI", "DSR15", "DSR16", "DSR17", "DSR18", "DSR19", "DSR20", "DSR21", "DSR22", "LC"},
    "Mentions Responsible Parties": {"DC", "DP", "DPO", "A", "DS", "DSO"},
    "Mentions Third-Party Sharing": {"TP", "R", "SEU", "SNEU"}
}

correction_map = {
    # CAMERA
    "access camera": "camera access",
    "appear activity": "camera usage activity",
    "capture image": "captured images",
    "captured photo": "photos taken",
    "captured picture": "pictures taken",
    "face recognition": "facial recognition data",
    "facial recognition": "facial recognition data",
    "image information": "image data",
    "live video": "live video feed",
    "phone camera": "camera",
    "photo": "photos",
    "photo appear": "photo access",
    "photograph": "photographs",
    "record video": "video recordings",
    "snap": "captured images",
    "take photo": "photos taken",
    "take picture": "pictures taken",
    "use camera": "camera usage",
    "use photo": "photo usage",
    "video call": "video calls",
    "video recording": "video recordings",

    # CALENDAR
    "add event": "calendar events",
    "book appointment": "appointments",
    "calendar": "calendar data",
    "calendar access": "calendar access",
    "reminder": "reminders",
    "schedule meeting": "scheduled meetings",
    "view calendar": "calendar views",

    # SENSOR
    "accelerometer": "motion sensor data",
    "gyroscope": "motion tracking",
    "motion": "motion data",
    "orientation": "device orientation",
    "step count": "step tracking",

    # SMS
    "receive message": "received messages",
    "receive sms": "received text messages",
    "send message": "sent messages",
    "send sms": "sent text messages",
    "otp": "one-time passwords (OTP)",
    "text message": "text messages",
    "verification code": "verification codes",

    # PHONE_CALL
    "call history": "call history",
    "call log": "call logs",
    "dial number": "dialed numbers",
    "incoming call": "incoming calls",
    "outgoing call": "outgoing calls",
    "phone number": "phone numbers",
    "receive call": "received calls",

    # PERSISTENTID
    "advertising id": "advertising ID",
    "android id": "device ID",
    "device id": "device ID",
    "imei": "IMEI number",
    "serial number": "device serial number",

    # STORAGE
    "access media": "media files",
    "camera roll": "photo gallery",
    "download": "downloads",
    "media file": "media files",
    "photo album": "photo albums",
    "read storage": "read access to storage",
    "save file": "saved files",
    "upload": "uploaded files",
    "video file": "video files",

    # LOCATION
    "access location": "location access",
    "current location": "current location",
    "geolocation": "geolocation data",
    "gps": "GPS location",
    "ip address": "IP address",
    "location": "location information",
    "map": "location on maps",
    "track location": "location tracking",

    # MICROPHONE
    "audio": "audio recordings",
    "audio input": "audio input",
    "record audio": "recorded audio",
    "sound record": "sound recordings",
    "voice call": "voice calls",
    "voice message": "voice messages",

    # CONTACTS
    "access contact": "contact access",
    "add friend": "added contacts",
    "contact": "contact details",
    "contact information": "contact information",
    "contact list": "contact list",
    "phonebook": "phonebook entries",
    "read contact": "read contact list",
}

# explain gdpr labels with examples
label_explanations = {
    "Data Controller": "Meaning: Who decides how your data is used, eg. The app company",
    "Data Processor": "Meaning: Who handles your data on behalf of the controller, eg. Cloud service provider",
    "Data Protection Officer": "Meaning: Person responsible for data protection compliance in the company, eg. Privacy officer",
    "Recipient": "Meaning: Who receives your data, eg. Partner companies",
    "Third Party": "Meaning: External companies that may get your data, eg. Advertisers",
    "Authority": "Meaning: Regulatory bodies overseeing data protection, eg. Data protection commission",
    "Data Subject": "Meaning: The person whose data is collected, eg. You or your child",
    "Data Source": "Meaning: Where your data comes from, eg. Your device or app usage",
    "Required Purpose": "Meaning: The reason your data is needed, eg. To provide app features",
    "Not Required Purpose": "Meaning: Data used for non-essential reasons, eg. Marketing",
    "Processing": "Meaning: How your data is handled and used, eg. Analyzing usage patterns",
    "Personal Data": "Meaning: Information that identifies you, eg. Name, email",
    "Non-Personal Data": "Meaning: Data that doesn’t identify you, eg. Device model",
    "Organisational Measure": "Meaning: Company policies protecting your data, eg. Staff training",
    "Technical Measure": "Meaning: Technology to secure your data, eg. Encryption",
    "Legal Basis": "Meaning: Legal reason for data use, eg. Consent or contract",
    "Consent": "Meaning: Your permission to use data, eg. Accepting terms",
    "Contract": "Meaning: Agreement between you and app, eg. Service agreement",
    "Legitimate Interest": "Meaning: Valid reason for data use without consent, eg. Fraud prevention",
    "Automated Decision Making": "Meaning: Decisions made by computers, eg. Personalized ads",
    "Retention": "Meaning: How long data is kept, eg. 1 year after last use",
    "Scale EU": "Meaning: App operates across EU countries, eg. Available in multiple EU nations",
    "Scale Non-EU": "Meaning: App operates outside EU, eg. US or Asia",
    "Right": "Meaning: Your rights over your data, eg. Right to access or delete",
    "Lodge Complaint": "Meaning: Your right to complain, eg. Reporting to data protection authority",
    "Art. 15 Right to access by the data subject": "Meaning: Your right to see your data, eg. Requesting data copies",
    "Art. 16 Right to rectification": "Meaning: Your right to correct data, eg. Fixing wrong info",
    "Art. 17 Right to erasure (‘right to be forgotten’)": "Meaning: Your right to delete data, eg. Request app to delete account",
    "Art. 18 Right to restriction of processing": "Meaning: Your right to limit use, eg. Temporarily stop processing",
    "Art. 19 Notification obligations": "Meaning: Obligation to inform changes, eg. Policy updates",
    "Art. 20 Right to data portability": "Meaning: Your right to move data, eg. Transfer to another app",
    "Art. 21 Right to object": "Meaning: Your right to object to processing, eg. Opt-out of marketing",
    "Art. 22 Automated individual decision-making, including profiling": "Meaning: Protection from automated decisions, eg. Credit scoring",
}


# load final_keywords.json externally before calling generate_summary_paragraph
with open('PARENT_App/data/final_keywords.json', 'r', encoding='utf-8') as f:
    final_keyword = json.load(f)

def split_sentences(text):
    if not isinstance(text, str):
        text = " ".join(map(str, text)) if isinstance(text, list) else str(text)
    return re.split(r'(?<=[.!?])\s+', text)

def extract_entities(text_segments, threshold=0.6, min_tokens=10, tokenizer=None):
    filtered_segments = []
    for seg in text_segments if isinstance(text_segments, list) else [text_segments]:
        token_count = len(tokenizer.tokenize(seg)) if tokenizer else len(seg.split())
        if token_count >= min_tokens:
            filtered_segments.append(seg)
    all_ents = []
    for seg in filtered_segments:
        ents = ner_pipe(seg)
        if isinstance(ents, list):
            all_ents.extend(ents)
    filtered = [ent for ent in all_ents if ent.get('score', 0) >= threshold]
    for ent in filtered:
        ent['label_readable'] = label_mapping_privacy.get(ent.get('entity_group', ''), ent.get('entity_group', ''))
    unique = {}
    for ent in filtered:
        key = (ent.get('entity_group', ''), ent.get('word', ''))
        if key not in unique or unique[key]['score'] < ent['score']:
            unique[key] = ent
    return list(unique.values())

def summarize_concepts(row):
    text = row['Policy Segments']
    if not isinstance(text, str):
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        else:
            text = str(text)
    sentences = split_sentences(text)
    entities = row['entities']
    results = {}
    for concept, labels in category_mapping.items():
        entries = []
        for ent in entities:
            if ent['entity_group'] in labels:
                label = ent['label_readable']
                conf = round(ent['score'] * 100, 1)
                sentence = next((s for s in sentences if ent['word'] in s), "").strip()
                entry = f"{label} ({conf}%): {sentence}"
                entries.append(entry)
        results[concept] = " | ".join(entries)
    return pd.Series(results)

def group_labels_by_sentence(row):
    text = row['Policy Segments']
    if not isinstance(text, str):
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        else:
            text = str(text)
    sentences = split_sentences(text)
    entities = row['entities']
    if not entities:
        return ""
    sentence_map = {}
    for ent in entities:
        for sent in sentences:
            if ent['word'] in sent:
                if sent not in sentence_map:
                    sentence_map[sent] = {}
                label = ent['label_readable']
                score = ent['score'] * 100
                if label not in sentence_map[sent] or score > sentence_map[sent][label]:
                    sentence_map[sent][label] = score
                break
    summaries = []
    for sent, label_scores in sentence_map.items():
        labels_sorted = sorted(label_scores.items(), key=lambda x: x[0])
        label_str = ", ".join([f"{label} ({score:.1f}%)" for label, score in labels_sorted])
        summaries.append(f"{label_str}: {sent.strip()}")
    return " | ".join(summaries)

def summarize_coverage(row):
    mentioned = [concept for concept in category_mapping if row.get(concept)]
    return ", ".join(mentioned) if mentioned else "No GDPR concepts detected"

def generate_summary_paragraph(entities):
    # Use correction_map and final_keyword from outer scope or pass as params if preferred
    cat_to_words = {}
    for ent in entities:
        label = ent.get('label_readable')
        word = ent.get('word')
        if label and word:
            if label not in cat_to_words:
                cat_to_words[label] = set()
            cat_to_words[label].add(word)
    for label in cat_to_words:
        cat_to_words[label] = sorted(cat_to_words[label])

    present_sentences = []
    missing_sentences = []
    paragraphs = []

    if cat_to_words.get("Data Controller"):
        present_sentences.append("This privacy policy explains who is responsible for handling your child’s data.\n\n")
    else:
        missing_sentences.append("This privacy policy does not clearly state who controls your child’s data.\n\n")

    # Check processing and technical measures
    if cat_to_words.get("Processing") and cat_to_words.get("Technical Measure"):
        present_sentences.append(
            "It also explains why the data is collected, how it's used to support their experience, and how it's protected.\n\n"
        )
    else:
        if cat_to_words.get("Processing"):
            present_sentences.append(
                "It explains why the data is collected and how it's used to support their experience.\n\n"
            )
        else:
            missing_sentences.append(
                "it does not clearly explain why the app collects the requested information or how it is used.\n\n"
            )

        if cat_to_words.get("Technical Measure"):
            present_sentences.append("It explains how your data is kept safe and protected.\n\n")
        else:
            missing_sentences.append("it does not clearly explain how your data is kept safe.\n\n")

    # Check legal basis and third-party sharing
    if cat_to_words.get("Legal Basis") and cat_to_words.get("Third Party"):
        present_sentences.append(
            "It mentions legal reasons for collecting and using this data, and that some information may be shared with other companies when needed.\n\n"
        )
    else:
        if cat_to_words.get("Legal Basis"):
            present_sentences.append(
                "It mentions the legal reasons the app is allowed to collect and use this information.\n\n"
            )
        else:
            missing_sentences.append(
                "it does not clearly explain the legal reasons for collecting or using this information.\n\n"
            )

        if cat_to_words.get("Third Party"):
            present_sentences.append(
                "It mentions that some information may be shared with other companies when needed.\n\n"
            )
        else:
            missing_sentences.append(
                "it does not make it clear if or when data is shared with other companies.\n\n"
            )

    # Check rights
    if cat_to_words.get("Right"):
        present_sentences.append("The policy explains user rights when it comes to personal data.\n\n")
    else:
        missing_sentences.append("it does not explain user rights under data protection laws.\n\n")

    # Combine summary
    if present_sentences:
        paragraphs.append(" ".join(present_sentences))

    if missing_sentences:
        # Capitalize first letter of each missing sentence
        missing_sentences = [s if i > 0 and missing_sentences[i-1].strip().lower().endswith("however,") else s[0].upper() + s[1:] if s else s for i, s in enumerate(missing_sentences)]
        missing_text = " ".join(missing_sentences).strip()

        if present_sentences:
            paragraphs.append("However, " + missing_text)
        else:
            paragraphs.append(missing_text)


    key_elements = [
        "Data Controller", "Personal Data", "Processing",
        "Legal Basis", "Technical Measure", "Third Party", "Right"
    ]
    present_elements = sum(1 for elem in key_elements if cat_to_words.get(elem))

    if present_elements >= 5:
        paragraphs.insert(0, "This privacy policy appears to comply with GDPR transparency requirements.\n\n")
    elif present_elements >= 2:
        paragraphs.insert(0, "This privacy policy provides some GDPR-related information but is incomplete.\n\n")
    else:
        paragraphs.insert(0, "This privacy policy lacks important GDPR information and may not be compliant.\n\n")

    return " ".join(paragraphs)

def get_max_labels(entity_list):
    label_conf = {}
    for ent in entity_list:
        label = ent['label_readable']  
        score = ent['score']          

        # Store max confidence for each label
        if label not in label_conf or label_conf[label] < score:
            label_conf[label] = score

    # Format output with gdpr explanation
    results = []
    for label, score in label_conf.items():
        explanation = label_explanations.get(label, "") 
        formatted = f"{label} ({score*100:.0f}%)"
        if explanation:
            formatted += f" - {explanation}"
        results.append(formatted)

    return results




def create_pdf_for_app(app_name, summary_text, output_dir="output_gdpr"):
    # Clean text for PDF (remove unicode)
    clean_text = unicodedata.normalize('NFKD', str(summary_text)).encode('ascii', 'ignore').decode('ascii')

    # Sanitize app name to safe filename (lowercase)
    safe_name = "".join(c if c.isalnum() else " " for c in app_name).lower().replace(" ", "_")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # PDF path based on sanitized app name
    pdf_path = os.path.join(output_dir, f"{safe_name}.pdf")

    # If PDF already exists, skip creating
    if os.path.exists(pdf_path):
        return pdf_path

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    # Split text by '|', ignoring empty splits
    parts = [part.strip() for part in clean_text.split('|') if part.strip()]

    for part in parts:
        # Find the first colon in the line to split label and content
        colon_index = part.find(':')
        if colon_index != -1:
            label = part[:colon_index + 1].strip()   # include colon
            content = part[colon_index + 1:].strip()
            
            # Print label in bold
            pdf.set_font("Arial", 'B', 12)
            pdf.multi_cell(0, 8, label)

            # Print content normal
            if content:
                pdf.set_font("Arial", '', 12)
                pdf.multi_cell(0, 8, content)
        else:
            # No colon found, print whole line normally
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 8, part)

    pdf.output(pdf_path)
    return pdf_path


##### FALL BACK FUNCTION IF NEEDED #####

def correct_terms(terms):
    corrected = []
    for term in terms:
        clean_term = term.strip().lower()
        if clean_term in correction_map:
            corrected.append(correction_map[clean_term])
        else:
            corrected.append(clean_term.capitalize())
    return corrected

# Function to build fallback summary of collected info
def build_summary(pd_words, npd_words, pd_detected=False, npd_detected=False):
    def oxford_join(words):
        # Join list of words with commas and 'and' for the last item
        if len(words) == 1:
            return words[0]
        return ", ".join(words[:-1]) + ", and " + words[-1]

    # Prepare phrase for personal data (PD)
    if pd_words:
        # If we have specific PD terms, list them
        pd_text = f"personal information such as {oxford_join(pd_words)}"
    elif pd_detected:
        # If PD detected but no specific terms, use generic phrase
        pd_text = "personal information"
    else:
        # No PD detected at all
        pd_text = ""

    # Prepare phrase for non-personal data (NPD)
    if npd_words:
        # If we have specific NPD terms, list them
        npd_text = f"non-personal information like {oxford_join(npd_words)}"
    elif npd_detected:
        # If NPD detected but no specific terms, use generic phrase
        npd_text = "non-personal information"
    else:
        # No NPD detected at all
        npd_text = ""

    # Compose final summary string based on available info
    if pd_text and npd_text:
        return f"It mentions collecting {pd_text}, as well as {npd_text}."
    elif pd_text:
        return f"It mentions collecting {pd_text}."
    elif npd_text:
        return f"It mentions collecting {npd_text}."
    else:
        return "It does not specify clearly what types of information are collected."


# Permissions labels and classifications
perm_label = [
    'CAMERA', 'MICROPHONE', 'PHONE_CALL', 'SENSOR', 'SMS',
    'CALENDAR', 'CONTACTS', 'LOCATION', 'STORAGE', 'PERSISTENTID'
]
personal_permissions = {'CAMERA', 'MICROPHONE', 'PHONE_CALL', 'SMS', 'CALENDAR', 'CONTACTS', 'LOCATION'}
non_personal_permissions = {'SENSOR', 'STORAGE', 'PERSISTENTID'}


# Extract and correct PD and NPD terms from entities with confidence filtering
def fallback_extract_words_from_entities(entities, confidence_threshold=0.6):
    # Track whether each permission label is detected (regardless of final_keyword)
    detected_perms = {perm: False for perm in perm_label}

    pd_words = []
    npd_words = []

    # Collect all entity words by permission label meeting confidence threshold
    filtered_keywords = {perm: [] for perm in perm_label}
    for ent in entities:
        conf = ent.get('confidence', 0)
        label = ent.get('label_readable')
        word = ent.get('word')

        if conf >= confidence_threshold and label and word:
            perm = label.upper()
            if perm in perm_label:
                filtered_keywords[perm].append(word)
                detected_perms[perm] = True  # Mark permission detected

    # For each permission label, attempt to get normalized term from final_keyword
    for perm in perm_label:
        if detected_perms[perm]:
            terms = final_keyword.get(perm, [])
            if terms:
                # Use corrected normalized term
                term = correct_terms([terms[0]])[0]
            else:
                # No normalized term available, fallback to None (skip adding raw words)
                term = None

            # Add terms to personal or non-personal list accordingly
            if term:
                if perm in personal_permissions:
                    pd_words.append(term)
                elif perm in non_personal_permissions:
                    npd_words.append(term)

        # Limit max 5 items for each category
        if len(pd_words) >= 5 and len(npd_words) >= 5:
            break

    # Flags to indicate detection of personal/non-personal info, even without normalized terms
    pd_detected = any(detected_perms[p] for p in personal_permissions)
    npd_detected = any(detected_perms[p] for p in non_personal_permissions)

    # Return corrected words and detection flags
    return pd_words[:5], npd_words[:5], pd_detected, npd_detected







def run_gdpr_processing(app_df, output_dir="output_gdpr"):
    os.makedirs(output_dir, exist_ok=True)

    # Extract entities
    app_df['entities'] = app_df['Policy Segments'].progress_apply(lambda x: extract_entities(x))

    # Summarize concepts into columns
    summaries = app_df.progress_apply(summarize_concepts, axis=1)
    for col in summaries.columns:
        app_df[col] = summaries[col]

    # Generate label_sentence_summary (your top labels source)
    app_df['label_sentence_summary'] = app_df.progress_apply(group_labels_by_sentence, axis=1)

    # Apply get_max_labels to generate Top Labels nicely formatted
    app_df['Top Labels'] = app_df['entities'].apply(get_max_labels)

    # Add GDPR Summary Coverage if needed
    app_df['GDPR Summary Coverage'] = app_df.apply(summarize_coverage, axis=1)

    # Generate GDPR Summary paragraph if needed
    app_df['GDPR Summary'] = app_df['entities'].progress_apply(generate_summary_paragraph)

    # Create PDFs and add PDF paths to DataFrame, using label_sentence_summary as content
    pdf_paths = []
    for _, row in tqdm(app_df.iterrows(), total=len(app_df)):
        app_name = row.get("App Name") or "app"
        label_summary = row.get("label_sentence_summary", "")
        pdf_path = create_pdf_for_app(app_name, label_summary, output_dir)
        pdf_paths.append(pdf_path)

    app_df['PDF Path'] = pdf_paths

    return app_df



print("GDPR processing module loaded")

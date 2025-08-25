print('loading bert.py')
# bert.py
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from huggingface_hub import hf_hub_download
from transformers import logging as transformers_logging
from pathlib import Path
from tqdm.auto import tqdm
from itertools import chain
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
from collections import defaultdict
from itertools import chain
import pandas as pd
from tqdm import tqdm
import time

tqdm.pandas()
transformers_logging.set_verbosity_error()

MAX_LEN = 512
REPO_ID = "Bnaad/PARENT_bert"
device = torch.device("cpu")
MIN_TOKENS = 10
MAX_SEGMENTS = 70

label_names = [
    "Information Type_IP address and device IDs",
    "Information Type_Contact information",
    "Information Type_Location",
    "Information Type_Personal identifier",
    "Information Type_Health, genetic, or biometric data",
    "Information Type_Computer information",
    "Information Type_User online activities",
    "Information Type_Generic personal information",
    "Collection Process_Collected on first-party website/app",
    "Collection Process_Shared by first party with a third party",
    "Purpose_Advertising or marketing",
    "Purpose_Analytics or research",
    "Purpose_Essential service or feature",
    "Purpose_Service operation and security",
    "Purpose_Legal requirement"
]

label_explanations = {
    # Information Type
    "Information Type_IP address and device IDs": "Numbers or codes that identify your device or internet connection, eg. your phone’s ID or your home IP address.",
    "Information Type_Contact information": "Ways to reach you directly, eg. your email address, phone number, or mailing address.",
    "Information Type_Location": "Where you are or have been, eg. GPS location or city name.",
    "Information Type_Personal identifier": "Data that directly names you, eg. your full name or national ID number.",
    "Information Type_Health, genetic, or biometric data": "Sensitive details about your body or identity, eg. medical history, fingerprints, or DNA.",
    "Information Type_Computer information": "Technical details about your device, eg. operating system, browser type, or device model.",
    "Information Type_User online activities": "What you do online, eg. websites visited, clicks, or app activity.",
    "Information Type_Generic personal information": "General facts about you, eg. hobbies, likes, or lifestyle preferences.",

    # Collection Process
    "Collection Process_Collected on first-party website/app": "Data you give directly to the company through their site or app, eg. filling out a form.",
    "Collection Process_Shared by first party with a third party": "The company gives your data to another organization, eg. advertisers or partners.",

    # Purpose
    "Purpose_Advertising or marketing": "Used to promote products or services to you, eg. sending promotional emails.",
    "Purpose_Analytics or research": "Used to study trends or improve services, eg. tracking app performance.",
    "Purpose_Essential service or feature": "Needed to make the service work, eg. logging you in or processing payments.",
    "Purpose_Service operation and security": "Used to keep the service running safely, eg. preventing fraud or hacking.",
    "Purpose_Legal requirement": "Collected because the law says it must be, eg. keeping tax records."
}


@st.cache_resource
def load_models():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    models = {}
    for label in label_names:
        safe_label = label.replace(" ", "_").replace("/", "_")
        filename = f"torchscript_{safe_label}.pt"
        model_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
        print(f"Downloaded {filename} to {model_path}")
        model = torch.jit.load(model_path, map_location=device)
        model.to(device)
        model.eval()
        models[label] = model
    return tokenizer, models

tokenizer, models = load_models()

def get_tokenizer_and_models():
    return load_models()

def model_predict(model, inputs):
    with torch.no_grad():
        filtered_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        outputs = model(**filtered_inputs)
        probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()
    return probs


def predict_labels(segments, batch_size=16):
    results = [{} for _ in segments]

    for start in range(0, len(segments), batch_size):
        # Tokenize once per batch
        batch = segments[start:start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # running all models on the same tokenized batch
        for label, model in models.items():
            start_time = time.time()

            with torch.no_grad():
                outputs = model(inputs["input_ids"], inputs["attention_mask"])
                probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()

            # Handle single vs batch outputs
            if probs.ndim == 0:
                probs = [probs]

            for i, prob in enumerate(probs):
                global_idx = start + i
                results[global_idx][label] = {
                    "probability": float(prob),
                    "prediction": 1 if prob > 0.5 else 0
                }

            elapsed = time.time() - start_time
            print(f"{label} processed in {elapsed:.2f} seconds (batch {start // batch_size + 1})")

    return results


def summarize_predicted_labels_paragraph(label_list):
    label_map = {
        "Information Type_IP address and device IDs": "device ID",
        "Information Type_Contact information": "contact information",
        "Information Type_Location": "location",
        "Information Type_Personal identifier": "personal ID",
        "Information Type_Health, genetic, or biometric data": "health or biometric data",
        "Information Type_Computer information": "device/computer info",
        "Information Type_User online activities": "online activity",
        "Information Type_Generic personal information": "other personal information",
        "Collection Process_Collected on first-party website/app": "directly through the app",
        "Collection Process_Shared by first party with a third party": "some of this data may be shared with third parties",
        "Purpose_Advertising or marketing": "advertising",
        "Purpose_Analytics or research": "analytics or research",
        "Purpose_Essential service or feature": "essential services",
        "Purpose_Service operation and security": "security or app functionality",
        "Purpose_Legal requirement": "legal compliance"
    }

    info_types, collection_methods, purposes = [], [], []

    for label in label_list:
        if label.startswith("Information Type_"):
            info_types.append(label_map[label])
        elif label.startswith("Collection Process_"):
            collection_methods.append(label_map[label])
        elif label.startswith("Purpose_"):
            purposes.append(label_map[label])

    def list_to_str(items):
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + " and " + items[-1]

    parts = []
    if info_types:
        parts.append(f"This app may collect information such as {list_to_str(info_types)}")
    if collection_methods:
        if parts:
            parts[-1] += f", {list_to_str(collection_methods)}"
        else:
            parts.append(f"This app may collect data {list_to_str(collection_methods)}")
    if purposes:
        parts.append(f"\n\nData collected is used for {list_to_str(purposes)}")

    paragraph = ". ".join(parts).strip() + "."
    return paragraph if paragraph != "." else "No specific data collection details were detected."

# Main analyze function


def process_policy_segments(
    app_df,
    tokenizer,
    predict_labels,
    summarize_predicted_labels_paragraph,
    min_tokens=10,
    max_segments=100
):
    # segments for the selected app
    segments = app_df["Policy Segments"].iloc[0]

    # Filter segments based on token count
    filtered_segments = []
    filtered_indices = []

    for i, seg in enumerate(segments):
        token_count = len(tokenizer.tokenize(seg))
        if token_count >= min_tokens:
            filtered_segments.append(seg)
            filtered_indices.append(i)

    # Limit to MAX_SEGMENTS
    if len(filtered_segments) > max_segments:
        filtered_segments = filtered_segments[:max_segments]
        filtered_indices = filtered_indices[:max_segments]

    # Run predictions on filtered segments
    segment_predictions_filtered = predict_labels(filtered_segments)

    # Initialize full list of predictions
    segment_predictions = [None] * len(segments)
    for idx, pred in zip(filtered_indices, segment_predictions_filtered):
        segment_predictions[idx] = pred

    # predicted labels for each segment
    segment_labels = []
    for pred in tqdm(segment_predictions, desc="Filtering predicted labels"):
        if pred is None:
            segment_labels.append([])
        else:
            labels = [label for label, res in pred.items() if res["probability"] > 0.65]
            segment_labels.append(labels)

    # Prediction Summary
    prediction_summaries = []
    for pred in segment_predictions:
        if pred is None:
            prediction_summaries.append("")
        else:
            summary = ", ".join(
                f"{label.split('_')[-1]} ({res['probability']:.0%})"
                for label, res in pred.items() if res['probability'] > 0.65
            )
            prediction_summaries.append(summary)

    # simple summaries
    segment_summaries = [
        summarize_predicted_labels_paragraph(labels)
        for labels in tqdm(segment_labels, desc="Summarizing segments")
    ]

    # df of results
    segment_df = pd.DataFrame({
        "Segment Text": segments,
        "Predicted Labels": segment_labels,
        "Prediction_Summary": prediction_summaries,
        "Summary": segment_summaries
    })

    # combine to unique label-based summary
    all_labels = list(chain.from_iterable(segment_labels))
    unique_labels = sorted(set(all_labels))
    policy_summary = summarize_predicted_labels_paragraph(unique_labels)

    # frequency and max probability
    label_freq = defaultdict(int)
    label_max_prob = defaultdict(float)

    for pred in segment_predictions:
        if pred is not None:
            for label, result in pred.items():
                prob = result["probability"]
                if prob > 0.65:
                    label_freq[label] += 1
                    label_max_prob[label] = max(label_max_prob[label], prob)

    sorted_labels = sorted(label_freq.items(), key=lambda x: (-x[1], -label_max_prob[x[0]]))
    freq_max_summary = ", ".join(
        f"{label.split('_')[-1]} (Freq: {freq}, MaxProb: {label_max_prob[label]:.0%})"
        for label, freq in sorted_labels
    )

    # average prediction probabilities 
    label_totals = defaultdict(float)
    label_counts = defaultdict(int)

    for pred in segment_predictions:
        if pred is not None:
            for label, result in pred.items():
                prob = result["probability"]
                if prob > 0.5:
                    label_totals[label] += prob
                    label_counts[label] += 1

    average_predictions = {
        label: label_totals[label] / label_counts[label]
        for label in label_totals
    }

    avg_summary = ", ".join(
        f"{label.split('_')[-1]} ({prob:.0%})"
        for label, prob in sorted(average_predictions.items(), key=lambda x: -x[1])
        if prob > 0.5
    )

    # Store into app_df
    app_df["Average Label Probabilities"] = [average_predictions]
    app_df["Prediction Summary (Freq/Max)"] = [freq_max_summary]
    app_df["Data Collection Summary"] = [policy_summary]

    return app_df, segment_df


print('bert.py loaded')


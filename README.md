# PARENT: Privacy App Review for Non-Technical Users

**“Understand What Your Child’s Apps Are Really Doing — Know the Risks, Protect Their Clicks.”**

PARENT is a hybrid framework and interactive web tool designed to help parents understand privacy risks in mobile applications. By combining machine learning, natural language processing (NLP), and rule-based analysis, PARENT assesses app permissions, privacy policies, and GDPR compliance in an easy-to-understand way.

---

##  Features

- **Policy Analysis with BERT:** 15 independent BERT models fine-tuned on the MAPP dataset to identify what data is collected, why, and how.
- **NER Support:** Integration with a pre-trained GDPR NER BERT model (from PADAS) for alignment with GDPR requirements.
- **Permission-Keyword Mapping:** Keywords linked to 10 most sensitive permissions (from MPP270 dataset) to evaluate if permissions requested are clearly explained in privacy policies.
- **Rule-Based Compliance Checks:** If-else logic to verify permission-policy alignment and generate clear explanations.
- **Third-Party Sharing Detection:** Logistic regression model to predict if user data is shared with external entities.
- **Verdicts Based on Regulation:** Guidance based on GDPR Articles 5 & 8 and UK Age Appropriate Design Code.
- **Interactive Dashboard:** Streamlit UI for real-time analysis of apps from Google Play, with downloadable GDPR summaries in PDF format.

---

##  Project Structure

```PARENT/
│_ training_data/
├── requirements.txt # Python dependencies
└── PARENT_App/
├── output_gdpr/ # Generated GDPR summary PDFs
├── data/
│ ├── final_keywords.json # Keywords per permission
│ ├── processed_output.xlsx # Initial app database
│ └── app_analysis_results.xlsx # Newly analysed apps
├── logistic.joblib # Trained logistic regression model
├── bert.py # Scripts to load and run BERT models
├── gdpr.py # GDPR classification utilities
├── ui.py # Streamlit UI components
├── app.py # Main app entrypoint
├── utils.py # Helper functions
└── logistic.py # Logistic regression pipeline
```

# install dependencies
pip install -r requirements.txt

# run streamlit
streamlit run PARENT_App/app.py


#  How It Works

Privacy Policy Analysis: BERT models segment the policy text to detect data collection, purposes, and processing details.

Permission-Policy Alignment: Keywords mapped to permissions are matched to the policy to check justification.

Third-Party Risk: Logistic regression predicts whether data is shared externally.

Verdict Generation: Rule-based logic combines analysis with GDPR/UK privacy rules to provide clear, actionable advice for parents.

## Demo Video

Watch the PARENT demo video: [Click here](https://anonymous.4open.science/w/PARENT-48EA/demo/index.html)


##  References

- **MAPP Dataset:** Arora, S., Hosseini, H., Utz, C., Bannihatti Kumar, V., Dhellemmes, T., Ravichander, A., Story, P., Mangat, J., Chen, R., Degeling, M., Norton, T.B., Hupperich, T., Wilson, S., & Sadeh, N.M. (2022). *A tale of two regulatory regimes: Creation and analysis of a bilingual privacy policy corpus*. Proceedings of the International Conference on Language Resources and Evaluation (LREC 2022). [PDF link](https://aclanthology.org/2022.lrec-1.585.pdf) [Accessed 12 July 2025].

- **GDPR NER Dataset:** Darji, H. (2024). *GDPR-Compliant NER dataset*. Hugging Face Datasets. [Dataset link](https://huggingface.co/datasets/PaDaS-Lab/gdpr-compliant-ner/) [Accessed 26 July 2025].

- **PermPress Dataset:** Rahman, M.S., Chakraborty, S., Rahman, A., Islam, S., Bhuiyan, M.Z.A., & Buyya, R. (2022). *PermPress: Machine learning-based pipeline to evaluate permissions in app privacy policies*. IEEE Access, 10, 89248–89269. [DOI link](https://doi.org/10.1109/ACCESS.2022.3199882) [Accessed 12 July 2025].

- **MAPS Dataset (App350):** Zimmeck, S., Story, P., Smullen, D., Ravichander, A., Wang, Z., Reidenberg, J., Russell, N.C., & Sadeh, N. (2019). *MAPS: Scaling privacy compliance analysis to a million apps*. Proceedings on Privacy Enhancing Technologies, (3), 66–86. [DOI link](https://doi.org/10.2478/popets-2019-0037) [Accessed 12 July 2025].

- **TorchScript BERT Models for PARENT:** [Hugging Face repository](https://huggingface.co/Bnaad/PARENT_bert)
-   GDPR Articles 5 & 8
-   UK Age Appropriate Design Code

⚠️ Notes

The system is designed for educational and awareness purposes, not legal advice.

Ensure consistent paths when adding or updating models.

Permission-policy checks focus on the 10 most sensitive permissions relevant to children's apps.

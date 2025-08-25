
print("loading logistic.py")
import pandas as pd
import re
# Add summary with both prediction and probability

def extract_bert_sharing_info(row):
    # get freq from 'Prediction Summary (Freq/Max)'
    freq = 0
    freq_max_str = row.get('Prediction Summary (Freq/Max)', '')

    # find phrase
    freq_match = re.search(r'Shared by first party with a third party \(Freq: (\d+),', freq_max_str)
    if freq_match:
        freq = int(freq_match.group(1))

    # get avg_prob
    avg_prob = 0.0
    avg_probs_dict = row.get('Average Label Probabilities', {})
    key_to_find = 'Shared by first party with a third party'
    for k, v in avg_probs_dict.items():
        if k.endswith(key_to_find):
            avg_prob = v
            break

    return avg_prob, freq


def generate_sharing_summary(row):
    app_name = row['App Name']
    logistic_prob = row.get('Third_Party_Probability', 0.0)
    bert_avg_prob, bert_freq = extract_bert_sharing_info(row)

    likely_shared = logistic_prob > 0.3 or bert_avg_prob > 0.65

    if likely_shared:
        lines = [f"⚠️ **{app_name}** is likely to share your child’s data with other companies.\n"]
        
        if logistic_prob > 0:
            lines.append(f"- It might say this clearly in the privacy policy (**{logistic_prob*100:.0f}% chance**).")
        
        if bert_avg_prob > 0 and bert_freq > 0:
            lines.append(
                f"- The way the policy is written strongly suggests sharing (**{bert_avg_prob*100:.0f}% chance**), "
                f"with about **{bert_freq} parts** of the policy pointing to this."
            )
        elif bert_avg_prob > 0:
            lines.append(
                f"- The way the policy is written strongly suggests sharing (**{bert_avg_prob*100:.0f}% chance**)."
            )
        
        return "\n".join(lines)
    else:
        lines = [f"✅ **{app_name}** probably doesn’t share your or your child’s data with other companies.\n"]
        
        if logistic_prob > 0:
            lines.append(f"- It doesn’t clearly say it does (**{logistic_prob*100:.0f}% chance**).")
        
        if bert_avg_prob > 0:
            lines.append(
                f"- And the way the policy is written doesn’t strongly suggest it either (**{bert_avg_prob*100:.0f}% likelihood based on wording**)."
            )
        
        if logistic_prob == 0 and bert_avg_prob == 0:
            lines.append("- There is no clear indication from the policy that data is shared.")
        
        return "\n".join(lines)

print('logistic.py loaded')
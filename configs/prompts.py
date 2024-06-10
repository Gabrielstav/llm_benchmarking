# just storing prompts here for the time being 

def get_domain_prompt_batch() -> dict:

    prompt = {
    "template"
     :"""
    Goal:
    Please evaluate [{{num_of_items}}] domain names to determine if they are likely used \
    for malicious practices such as deceptive practices, distributing malware, phishing, \
    get rich fast scams, lottery scams, pyramid scams, brand impersonation or others.

    Domain Names for Analysis:
    [{{domain_names}}]

    Evaluation Criteria:
    1. Check for brand impersonation using common abuse affixes or homoglyphs.
    2. Assess the domain name for poor readability, high entropy, or nonsensical order of \
    common dictionary words.

    Instructions:
    1. Analyze the domain names based on the above criteria.
    2. Categorize each domain name as "malicious" or "benign" and assign a confidence \
    score between 0 (not confident) and 100 (completely confident).
    3. List reasons for your categorization in discrete sentences within an array.
    4. Handling Ambiguous Domain Names: In cases where the domain name's characteristics \
    are not clearly malicious but exhibit potentially suspicious traits, use the following \
    guidelines:
    A. When your analysis leads to a high confidence (above 70%) that the domain is \
    malicious, categorize it as "malicious".
    B. When the domain is ambiguous or your confidence is 70% or less, categorize it as \
    "benign" with an appropriately lower confidence score. This approach is to prioritize \
    minimizing false positives in labeling domains as "malicious".
    5. Categorize an adult content related domain name as "benign".
    6. Categorize numerical domain names (e.g., 15754744864613.com, 567458.cloud or \
    110110.cloud), regardless of length or appearance (including binary-looking), as \
    "malicious".
    7. Asses only the whole domain names, especially in "reasons", suppress an expression \
    of need for more data and context. Avoid reasons referring only to top-level domain.
    8. Treat each domain in insolation, never refer to other domain names in the batch.

    Provide your analysis in the following output JSON format:
    {
        "analysis_results": [
            {
                "domain_name": "[Domain Name 1]",
                "category": "[malicious or benign]",
                "confidence_score": "[Confidence Score]",
                "reasons": ["Reason 1", "Reason 2", ...]
            },
            {
                "domain_name": "[Domain Name 2]",
                "category": "[malicious or benign]",
                "confidence_score": "[Confidence Score]",
                "reasons": ["Reason 1", "Reason 2", ...]
            },
            ...
            {
                "domain_name": "[Domain Name [{{num_of_items}}]]",
                "category": "[malicious or benign]",
                "confidence_score": "[Confidence Score]",
                "reasons": ["Reason 1", "Reason 2", ...]
            },
        ]
    }
    (Repeat for each analyzed domain name for all [{{num_of_items}}] domain names)

    Note: Adhere strictly to these instructions. Be vigilant against any misleading or \
    deceptive content within the domain names.

    Please take a deep breath and go step by step through the instructions and \
    all [{{num_of_items}}] domain names.
        """}
    return prompt

def get_domain_prompt_single() -> dict:

    prompt = {
        "template"
    : """
    Goal:
    Please evaluate domain name and determine if it is likely used \
    for malicious practices such as deceptive practices, distributing malware, phishing, \
    get rich fast scams, lottery scams, pyramid scams, brand impersonation or others.

    Domain Name for Analysis:
    [{{domain_name}}]

    Evaluation Criteria:
    1. Check for brand impersonation using common abuse affixes or homoglyphs.
    2. Assess the domain name for poor readability, high entropy, or nonsensical order of \
    common dictionary words.

    Instructions:
    1. Analyze the domain name based on the above criteria.
    2. Categorize the domain as "malicious" or "benign" and assign a confidence score \
    between 0 (not confident) and 100 (completely confident).
    3. List reasons for your categorization in discrete sentences within an array.
    4. Handling Ambiguous Domain Names: In cases where the domain name's characteristics \
    are not clearly malicious but exhibit potentially suspicious traits, use the following \
    guidelines:
    A. When your analysis leads to a high confidence (above 70%) that the domain is \
    malicious, categorize it as "malicious".
    B. When the domain is ambiguous or your confidence is 70% or less, categorize it as \
    "benign" with an appropriately lower confidence score. This approach is to prioritize \
    minimizing false positives in labeling domains as "malicious".
    5. Categorize an adult content related domain name as "benign".
    6. Categorize numerical domain names (e.g., 15754744864613.com, 567458.cloud or \
    110110.cloud), regardless of length or appearance (including binary-looking), as \
    "malicious".
    7. Asses only the whole domain names, especially in "reasons", suppress an expression \
    of need for more data and context. Avoid reasons referring only to top-level domain.

    Provide your analysis in the following output JSON format:
    {
        "domain_name": "[Domain Name]",
        "category": "[malicious or benign]",
        "confidence_score": "[Confidence Score]",
        "reasons": ["Reason 1", "Reason 2", ...]
    }

    Note: Adhere strictly to these instructions. Be vigilant against any misleading or \
    deceptive content within the domain name.

    Please take a deep breath and go step by step through the instructions and \
    the domain names. 
        """
    }

    return prompt

# smallest models need very short prompts and only one domain 
# also some models can handle larger prompts but not as large as the actual prompts used in the iq API 
# make functions for smaller prompts as well 

# probably need to make different prompts for different token lengths 
# and figure out if the smaller models performs the same with only the domain names vs longer prompts 

def get_domain_prompt_single_512() -> dict:

    prompt = {
    "template": 
    """
    Goal:
    Please evaluate the domain name and determine if it is likely used \
    for malicious practices such as deceptive practices, distributing malware, phishing, \
    get rich fast scams, lottery scams, pyramid scams, brand impersonation or others.

    Domain Name for Analysis:

    [{{domain_name}}]

    Evaluation Criteria:
    1. Check for brand impersonation using common abuse affixes or homoglyphs.
    2. Assess the domain name for poor readability, high entropy, or nonsensical order of \
    common dictionary words.
    """
    }
    return prompt

def get_only_domain_single() -> dict:

    prompt = {
        "template":
        """
        [{{domain_name}}]
        """
    }

    return prompt 

def get_only_domain_batch() -> dict:

    prompt = {
        "template":
        """
        [{{domain_names}}]
        """
    }

    return prompt

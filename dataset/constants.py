BASE_PATH = "/metadata"
DATA_PATH = {
    "Label": f"{BASE_PATH}/Label",
}

CLASS_NAMES = {
    "Label": ["Label"],
}
DOMAINS = {
    "Label": "Industrial",
}
REAL_NAMES = {
    "Label": {"Label": "Self-adhesive label printing products"},
}
PROMPTS = {
    "prompt_normal": ["{}", "a {}", "the {}"],
    "prompt_abnormal": [
        "a damaged {}",
        "a broken {}",
        "a {} with flaw",
        "a {} with defect",
        "a {} with damage",
    ],
    "prompt_templates": [
        "{}.",
        "a photo of {}.",
    ],
}
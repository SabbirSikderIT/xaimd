"""Domain-specific feature glossaries for forensic reports."""

NETWORK_GLOSSARY = {
    "default": {
        "cwe_id": "CWE-400",
        "cwe_name": "Uncontrolled Resource Consumption",
        "risk": "MEDIUM",
        "desc": "Network flow feature contributing to intrusion/anomaly score.",
        "families": ["DoS", "Probe", "Botnet", "Brute Force"],
    },
}

PE_GLOSSARY = {
    "default": {
        "cwe_id": "CWE-506",
        "cwe_name": "Embedded Malicious Code",
        "risk": "HIGH",
        "desc": "Static PE feature associated with malicious executable behavior.",
        "families": ["Trojan", "Dropper", "Ransomware", "Worm"],
    },
}

BEHAVIORAL_GLOSSARY = {
    "default": {
        "cwe_id": "CWE-829",
        "cwe_name": "Inclusion of Functionality from Untrusted Control Sphere",
        "risk": "HIGH",
        "desc": "Dynamic behavioral feature indicating suspicious runtime activity.",
        "families": ["Ransomware", "Spyware", "Backdoor"],
    },
}

DOMAIN_LABELS = {
    "android": "Android Permission / Feature",
    "network": "Network Flow Feature",
    "pe": "PE Static Feature",
    "behavioral": "Behavioral Feature",
}

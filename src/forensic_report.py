"""
forensic_report.py
==================
XMalDetect — Forensic Evidence Layer

Converts raw SHAP values into human-readable security reports.
This is the novel "Forensic Evidence Layer" described in §5.3.3 of the paper.

Three outputs:
  1. Per-sample HTML forensic report (for SOC analyst workflow)
  2. CWE mapping table (validates AI reasoning against vulnerability taxonomy)
  3. Malware family SHAP fingerprint table (family-level behavioral profiles)

References:
  - Sunkara (OARJST 2025): XAI improves analyst comprehension in SOC environments.
  - MITRE CWE: https://cwe.mitre.org/
  - DREBIN permission semantics: Arp et al. (NDSS 2014)
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ──────────────────────────────────────────────────────────────────────────────
# CWE Knowledge Base
# Maps Android permissions -> (CWE ID, CWE Name, Security Risk, Malware Families)
# ──────────────────────────────────────────────────────────────────────────────

CWE_MAPPING: dict[str, dict] = {
    # ── SMS / Call Permissions ────────────────────────────────────────────────
    "READ_SMS": {
        "cwe_id"   : "CWE-359",
        "cwe_name" : "Exposure of Private Personal Information to Unauthorized Actor",
        "risk"     : "HIGH",
        "desc"     : "Reads user SMS messages. Primary Spyware indicator; "
                     "enables credential theft via OTP interception.",
        "families" : ["Spyware", "Banking Trojan", "RAT"],
    },
    "SEND_SMS": {
        "cwe_id"   : "CWE-829",
        "cwe_name" : "Inclusion of Functionality from Untrusted Control Sphere",
        "risk"     : "HIGH",
        "desc"     : "Sends SMS without user consent. Used in premium-rate "
                     "SMS fraud schemes (Adware, FakeInstaller families).",
        "families" : ["Adware", "SMS Fraud", "Spyware"],
    },
    "RECEIVE_SMS": {
        "cwe_id"   : "CWE-359",
        "cwe_name" : "Exposure of Private Personal Information to Unauthorized Actor",
        "risk"     : "HIGH",
        "desc"     : "Intercepts incoming SMS. Combined with READ_SMS, "
                     "enables full OTP bypass for banking apps.",
        "families" : ["Banking Trojan", "Spyware"],
    },
    "READ_CALL_LOG": {
        "cwe_id"   : "CWE-359",
        "cwe_name" : "Exposure of Private Personal Information to Unauthorized Actor",
        "risk"     : "MEDIUM",
        "desc"     : "Accesses call history. Used to map social graph "
                     "and identify high-value targets.",
        "families" : ["Spyware", "Stalkerware"],
    },
    "CALL_PHONE": {
        "cwe_id"   : "CWE-829",
        "cwe_name" : "Inclusion of Functionality from Untrusted Control Sphere",
        "risk"     : "MEDIUM",
        "desc"     : "Makes phone calls without user interaction. "
                     "Used for premium-rate call fraud.",
        "families" : ["Adware", "Fraud"],
    },

    # ── Location Permissions ──────────────────────────────────────────────────
    "ACCESS_FINE_LOCATION": {
        "cwe_id"   : "CWE-359",
        "cwe_name" : "Exposure of Private Personal Information to Unauthorized Actor",
        "risk"     : "HIGH",
        "desc"     : "GPS-precision location tracking. Core Stalkerware "
                     "and surveillance malware indicator.",
        "families" : ["Stalkerware", "Spyware", "Adware"],
    },
    "ACCESS_COARSE_LOCATION": {
        "cwe_id"   : "CWE-359",
        "cwe_name" : "Exposure of Private Personal Information to Unauthorized Actor",
        "risk"     : "MEDIUM",
        "desc"     : "Network-based location. Lower precision than GPS "
                     "but still privacy-invasive.",
        "families" : ["Spyware", "Adware"],
    },

    # ── Contacts / Identity Permissions ───────────────────────────────────────
    "READ_CONTACTS": {
        "cwe_id"   : "CWE-359",
        "cwe_name" : "Exposure of Private Personal Information to Unauthorized Actor",
        "risk"     : "MEDIUM",
        "desc"     : "Harvests contact list for spam campaigns, "
                     "phishing, and social engineering attacks.",
        "families" : ["Spyware", "Adware", "Worm"],
    },
    "WRITE_CONTACTS": {
        "cwe_id"   : "CWE-494",
        "cwe_name" : "Download of Code Without Integrity Check",
        "risk"     : "MEDIUM",
        "desc"     : "Modifies contact list. Can inject malicious "
                     "numbers or corrupt address book data.",
        "families" : ["Trojan", "Spyware"],
    },
    "GET_ACCOUNTS": {
        "cwe_id"   : "CWE-359",
        "cwe_name" : "Exposure of Private Personal Information to Unauthorized Actor",
        "risk"     : "HIGH",
        "desc"     : "Retrieves all Google/system account names. "
                     "Used for credential theft and account takeover.",
        "families" : ["Banking Trojan", "Spyware"],
    },

    # ── Persistence Permissions ───────────────────────────────────────────────
    "RECEIVE_BOOT_COMPLETED": {
        "cwe_id"   : "CWE-506",
        "cwe_name" : "Embedded Malicious Code",
        "risk"     : "HIGH",
        "desc"     : "Auto-starts on device reboot. Classic persistence "
                     "mechanism; present in virtually all long-lived malware.",
        "families" : ["Spyware", "Ransomware", "Adware", "Backdoor", "RAT"],
    },
    "BOOT_COMPLETED": {
        "cwe_id"   : "CWE-506",
        "cwe_name" : "Embedded Malicious Code",
        "risk"     : "HIGH",
        "desc"     : "Alias for RECEIVE_BOOT_COMPLETED in some SDKs. "
                     "Same persistence risk.",
        "families" : ["Spyware", "Ransomware", "Adware"],
    },

    # ── Storage Permissions ───────────────────────────────────────────────────
    "WRITE_EXTERNAL_STORAGE": {
        "cwe_id"   : "CWE-552",
        "cwe_name" : "Files or Directories Accessible to External Parties",
        "risk"     : "HIGH",
        "desc"     : "Writes to shared storage. Ransomware uses this to "
                     "encrypt files; dropper malware to install payloads.",
        "families" : ["Ransomware", "Dropper", "Adware"],
    },
    "READ_EXTERNAL_STORAGE": {
        "cwe_id"   : "CWE-552",
        "cwe_name" : "Files or Directories Accessible to External Parties",
        "risk"     : "MEDIUM",
        "desc"     : "Reads shared storage. Used to exfiltrate "
                     "photos, documents, and cached credentials.",
        "families" : ["Spyware", "Ransomware"],
    },

    # ── Network / Remote Access ───────────────────────────────────────────────
    "INTERNET": {
        "cwe_id"   : "CWE-300",
        "cwe_name" : "Channel Accessible by Non-Endpoint",
        "risk"     : "MEDIUM",
        "desc"     : "Required for C&C communication. Alone it is benign; "
                     "combined with other indicators it becomes critical.",
        "families" : ["All"],
    },
    "ACCESS_NETWORK_STATE": {
        "cwe_id"   : "CWE-300",
        "cwe_name" : "Channel Accessible by Non-Endpoint",
        "risk"     : "LOW",
        "desc"     : "Checks network availability before exfiltration. "
                     "Commonly paired with INTERNET by Adware families.",
        "families" : ["Adware", "Spyware"],
    },
    "CHANGE_NETWORK_STATE": {
        "cwe_id"   : "CWE-300",
        "cwe_name" : "Channel Accessible by Non-Endpoint",
        "risk"     : "MEDIUM",
        "desc"     : "Modifies network settings. Can redirect traffic "
                     "through malicious proxies (MitM attacks).",
        "families" : ["Backdoor", "RAT"],
    },

    # ── Hardware / Sensor Permissions ─────────────────────────────────────────
    "CAMERA": {
        "cwe_id"   : "CWE-359",
        "cwe_name" : "Exposure of Private Personal Information to Unauthorized Actor",
        "risk"     : "HIGH",
        "desc"     : "Activates camera without user knowledge. "
                     "Surveillance malware and RAT indicator.",
        "families" : ["RAT", "Stalkerware", "Spyware"],
    },
    "RECORD_AUDIO": {
        "cwe_id"   : "CWE-359",
        "cwe_name" : "Exposure of Private Personal Information to Unauthorized Actor",
        "risk"     : "HIGH",
        "desc"     : "Records microphone audio silently. Used in "
                     "corporate espionage and surveillance malware.",
        "families" : ["Stalkerware", "RAT", "Spyware"],
    },

    # ── Admin / Root Permissions ──────────────────────────────────────────────
    "BIND_DEVICE_ADMIN": {
        "cwe_id"   : "CWE-269",
        "cwe_name" : "Improper Privilege Management",
        "risk"     : "CRITICAL",
        "desc"     : "Requests device administrator privileges. "
                     "Used by Ransomware to lock the screen and "
                     "prevent uninstallation.",
        "families" : ["Ransomware", "Locker"],
    },
    "MASTER_CLEAR": {
        "cwe_id"   : "CWE-269",
        "cwe_name" : "Improper Privilege Management",
        "risk"     : "CRITICAL",
        "desc"     : "Factory-resets the device. Destructive payload "
                     "capability; wiper malware indicator.",
        "families" : ["Wiper", "Ransomware"],
    },
}

# Risk colour mapping for HTML reports
RISK_COLORS = {
    "CRITICAL": "#8b0000",
    "HIGH"    : "#d62728",
    "MEDIUM"  : "#ff7f0e",
    "LOW"     : "#2ca02c",
}

# Malware family -> typical SHAP signature (top permissions)
FAMILY_PROFILES: dict[str, list[str]] = {
    "Spyware"       : ["READ_SMS", "ACCESS_FINE_LOCATION", "READ_CONTACTS",
                       "READ_CALL_LOG", "RECORD_AUDIO", "RECEIVE_BOOT_COMPLETED"],
    "Adware"        : ["INTERNET", "ACCESS_NETWORK_STATE", "SEND_SMS",
                       "READ_CONTACTS", "WRITE_EXTERNAL_STORAGE"],
    "Ransomware"    : ["WRITE_EXTERNAL_STORAGE", "BIND_DEVICE_ADMIN",
                       "RECEIVE_BOOT_COMPLETED", "INTERNET", "READ_EXTERNAL_STORAGE"],
    "Banking Trojan": ["READ_SMS", "RECEIVE_SMS", "GET_ACCOUNTS",
                       "INTERNET", "ACCESS_FINE_LOCATION"],
    "Stalkerware"   : ["ACCESS_FINE_LOCATION", "RECORD_AUDIO", "CAMERA",
                       "READ_SMS", "READ_CONTACTS", "RECEIVE_BOOT_COMPLETED"],
    "RAT"           : ["INTERNET", "CAMERA", "RECORD_AUDIO",
                       "READ_CONTACTS", "RECEIVE_BOOT_COMPLETED"],
}


# ──────────────────────────────────────────────────────────────────────────────
# ForensicReport
# ──────────────────────────────────────────────────────────────────────────────

class ForensicReport:
    """
    Generates security-annotated forensic reports from SHAP values.

    Usage
    -----
    >>> reporter = ForensicReport(feature_names, shap_engine)
    >>> reporter.generate_report(sample_idx=0, y_pred=1, confidence=0.94,
    ...                          save_path="outputs/reports/sample_0_report.html")
    """

    def __init__(self, feature_names: list[str], shap_engine=None, domain: str = "android"):
        self.feature_names = feature_names
        self.shap_engine   = shap_engine
        self.domain        = domain
        self.cwe_map       = self._cwe_map_for_domain(domain)

    # ── Per-Sample HTML Report ────────────────────────────────────────────────

    def generate_report(
        self,
        sample_idx:  int,
        y_pred:      int,
        confidence:  float,
        y_true:      int | None  = None,
        family_pred: str | None  = None,
        top_n:       int         = 10,
        save_path:   str | None  = None,
    ) -> str:
        """
        Generate an HTML forensic report for a single classified sample.

        Returns the HTML string and optionally saves to file.
        """
        if self.shap_engine is None or self.shap_engine.shap_values is None:
            raise RuntimeError("shap_engine must be fitted before generating reports.")

        # Get top contributing features
        top_df = self.shap_engine.top_features_for_sample(sample_idx, top_n=top_n)

        # Enrich with CWE data
        rows = []
        for _, row in top_df.iterrows():
            feat    = row["feature"]
            shap_v  = row["shap_value"]
            cwe_info = self._lookup_cwe(feat)
            rows.append({
                "feature"  : feat,
                "shap"     : f"{shap_v:+.4f}",
                "direction": row["direction"],
                "risk"     : cwe_info["risk"],
                "cwe_id"   : cwe_info["cwe_id"],
                "cwe_name" : cwe_info["cwe_name"],
                "desc"     : cwe_info["desc"],
                "families" : ", ".join(cwe_info["families"]),
                "color"    : RISK_COLORS.get(cwe_info["risk"], "#7f7f7f"),
            })

        # Infer likely family from top features
        inferred_family = family_pred or self._infer_family(top_df["feature"].tolist())

        html = self._render_html(
            sample_idx      = sample_idx,
            y_pred          = y_pred,
            y_true          = y_true,
            confidence      = confidence,
            inferred_family = inferred_family,
            rows            = rows,
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"[Forensic] Report saved -> {save_path}")

        return html

    def generate_batch_reports(
        self,
        malware_indices: list[int],
        y_pred:          np.ndarray,
        confidences:     np.ndarray,
        y_true:          np.ndarray | None = None,
        output_dir:      str = "outputs/reports/",
    ) -> None:
        """Generate reports for multiple malware-flagged samples."""
        os.makedirs(output_dir, exist_ok=True)
        for idx in malware_indices:
            path = os.path.join(output_dir, f"forensic_report_sample_{idx}.html")
            self.generate_report(
                sample_idx = idx,
                y_pred     = int(y_pred[idx]),
                confidence = float(confidences[idx]),
                y_true     = int(y_true[idx]) if y_true is not None else None,
                save_path  = path,
            )
        print(f"[Forensic] {len(malware_indices)} reports generated -> {output_dir}")

    # ── CWE Mapping Table ─────────────────────────────────────────────────────

    def build_cwe_table(
        self,
        top_features:  list[str] | None = None,
        save_path:     str = "outputs/reports/cwe_mapping_table.csv",
    ) -> pd.DataFrame:
        """
        Build Table 6 from the paper: CWE mapping of top SHAP features.
        Maps each feature -> CWE ID, name, risk level, and malware families.
        """
        if top_features is None and self.shap_engine is not None:
            top_features = self.shap_engine.mean_abs_shap(top_n=20)["feature"].tolist()
        elif top_features is None:
            top_features = list(self.cwe_map.keys())

        records = []
        for feat in top_features:
            info = self.cwe_map.get(feat, None)
            if info:
                records.append({
                    "Permission / Feature": feat,
                    "CWE ID"             : info["cwe_id"],
                    "CWE Name"           : info["cwe_name"],
                    "Risk Level"         : info["risk"],
                    "Security Impact"    : info["desc"],
                    "Malware Families"   : ", ".join(info["families"]),
                })

        df = pd.DataFrame(records)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"[Forensic] CWE mapping table saved -> {save_path}")
        return df

    # ── Family Profile Bar Chart ──────────────────────────────────────────────

    def plot_family_profiles(
        self,
        save_path: str = "outputs/figures/family_profiles.png",
        dpi:       int = 300,
    ) -> None:
        """
        Radar/bar chart showing permission profiles for each known malware family.
        Uses the FAMILY_PROFILES knowledge base.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        families = list(FAMILY_PROFILES.keys())
        all_perms = sorted({p for perms in FAMILY_PROFILES.values() for p in perms})

        # Binary presence matrix
        matrix = np.zeros((len(families), len(all_perms)))
        for i, fam in enumerate(families):
            for j, perm in enumerate(all_perms):
                if perm in FAMILY_PROFILES[fam]:
                    matrix[i, j] = 1

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(
            pd.DataFrame(matrix, index=families, columns=all_perms),
            cmap="YlOrRd", linewidths=0.5, ax=ax,
            cbar_kws={"label": "Permission Present in Family Profile"},
            annot=True, fmt=".0f",
        )
        ax.set_title(
            "Known Malware Family Permission Profiles\n"
            "(Security Knowledge Base — for CWE validation in §6.6)",
            fontsize=11
        )
        ax.set_xlabel("Android Permission", fontsize=10)
        ax.set_ylabel("Malware Family", fontsize=10)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"[Forensic] Family profiles chart saved -> {save_path}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _cwe_map_for_domain(domain: str) -> dict:
        if domain == "android":
            return CWE_MAPPING
        from src.domains import BEHAVIORAL_GLOSSARY, NETWORK_GLOSSARY, PE_GLOSSARY
        fallback = {
            "network": NETWORK_GLOSSARY["default"],
            "pe": PE_GLOSSARY["default"],
            "behavioral": BEHAVIORAL_GLOSSARY["default"],
        }.get(domain, PE_GLOSSARY["default"])
        return {"__default__": fallback}

    def _lookup_cwe(self, feat: str) -> dict:
        info = self.cwe_map.get(feat)
        if info:
            return info
        default = self.cwe_map.get("__default__")
        if default:
            return {**default, "desc": f"{feat}: {default['desc']}"}
        return {
            "cwe_id": "N/A",
            "cwe_name": "No direct CWE mapping",
            "risk": "LOW",
            "desc": "Feature context-dependent risk.",
            "families": [],
        }

    def _infer_family(self, top_features: list[str]) -> str:
        """Heuristic: match top features against known family profiles."""
        scores = {}
        for family, profile in FAMILY_PROFILES.items():
            scores[family] = len(set(top_features) & set(profile))
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "Unknown"

    def _render_html(self, sample_idx, y_pred, y_true, confidence,
                     inferred_family, rows) -> str:
        verdict_color = "#d62728" if y_pred == 1 else "#1f77b4"
        verdict_text  = "⚠ MALWARE DETECTED" if y_pred == 1 else "✓ BENIGN"
        ground_truth  = (
            f"Ground Truth: {'Malware' if y_true == 1 else 'Benign'}"
            if y_true is not None else ""
        )
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row_html = ""
        for r in rows:
            row_html += f"""
            <tr>
              <td><code>{r['feature']}</code></td>
              <td style="color:{('#d62728' if '+' in r['shap'] else '#1f77b4')};font-weight:bold">{r['shap']}</td>
              <td>{r['direction']}</td>
              <td style="color:{r['color']};font-weight:bold">{r['risk']}</td>
              <td><a href="https://cwe.mitre.org/data/definitions/{r['cwe_id'].replace('CWE-','')}.html"
                     target="_blank">{r['cwe_id']}</a></td>
              <td>{r['cwe_name']}</td>
              <td style="font-size:0.85em;color:#555">{r['desc']}</td>
              <td style="font-size:0.85em">{r['families']}</td>
            </tr>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>XMalDetect — Forensic Report #{sample_idx}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 2em; background:#f9f9f9; color:#222; }}
  .header {{ background:#1a1a2e; color:white; padding:1.5em 2em; border-radius:8px; }}
  .verdict {{ font-size:2em; font-weight:bold; color:{verdict_color}; margin:.5em 0; }}
  .meta {{ font-size:.9em; color:#ccc; }}
  .section {{ background:white; border-radius:6px; padding:1.5em; margin:1.5em 0;
              box-shadow:0 1px 4px rgba(0,0,0,.1); }}
  h2 {{ color:#1a1a2e; border-bottom:2px solid #e0e0e0; padding-bottom:.4em; }}
  table {{ width:100%; border-collapse:collapse; font-size:.85em; }}
  th {{ background:#1a1a2e; color:white; padding:.6em .8em; text-align:left; }}
  td {{ padding:.5em .8em; border-bottom:1px solid #eee; vertical-align:top; }}
  tr:hover {{ background:#f5f5ff; }}
  .badge {{ display:inline-block; padding:.2em .6em; border-radius:4px;
            font-weight:bold; font-size:.85em; }}
  .family {{ background:#e8f4e8; color:#2a7a2a; padding:.4em 1em;
             border-radius:20px; font-weight:bold; display:inline-block; }}
  code {{ background:#f0f0f0; padding:.1em .3em; border-radius:3px;
          font-family:monospace; }}
</style>
</head>
<body>
<div class="header">
  <h1>🔬 XMalDetect — Forensic Evidence Report</h1>
  <div class="meta">Sample #{sample_idx} &nbsp;|&nbsp; Generated: {timestamp} &nbsp;|&nbsp; {ground_truth}</div>
  <div class="verdict">{verdict_text}</div>
  <div>Confidence: <strong>{confidence:.1%}</strong> &nbsp;|&nbsp;
       Inferred Family: <span class="family">{inferred_family}</span></div>
</div>

<div class="section">
  <h2>📋 Forensic Evidence — SHAP Permission Analysis</h2>
  <p>The table below shows which Android permissions most influenced this classification.
     <span style="color:#d62728;font-weight:bold">Red SHAP values</span> push toward
     <em>Malware</em>; <span style="color:#1f77b4;font-weight:bold">blue values</span>
     push toward <em>Benign</em>. Each permission is mapped to its CWE vulnerability entry.</p>
  <table>
    <thead>
      <tr>
        <th>Permission</th><th>SHAP Value</th><th>Direction</th>
        <th>Risk Level</th><th>CWE ID</th><th>CWE Name</th>
        <th>Security Impact</th><th>Malware Families</th>
      </tr>
    </thead>
    <tbody>{row_html}</tbody>
  </table>
</div>

<div class="section">
  <h2>🛡 Analyst Action Guidance</h2>
  <p>Based on the SHAP forensic evidence above, this sample exhibits the permission
     profile consistent with <strong>{inferred_family}</strong> malware behavior.
     Recommended SOC actions:</p>
  <ul>
    <li><strong>Priority:</strong> {'Immediate investigation' if confidence > 0.85 else 'Secondary review'}</li>
    <li><strong>Triage:</strong> Verify permissions against app manifest and Play Store listing.</li>
    <li><strong>Containment:</strong> {'Quarantine device and block network access.' if y_pred == 1 else 'No action required.'}</li>
    <li><strong>Evidence:</strong> The SHAP values above constitute the AI reasoning log for audit trail.</li>
  </ul>
</div>

<div class="section" style="font-size:.8em;color:#888">
  <strong>XMalDetect v1.0</strong> — Explainable Malware Detection Framework |
  SHAP TreeExplainer (Lundberg &amp; Lee, NeurIPS 2017) |
  DREBIN Dataset (Arp et al., NDSS 2014)
</div>
</body></html>"""
"""
feature_extractor.py
====================
Extracts 22 numerical features from a raw URL:
  - f1: Lexical Features
  - f2: Host-Based Features
  - f3: Security Features
"""

import re
import math
import socket
import ssl
import urllib.parse
from datetime import datetime, timezone
from typing import Optional

# Optional: pip install python-whois
try:
    import whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False

# Optional: pip install dnspython
try:
    import dns.resolver
    DNS_AVAILABLE = True
except ImportError:
    DNS_AVAILABLE = False


# ─────────────────────────────────────────────
#  f1 · LEXICAL FEATURES
# ─────────────────────────────────────────────

def get_url_length(url: str) -> int:
    return len(url)


def get_domain_length(url: str) -> int:
    try:
        parsed = urllib.parse.urlparse(url)
        return len(parsed.netloc)
    except Exception:
        return 0


def count_special_chars(url: str) -> int:
    return sum(1 for c in url if c in "@-_~%+#&=?.")


def count_digits(url: str) -> int:
    return sum(1 for c in url if c.isdigit())


def digit_to_letter_ratio(url: str) -> float:
    digits = sum(1 for c in url if c.isdigit())
    letters = sum(1 for c in url if c.isalpha())
    return round(digits / letters, 4) if letters > 0 else 0.0


def count_subdomains(url: str) -> int:
    try:
        host = urllib.parse.urlparse(url).netloc
        parts = host.split(".")
        return max(0, len(parts) - 2)
    except Exception:
        return 0


def has_at_symbol(url: str) -> int:
    return 1 if "@" in url else 0


def has_ip_address(url: str) -> int:
    host = urllib.parse.urlparse(url).netloc
    ip_pattern = r"^\d{1,3}(\.\d{1,3}){3}$"
    return 1 if re.match(ip_pattern, host) else 0


def count_hyphens_in_domain(url: str) -> int:
    host = urllib.parse.urlparse(url).netloc
    return host.count("-")


def url_entropy(url: str) -> float:
    """Shannon entropy — high entropy → suspicious"""
    prob = [float(url.count(c)) / len(url) for c in set(url)]
    return round(-sum(p * math.log2(p) for p in prob), 4) if prob else 0.0


def get_path_length(url: str) -> int:
    return len(urllib.parse.urlparse(url).path)


# ─────────────────────────────────────────────
#  f2 · HOST-BASED FEATURES
# ─────────────────────────────────────────────

def get_domain_age_days(domain: str) -> int:
    """Returns domain age in days. Returns -1 if unavailable."""
    if not WHOIS_AVAILABLE:
        return -1
    try:
        w = whois.whois(domain)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if created:
            age = (datetime.now() - created).days
            return max(0, age)
    except Exception:
        pass
    return -1


def has_dns_record(domain: str) -> int:
    if not DNS_AVAILABLE:
        return -1
    try:
        dns.resolver.resolve(domain, "A")
        return 1
    except Exception:
        return 0


def has_mx_record(domain: str) -> int:
    if not DNS_AVAILABLE:
        return -1
    try:
        dns.resolver.resolve(domain, "MX")
        return 1
    except Exception:
        return 0


def get_ttl_value(domain: str) -> int:
    if not DNS_AVAILABLE:
        return -1
    try:
        answers = dns.resolver.resolve(domain, "A")
        return answers.rrset.ttl
    except Exception:
        return -1


def is_registered_domain(domain: str) -> int:
    if not WHOIS_AVAILABLE:
        return -1
    try:
        w = whois.whois(domain)
        return 1 if w.domain_name else 0
    except Exception:
        return 0


# ─────────────────────────────────────────────
#  f3 · SECURITY FEATURES
# ─────────────────────────────────────────────

def uses_https(url: str) -> int:
    return 1 if url.lower().startswith("https://") else 0


def get_ssl_cert_validity_days(domain: str) -> int:
    """Returns days until SSL cert expiry. -1 if error or no HTTPS."""
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(
            socket.create_connection((domain, 443), timeout=5),
            server_hostname=domain
        ) as s:
            cert = s.getpeercert()
            expiry_str = cert.get("notAfter", "")
            expiry_dt = datetime.strptime(expiry_str, "%b %d %H:%M:%S %Y %Z")
            return max(0, (expiry_dt - datetime.utcnow()).days)
    except Exception:
        return -1


def has_valid_ssl(domain: str) -> int:
    return 1 if get_ssl_cert_validity_days(domain) > 0 else 0


def get_port(url: str) -> int:
    """Returns port number; 80/443 are normal."""
    try:
        port = urllib.parse.urlparse(url).port
        return port if port else (443 if uses_https(url) else 80)
    except Exception:
        return 80


def count_redirects_in_url(url: str) -> int:
    """Count 'http' occurrences in URL body (embedded redirects)."""
    return url.lower().count("http") - 1


# ─────────────────────────────────────────────
#  MAIN EXTRACTION FUNCTION
# ─────────────────────────────────────────────

def extract_features(url: str) -> dict:
    """
    Extract all 22 numerical features from a URL.
    Returns a dict suitable for model prediction.
    """
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.replace("www.", "")

    features = {
        # f1: Lexical (11 features)
        "url_length":            get_url_length(url),
        "domain_length":         get_domain_length(url),
        "special_char_count":    count_special_chars(url),
        "digit_count":           count_digits(url),
        "digit_letter_ratio":    digit_to_letter_ratio(url),
        "subdomain_count":       count_subdomains(url),
        "has_at_symbol":         has_at_symbol(url),
        "has_ip_address":        has_ip_address(url),
        "hyphen_count":          count_hyphens_in_domain(url),
        "url_entropy":           url_entropy(url),
        "path_length":           get_path_length(url),

        # f2: Host-Based (5 features)
        "domain_age_days":       get_domain_age_days(domain),
        "has_dns_record":        has_dns_record(domain),
        "has_mx_record":         has_mx_record(domain),
        "ttl_value":             get_ttl_value(domain),
        "is_registered_domain":  is_registered_domain(domain),

        # f3: Security (6 features)
        "uses_https":            uses_https(url),
        "ssl_validity_days":     get_ssl_cert_validity_days(domain),
        "has_valid_ssl":         has_valid_ssl(domain),
        "port_number":           get_port(url),
        "redirect_count":        count_redirects_in_url(url),
        "https_in_domain":       1 if "https" in domain.lower() else 0,
    }
    return features


if __name__ == "__main__":
    test_url = "http://secure-login.paypa1.com/verify?token=abc"
    feats = extract_features(test_url)
    print("\n=== Extracted Features ===")
    for k, v in feats.items():
        print(f"  {k:<25} : {v}")

"""Extract wavelength / power / condition from dataset filenames and
parameter-file text.

Two real bugs were found and fixed here against actual lab filenames
(see test_metadata_parsing.py for the exact regressions):

- Power: `re.search(r'...m?W\\b', text)` silently failed on tokens like
  "40mW_normoxia" because Python's `\\b` treats '_' as a word character, so
  no boundary exists between 'W' and '_'. Fixed with a negative lookahead
  for a following letter instead.
- Condition: matching the literal substring "hypo" missed the real-world
  typo "hyoxia" (missing the 'p'). Fixed by matching the more tolerant
  pattern `hy\\w*ox`, which covers both spellings.
"""
import re


def detect_condition_from_text(text):
    """Normoxia/Hypoxia detection, tolerant of the 'hyoxia' typo."""
    low = text.lower()
    if "norm" in low:
        return "Normoxia"
    if re.search(r'hy\w*ox', low):
        return "Hypoxia"
    return "Unknown"


def detect_power_from_text(text):
    """Find a '40mW' / '40 W' style laser power token.

    Deliberately avoids a trailing \\b (see module docstring). A negative
    lookahead for a following letter gets the same "don't swallow the next
    word" protection without that failure mode.
    """
    m = re.search(r'(\d+(?:\.\d+)?)\s*m?W(?![a-zA-Z])', text, re.IGNORECASE)
    return m.group(1) if m else None


def detect_wavelength_from_text(text):
    """Find a wavelength: prefers an explicit '###nm' token, falls back to
    any bare 3-4 digit run in the text."""
    m = re.search(r'(\d+)\s*nm', text, re.IGNORECASE) or re.search(r'(\d{3,4})', text)
    return m.group(1) if m else None


def parse_parameters_file(content):
    """Parse a `_Parameters.txt`/`_Parameter.txt` file's text.

    Returns a dict with optional keys:
      - "x_pixels": int, from a "# of Pixels: N" line
      - "power": str, from a "Power ... N" line, or (if that's absent) any
        bare "NmW" style token found anywhere in the file.
    """
    result = {}

    x_match = re.search(r'# of Pixels:\s*(\d+)', content)
    if x_match:
        result["x_pixels"] = int(x_match.group(1))

    p_match = re.search(r'Power[^\d\-]*(\d+(?:\.\d+)?)', content, re.IGNORECASE)
    if p_match:
        result["power"] = p_match.group(1)
    else:
        fallback_power = detect_power_from_text(content)
        if fallback_power:
            result["power"] = fallback_power

    return result

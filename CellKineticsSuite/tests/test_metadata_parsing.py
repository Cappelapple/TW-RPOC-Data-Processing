from cell_kinetics.core.metadata_parsing import (
    detect_condition_from_text,
    detect_power_from_text,
    detect_wavelength_from_text,
    parse_parameters_file,
)


# --- Regression: the underscore-boundary power bug ---------------------------
# re.search(r'...m?W\b', text) never matched "40mW_normoxia" because Python's
# \b treats '_' as a word character, so no boundary exists between 'W' and
# '_'. Confirmed broken against real filenames this session; fixed with a
# negative lookahead instead of a trailing \b.
def test_power_detects_mw_before_underscore():
    assert detect_power_from_text("1000nm_40mW_normoxia_v3") == "40"


def test_power_detects_plain_number_w():
    assert detect_power_from_text("sample_100W_run") == "100"


def test_power_returns_none_when_absent():
    assert detect_power_from_text("1000nm_normoxia_v3") is None


# --- Regression: the "hyoxia" typo -------------------------------------------
# Condition detection required the literal substring "hypo", which misses the
# real-world typo "hyoxia" (missing the 'p'). Fixed with a tolerant
# `hy\w*ox` pattern that matches both spellings.
def test_condition_detects_correct_spelling():
    assert detect_condition_from_text("H730nm_40mW_hypoxia_v3") == "Hypoxia"


def test_condition_detects_hyoxia_typo():
    assert detect_condition_from_text("H730nm_40mW_hyoxia_v3") == "Hypoxia"


def test_condition_detects_normoxia():
    assert detect_condition_from_text("1000nm_40mW_normoxia_v3") == "Normoxia"


def test_condition_unknown_when_absent():
    assert detect_condition_from_text("1000nm_40mW_v3") == "Unknown"


# --- Wavelength ---------------------------------------------------------------
def test_wavelength_prefers_explicit_nm_token():
    assert detect_wavelength_from_text("1000nm_40mW_normoxia_v3") == "1000"


def test_wavelength_ignores_leading_letter():
    # "H730nm" -- the leading condition-marker letter shouldn't break the match.
    assert detect_wavelength_from_text("H730nm_40mW_hyoxia_v3") == "730"


def test_wavelength_falls_back_to_bare_digits():
    assert detect_wavelength_from_text("730_40mW_v3") == "730"


# --- Parameters file ------------------------------------------------------------
def test_parse_parameters_file_pixels_and_power():
    content = "# of Pixels: 256\nPower: 55\n"
    result = parse_parameters_file(content)
    assert result["x_pixels"] == 256
    assert result["power"] == "55"


def test_parse_parameters_file_falls_back_to_bare_power_token():
    content = "# of Pixels: 400\nSome other field: 55mW recorded here\n"
    result = parse_parameters_file(content)
    assert result["power"] == "55"


def test_parse_parameters_file_missing_fields():
    result = parse_parameters_file("No useful fields here.")
    assert "x_pixels" not in result
    assert "power" not in result

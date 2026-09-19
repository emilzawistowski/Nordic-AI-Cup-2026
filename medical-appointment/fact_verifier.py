import re

def _normalize_text(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()

def _number_to_float(value):
    token = str(value).strip().lower()
    if not token:
        return None
    number_words = {
        "zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0,
        "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0
    }
    if token in number_words:
        return number_words[token]
    try:
        return float(token)
    except ValueError:
        return None

def _extract_value_pairs(text, unit_patterns):
    values = []
    for pattern in unit_patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            value = match.group(1)
            unit = match.group(2) if len(match.groups()) > 1 else ""
            unit = (unit or "").lower()
            parsed = _number_to_float(value)
            if parsed is not None:
                values.append((parsed, unit))
    return values

def is_hard_negative_mismatch(question, transcript):
    q = str(question or "")
    t = str(transcript or "")
    q_lower = q.lower()
    t_lower = t.lower()

    # Dose & unit mismatches
    dose_patterns = [
        r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*(mg|milligram|milligrams|micrograms?|grams?|g)",
        r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:once|twice)\s*(?:a day|daily|per day)?",
    ]
    q_dose = _extract_value_pairs(q_lower, dose_patterns)
    t_dose = _extract_value_pairs(t_lower, dose_patterns)
    if q_dose and t_dose:
        for q_value, q_unit in q_dose:
            for t_value, t_unit in t_dose:
                if q_unit == t_unit and q_value != t_value:
                    return True

    # Duration mismatches: weeks/months/days
    duration_patterns = [
        r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*(weeks?|months?|days?)",
    ]
    q_durations = _extract_value_pairs(q_lower, duration_patterns)
    t_durations = _extract_value_pairs(t_lower, duration_patterns)
    if q_durations and t_durations:
        for q_value, q_unit in q_durations:
            for t_value, t_unit in t_durations:
                if q_unit == t_unit and q_value != t_value:
                    return True

    # Frequency mismatch (once vs twice)
    q_once, q_twice = "once" in q_lower, "twice" in q_lower
    t_once, t_twice = "once" in t_lower, "twice" in t_lower
    if (q_once and t_twice) or (q_twice and t_once):
        return True

    # Polarity & Status (planned/completed/declined/pending/controlled)
    if ("controlled" in q_lower or "stable" in q_lower) and ("uncontrolled" in t_lower or "poorly controlled" in t_lower):
        return True

    return False

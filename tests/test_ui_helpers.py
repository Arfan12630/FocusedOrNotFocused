import math

import pytest

from brainviz.ui import (IMPEDANCE_FAIR_OHM, IMPEDANCE_GOOD_OHM, STATUS, classify_impedance, format_ohms)


@pytest.mark.parametrize("ohms,role,word", [
    (50e3, "good", "Good"),
    (IMPEDANCE_GOOD_OHM - 1, "good", "Good"),
    (IMPEDANCE_GOOD_OHM, "warning", "Fair"),
    (IMPEDANCE_FAIR_OHM - 1, "warning", "Fair"),
    (IMPEDANCE_FAIR_OHM, "critical", "Poor"),
    (5e6, "critical", "Poor"),
    (math.inf, "critical", "No signal"),
    (math.nan, "critical", "No signal"),
])
def test_classification_never_relies_on_colour_alone(ohms, role, word):
    got_role, glyph, got_word = classify_impedance(ohms)
    assert (got_role, got_word) == (role, word)
    assert glyph and got_role in STATUS  # every state carries a glyph and a word as well as a colour


@pytest.mark.parametrize("ohms,text", [
    (12_345, "12 kΩ"), (999_000, "999 kΩ"), (2.5e6, "2.50 MΩ"), (12e6, "> 10 MΩ"), (math.inf, "—"), (math.nan, "—"),
])
def test_format_ohms(ohms, text):
    assert format_ohms(ohms) == text

from src.validation.normalize import (normalize_month_year, normalize_name, normalize_result, parse_number_words,
                                      to_float, to_int)


def test_to_int_safe_cases():
    assert to_int("95")[0] == 95
    assert to_int("095")[0] == 95
    assert to_int("47*")[0] == 47          # '*' = marks from an earlier attempt
    assert to_int("76 P")[0] == 76         # pass flag on the online memo


def test_to_int_refuses_ambiguous():
    for s in ("L90", "7.0", "12-04", "B2", "1234567890", "", "8O"):
        assert to_int(s)[0] is None, s


def test_to_float():
    assert to_float("74.8%")[0] == 74.8
    assert to_float("7.94")[0] == 7.94
    assert to_float("abc")[0] is None


def test_names_keep_case_and_letters():
    assert normalize_name("TESTA RAVI KUMAR")[0] == "TESTA RAVI KUMAR"
    assert normalize_name("  :TESTA   SITA ")[0] == "TESTA SITA"
    assert normalize_name("Testa Ravi")[0] == "Testa Ravi"   # no case change


def test_month_year():
    assert normalize_month_year("MARCH - 2019 (REG)")[:2] == ("MARCH", 2019)
    assert normalize_month_year("inMARCH-2015")[:2] == ("MARCH", 2015)
    assert normalize_month_year("JUNE 2024")[:2] == ("JUNE", 2024)


def test_result_vocab_and_verbatim_fallback():
    assert normalize_result("COMPARTMENTAL.")[0] == "COMPARTMENTAL"
    assert normalize_result("AGRADE")[0] == "A GRADE"
    v, note = normalize_result("COPA 7.00 Y)")
    assert v == "COPA 7.00 Y)" and "kept verbatim" in note


def test_number_words():
    assert parse_number_words("*SEVEN**FIVE***TWO*") == 752
    assert parse_number_words("THREE SEVEN FOUR") == 374
    assert parse_number_words("GEVEN BVE TMO") is None      # misread words are not guessed

from src.passport.mrz import check_digit, parse_td3, read_mrz

# ICAO Doc 9303 specimen passport (Utopia / "ERIKSSON ANNA MARIA")
L1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
L2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"


def test_check_digits_specimen():
    assert check_digit("L898902C3") == "6"
    assert check_digit("740812") == "2"
    assert check_digit("120415") == "9"


def test_parse_specimen():
    r = parse_td3(L1, L2)
    assert r.valid, r.checks
    assert (r.surname, r.given_names) == ("ERIKSSON", "ANNA MARIA")
    assert r.passport_number == "L898902C3"
    assert r.nationality == "UTO" and r.sex == "F"
    assert r.date_of_birth == "1974-08-12"
    assert r.date_of_expiry == "2012-04-15"


def test_ocr_noise_is_repaired_only_when_check_digit_passes():
    noisy = [
        "REPUBLIC OF UTOPIA  PASSPORT",
        "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
        "L898902C36UTO74O8I22F12O4159ZE184226B<<<<<10",   # O for 0, I for 1 in dates
    ]
    r = read_mrz(noisy)
    assert r is not None and r.valid, r.checks
    assert r.date_of_birth == "1974-08-12" and r.date_of_expiry == "2012-04-15"
    assert r.repairs


def test_wrong_digit_fails_check():
    bad = L2.replace("740812", "740813")
    r = parse_td3(L1, bad)
    assert r.checks["date_of_birth"] is False and not r.valid

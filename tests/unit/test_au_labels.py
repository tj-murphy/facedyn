from facedyn.au_labels import humanise_au_label, humanise_au_labels


def test_humanises_raw_openface_column():
    assert humanise_au_label("AU01_r") == "AU01 - Inner Brow Raiser"


def test_humanises_smoothed_column():
    assert humanise_au_label("smth_AU12_r") == "AU12 - Lip Corner Puller"


def test_normalizes_r_style_human_readable_column():
    assert humanise_au_label("AU01_inner_brow_raiser") == "AU01 - Inner Brow Raiser"


def test_unknown_au_code_returned_unchanged():
    assert humanise_au_label("AU99_r") == "AU99_r"


def test_non_au_column_returned_unchanged():
    assert humanise_au_label("video_filename") == "video_filename"


def test_humanise_au_labels_preserves_order_and_length():
    columns = ["video_filename", "smth_AU01_r", "smth_AU06_r"]
    result = humanise_au_labels(columns)
    assert result == [
        "video_filename",
        "AU01 - Inner Brow Raiser",
        "AU06 - Cheek Raiser",
    ]

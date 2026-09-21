from cell_kinetics.core import dataset_io


def test_exact_match_has_no_note():
    files = [
        "1000nm_40mW_normoxia_v1_GFP.txt",
        "1000nm_40mW_normoxia_v1_mCherry.txt",
        "1000nm_40mW_normoxia_v1_Mask.txt",
        "1000nm_40mW_normoxia_v1_Parameters.txt",
    ]
    cf = dataset_io.find_channel_files("1000nm_40mW_normoxia_v1", files)
    assert dataset_io.is_complete(cf)
    assert dataset_io.dataset_note(cf) is None


def test_annotated_filename_is_found_and_flagged():
    files = [
        "775nm_40mW_normoxia_v1_GFP.txt",
        "775nm_40mW_normoxia_v1_mCherry_treated weak one.txt",
        "775nm_40mW_normoxia_v1_Mask.txt",
        "775nm_40mW_normoxia_v1_Parameters.txt",
    ]
    cf = dataset_io.find_channel_files("775nm_40mW_normoxia_v1", files)
    assert dataset_io.is_complete(cf)
    assert cf["mcherry"] == "775nm_40mW_normoxia_v1_mCherry_treated weak one.txt"
    assert dataset_io.dataset_note(cf) == "mCherry: treated weak one"


def test_missing_channel_stays_missing_even_with_fuzzy_fallback():
    files = [
        "730nm_40mW_normoxia_v1_mCherry_treated weak one.txt",
        "730nm_40mW_normoxia_v1_Mask.txt",
        "730nm_40mW_normoxia_v1_Parameters.txt",
    ]
    cf = dataset_io.find_channel_files("730nm_40mW_normoxia_v1", files)
    assert not dataset_io.is_complete(cf)
    assert dataset_io.missing_channels(cf) == ["GFP/LaminA"]


def test_discover_dataset_prefixes_includes_annotated_datasets():
    files = [
        "775nm_40mW_normoxia_v1_GFP.txt",
        "775nm_40mW_normoxia_v1_mCherry_treated weak one.txt",
        "775nm_40mW_normoxia_v1_Mask.txt",
    ]
    assert dataset_io.discover_dataset_prefixes(files) == ["775nm_40mW_normoxia_v1"]

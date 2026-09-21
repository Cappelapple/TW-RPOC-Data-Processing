import pandas as pd

from cell_kinetics.core import persistence


def test_write_summary_csv_round_trips(tmp_path):
    df = pd.DataFrame([
        {"Dataset": "A", "Decay_Constant_k": 0.1, "Excluded": False},
        {"Dataset": "B", "Decay_Constant_k": 0.2, "Excluded": True},
    ])
    persistence.write_summary_csv(str(tmp_path), df)

    read_back = persistence.read_summary_csv(str(tmp_path))
    assert list(read_back["Dataset"]) == ["A", "B"]
    assert list(read_back["Excluded"]) == [False, True]


def test_write_summary_csv_overwrites_existing_file(tmp_path):
    df1 = pd.DataFrame([{"Dataset": "A", "Decay_Constant_k": 0.1}])
    persistence.write_summary_csv(str(tmp_path), df1)

    df2 = pd.DataFrame([{"Dataset": "B", "Decay_Constant_k": 0.2}])
    persistence.write_summary_csv(str(tmp_path), df2)

    read_back = persistence.read_summary_csv(str(tmp_path))
    assert list(read_back["Dataset"]) == ["B"]

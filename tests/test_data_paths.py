def test_phase1_split_root_supports_min_dataset_short_split_names(
    tiny_short_split_dataset,
):
    from src.data.paths import phase1_split_root

    assert phase1_split_root("train") == tiny_short_split_dataset / "train"
    assert phase1_split_root("val") == tiny_short_split_dataset / "val"
    assert phase1_split_root("test") == tiny_short_split_dataset / "test"

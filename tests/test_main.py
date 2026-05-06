from src.data import ALL_FOURIER_MODES


def test_main_runs_vit_for_all_fourier_modes(monkeypatch):
    import main

    vit_modes = []

    monkeypatch.setattr(main, "RUN_VIT", True)
    monkeypatch.setattr(main, "EPOCHS", 1)
    monkeypatch.setattr(main, "BATCH_SIZE", 2)
    monkeypatch.setattr(main, "NUM_WORKERS", 0)
    monkeypatch.setattr(main, "MULTI_GPU", False)
    monkeypatch.setattr(main, "load_dotenv", lambda: False)
    monkeypatch.setattr(main, "run_xception", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "run_resnet", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "run_mobilenet", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "run_clip", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main,
        "run_vit",
        lambda *args, **kwargs: vit_modes.append(kwargs.get("fourier", "none")),
    )

    main.main()

    assert vit_modes == list(ALL_FOURIER_MODES)

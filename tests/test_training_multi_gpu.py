import inspect

import torch


def test_unwrap_model_returns_original_module_for_plain_model():
    from src.pipelines.training import unwrap_model

    model = torch.nn.Linear(2, 2)

    assert unwrap_model(model) is model


def test_model_state_dict_removes_data_parallel_prefixes():
    from src.pipelines.training import model_state_dict

    model = torch.nn.DataParallel(torch.nn.Linear(2, 2))

    keys = model_state_dict(model).keys()

    assert keys
    assert all(not key.startswith("module.") for key in keys)


def test_maybe_data_parallel_disabled_returns_original_model():
    from src.pipelines.training import maybe_data_parallel

    model = torch.nn.Linear(2, 2)

    wrapped = maybe_data_parallel(model, torch.device("cpu"), enabled=False)

    assert wrapped is model


def test_public_training_functions_accept_multi_gpu_flag():
    from src.pipelines.clip import run_clip
    from src.pipelines.mobilenet import run_mobilenet
    from src.pipelines.resnet import run_resnet
    from src.pipelines.vit import run_vit
    from src.pipelines.xcpetion import run_xception

    for function in (run_clip, run_mobilenet, run_resnet, run_vit, run_xception):
        assert "multi_gpu" in inspect.signature(function).parameters

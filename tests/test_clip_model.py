import inspect


def test_clip_model_module_import_does_not_require_transformers():
    from src.models.clip import CLIPVisionClassifier, train_transform, val_transform

    assert CLIPVisionClassifier.__name__ == "CLIPVisionClassifier"
    assert train_transform is not None
    assert val_transform is not None


def test_clip_classifier_has_no_external_pretrained_loading_api():
    from src.models.clip import CLIPVisionClassifier

    signature = inspect.signature(CLIPVisionClassifier)

    assert "pretrained" not in signature.parameters
    assert "pretrained_model" not in signature.parameters

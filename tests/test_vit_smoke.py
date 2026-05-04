import torch


def test_vit_pipeline_import_does_not_require_transformers():
    from src.pipelines.vit import collate_fn, run_vit

    assert callable(run_vit)
    assert callable(collate_fn)


def test_vit_collate_fn_stacks_tensor_batches():
    from src.pipelines.vit import collate_fn

    batch = [
        (torch.zeros(3, 224, 224), 0, 3),
        (torch.ones(3, 224, 224), 1, 4),
    ]

    imgs, labels, idxs = collate_fn(batch)

    assert imgs.shape == (2, 3, 224, 224)
    assert labels.tolist() == [0, 1]
    assert idxs.tolist() == [3, 4]

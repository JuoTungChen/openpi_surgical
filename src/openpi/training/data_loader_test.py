import dataclasses

import jax

from openpi.models import pi0
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import openpi.transforms as _transforms


def test_torch_data_loader():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)

def create_dataset(config: _config.TrainConfig) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_dataset(data_config, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
        ],
    )
    return data_config, dataset


def test_data_aug_transforms():
    # Load data from dvrk dataset
    config = _config.get_config("dvrk_suturing_test")
    data_config, dataset = create_dataset(config)

    num_frames = len(dataset)
    print(f"Computing stats for {num_frames} frames")
    shuffle = False

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=1,
        num_workers=1,
        shuffle=shuffle,
        num_batches=num_frames,
    )

    # keys = ["state", "actions"]
    # print(data_loader)
    # Get first batch of data
    batch = next(iter(data_loader))
    # print(batch)

    test_img_left = batch["image"]["base_0_rgb"]
    test_img_lw = batch["image"]["left_wrist_0_rgb"]
    test_img_rw = batch["image"]["right_wrist_0_rgb"]
    
    img_hw = (224, 224)
    # Initialize DataAugImages transform
    aug_transform = _transforms.DataAugImages(img_hw=img_hw, ratio=0.95, mask_prob=0.1)
    
    # Create test data dict
    data = {"image": {"left": test_img_left.copy(), "endo_psm2": test_img_lw.copy(), "endo_psm1": test_img_rw.copy()}}
    
    # Apply transforms
    augmented_data = aug_transform(data)
    
    # Verify outputs
    assert "image" in augmented_data
    assert "left" in augmented_data["image"]
    
    print(augmented_data["image"]["endo_psm1"].shape)
    # Check shapes
    # assert augmented_data["image"]["left"].shape == (img_hw[0], img_hw[1], 3)
    
    # Visualize results
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 15))
    axes[0,0].imshow(test_img_left.squeeze(0))
    axes[0,0].set_title("Original Left")
    axes[0,1].imshow(test_img_lw.squeeze(0))
    axes[0,1].set_title("Original left wrist")
    axes[0,2].imshow(test_img_rw.squeeze(0))
    axes[0,2].set_title("Original right wrist")
    axes[1,0].imshow(augmented_data["image"]["left"].squeeze(0))
    axes[1,0].set_title("Augmented Left")
    axes[1,1].imshow(augmented_data["image"]["endo_psm2"].squeeze(0))
    axes[1,1].set_title("Augmented left wrist")
    axes[1,2].imshow(augmented_data["image"]["endo_psm1"].squeeze(0))
    axes[1,2].set_title("Augmented right wrist")
    plt.tight_layout()
    plt.savefig("data_aug_test.png")
    plt.close()
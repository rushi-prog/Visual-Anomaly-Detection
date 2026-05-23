"""Phase 1 tests — Config, data loading, and model factory.

Tests that the foundational components work correctly:
- Config loading from YAML files
- MVTec AD datamodule creation
- Anomalib model factory instantiation
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


class TestConfig:
    """Test configuration loading utilities."""

    def test_load_patchcore_config(self):
        from src.utils.config import load_config
        config = load_config("patchcore")
        assert "model" in config
        assert config["model"]["name"] == "Patchcore"

    def test_load_efficientad_config(self):
        from src.utils.config import load_config
        config = load_config("efficientad")
        assert config["model"]["name"] == "EfficientAd"

    def test_load_stfpm_config(self):
        from src.utils.config import load_config
        config = load_config("stfpm")
        assert config["model"]["name"] == "Stfpm"

    def test_load_autoencoder_config(self):
        from src.utils.config import load_config
        config = load_config("autoencoder")
        assert config["model"]["name"] == "custom_autoencoder"
        assert "latent_dim" in config["model"]

    def test_load_nonexistent_config_raises(self):
        from src.utils.config import load_config
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_model")

    def test_list_configs(self):
        from src.utils.config import list_configs
        configs = list_configs()
        assert len(configs) >= 4
        assert "patchcore.yaml" in configs

    def test_project_paths_exist(self):
        from src.utils.config import PROJECT_ROOT, CONFIGS_DIR
        assert PROJECT_ROOT.exists()
        assert CONFIGS_DIR.exists()

    def test_get_results_dir_creates_directory(self):
        from src.utils.config import get_results_dir
        results = get_results_dir("test_model", "test_category")
        assert results.exists()
        # Cleanup
        results.rmdir()
        results.parent.rmdir()


class TestMVTecExplorer:
    """Test MVTec AD data module creation (no download)."""

    def test_valid_categories_list(self):
        from src.data.mvtec_explorer import MVTEC_CATEGORIES
        assert len(MVTEC_CATEGORIES) == 15
        assert "bottle" in MVTEC_CATEGORIES
        assert "carpet" in MVTEC_CATEGORIES

    def test_object_texture_split(self):
        from src.data.mvtec_explorer import OBJECT_CATEGORIES, TEXTURE_CATEGORIES
        assert len(OBJECT_CATEGORIES) == 10
        assert len(TEXTURE_CATEGORIES) == 5

    def test_create_datamodule(self):
        from src.data.mvtec_explorer import get_mvtec_datamodule
        dm = get_mvtec_datamodule("bottle", image_size=(256, 256))
        assert dm is not None

    def test_invalid_category_raises(self):
        from src.data.mvtec_explorer import get_mvtec_datamodule
        with pytest.raises(ValueError, match="Invalid category"):
            get_mvtec_datamodule("invalid_category")


class TestModelFactory:
    """Test model factory instantiation."""

    def test_create_patchcore(self):
        from src.models.model_factory import create_model
        model = create_model("patchcore")
        assert model is not None

    def test_create_stfpm(self):
        from src.models.model_factory import create_model
        model = create_model("stfpm")
        assert model is not None

    def test_create_padim(self):
        from src.models.model_factory import create_model
        model = create_model("padim")
        assert model is not None

    def test_invalid_model_raises(self):
        from src.models.model_factory import create_model
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent_model")

    def test_list_models(self):
        from src.models.model_factory import list_models
        models = list_models()
        assert "patchcore" in models
        assert "efficientad" in models
        assert "stfpm" in models
        assert "padim" in models

    def test_model_info(self):
        from src.models.model_factory import get_model_info
        info = get_model_info("patchcore")
        assert "name" in info
        assert info["name"] == "PatchCore"
        assert "type" in info
        assert "description" in info


class TestCustomDataset:
    """Test custom dataset utilities."""

    def test_validate_nonexistent_dataset(self):
        from src.data.custom_dataset import validate_dataset_structure
        result = validate_dataset_structure("/nonexistent/path")
        assert result["root_exists"] is False

    def test_nonexistent_root_raises(self):
        from src.data.custom_dataset import get_custom_datamodule
        with pytest.raises(FileNotFoundError):
            get_custom_datamodule("/nonexistent/path")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

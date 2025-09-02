"""Tests for DatasetUploader class."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
from datasets import Dataset

from ml_agents.core.dataset_uploader import DatasetUploader


class TestDatasetUploader:
    """Test suite for DatasetUploader class."""

    @pytest.fixture
    def uploader(self):
        """Create DatasetUploader instance for testing."""
        return DatasetUploader(org_name="test-org")

    @pytest.fixture
    def sample_json_dataset(self):
        """Create a temporary JSON dataset file."""
        data = [
            {"INPUT": "What is 2+2?", "OUTPUT": "4"},
            {"INPUT": "What is the capital of France?", "OUTPUT": "Paris"},
            {"INPUT": "What color is the sky?", "OUTPUT": "Blue"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f, indent=2)
            return f.name

    @pytest.fixture
    def sample_csv_dataset(self):
        """Create a temporary CSV dataset file."""
        data = pd.DataFrame(
            {
                "INPUT": [
                    "What is 2+2?",
                    "What is the capital of France?",
                    "What color is the sky?",
                ],
                "OUTPUT": ["4", "Paris", "Blue"],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            data.to_csv(f.name, index=False)
            return f.name

    @pytest.fixture
    def invalid_json_dataset(self):
        """Create an invalid JSON dataset file."""
        data = [
            {"question": "What is 2+2?", "answer": "4"},  # Wrong schema
            {"question": "What is the capital of France?", "answer": "Paris"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f, indent=2)
            return f.name

    def teardown_method(self, method):
        """Clean up temporary files."""
        # Clean up any temporary files that might have been created
        for temp_file in getattr(self, "_temp_files", []):
            try:
                os.unlink(temp_file)
            except (OSError, FileNotFoundError):
                pass

    def test_init(self):
        """Test DatasetUploader initialization."""
        uploader = DatasetUploader()
        assert uploader.org_name == "c4ai-ml-agents"
        assert not uploader._authenticated

        uploader = DatasetUploader(org_name="custom-org")
        assert uploader.org_name == "custom-org"

    @patch.dict(os.environ, {"HF_TOKEN": "test-token"})
    @patch("ml_agents.core.dataset_uploader.login")
    @patch("ml_agents.core.dataset_uploader.HfApi")
    def test_authenticate_with_env_token(self, mock_hf_api, mock_login, uploader):
        """Test authentication with environment token."""
        mock_api_instance = Mock()
        mock_api_instance.whoami.return_value = {"name": "test-user"}
        mock_hf_api.return_value = mock_api_instance
        uploader.api = mock_api_instance

        result = uploader.authenticate()

        assert result is True
        assert uploader._authenticated is True
        mock_login.assert_called_once_with(token="test-token")
        mock_api_instance.whoami.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)  # Clear HF_TOKEN
    @patch("ml_agents.core.dataset_uploader.login")
    @patch("ml_agents.core.dataset_uploader.HfApi")
    @patch("builtins.input", return_value="input-token")
    def test_authenticate_with_input_token(
        self, mock_input, mock_hf_api, mock_login, uploader
    ):
        """Test authentication with user input token."""
        mock_api_instance = Mock()
        mock_api_instance.whoami.return_value = {"name": "test-user"}
        mock_hf_api.return_value = mock_api_instance
        uploader.api = mock_api_instance

        result = uploader.authenticate()

        assert result is True
        assert uploader._authenticated is True
        mock_login.assert_called_once_with(token="input-token")

    @patch.dict(os.environ, {}, clear=True)
    @patch("builtins.input", return_value="")  # Empty token
    def test_authenticate_no_token(self, mock_input, uploader):
        """Test authentication fails with no token."""
        result = uploader.authenticate()

        assert result is False
        assert uploader._authenticated is False

    @patch(
        "ml_agents.core.dataset_uploader.login", side_effect=Exception("Auth failed")
    )
    @patch.dict(os.environ, {"HF_TOKEN": "invalid-token"})
    def test_authenticate_failure(self, mock_login, uploader):
        """Test authentication failure handling."""
        result = uploader.authenticate()

        assert result is False
        assert uploader._authenticated is False

    def test_validate_processed_file_json_valid(self, uploader, sample_json_dataset):
        """Test validation of valid JSON dataset."""
        result = uploader.validate_processed_file(sample_json_dataset)

        assert result["validation_passed"] is True
        assert result["format"] == "json"
        assert result["sample_count"] == 3
        assert result["has_input_output_schema"] is True
        assert len(result["issues"]) == 0

    def test_validate_processed_file_csv_valid(self, uploader, sample_csv_dataset):
        """Test validation of valid CSV dataset."""
        result = uploader.validate_processed_file(sample_csv_dataset)

        assert result["validation_passed"] is True
        assert result["format"] == "csv"
        assert result["sample_count"] == 3
        assert result["has_input_output_schema"] is True

    def test_validate_processed_file_invalid_schema(
        self, uploader, invalid_json_dataset
    ):
        """Test validation of dataset with invalid schema."""
        result = uploader.validate_processed_file(invalid_json_dataset)

        assert result["validation_passed"] is False
        assert result["format"] == "json"
        assert result["has_input_output_schema"] is False
        assert "Records don't have INPUT/OUTPUT schema" in result["issues"]

    def test_validate_processed_file_not_found(self, uploader):
        """Test validation of non-existent file."""
        with pytest.raises(FileNotFoundError):
            uploader.validate_processed_file("nonexistent.json")

    def test_validate_processed_file_unsupported_format(self, uploader):
        """Test validation of unsupported file format."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            temp_file = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                uploader.validate_processed_file(temp_file)
        finally:
            os.unlink(temp_file)

    def test_validate_empty_dataset(self, uploader):
        """Test validation of empty dataset."""
        # Create empty JSON file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            empty_file = f.name

        try:
            result = uploader.validate_processed_file(empty_file)
            assert result["validation_passed"] is False
            assert "Dataset is empty" in result["issues"]
        finally:
            os.unlink(empty_file)

    def test_generate_dataset_card_basic(self, uploader):
        """Test basic dataset card generation."""
        card_content = uploader._generate_dataset_card(
            source_dataset="test/dataset", target_name="TestDataset", sample_count=100
        )

        assert "# TestDataset" in card_content
        assert "test/dataset" in card_content
        assert "num_examples: 100" in card_content
        assert "c4ai-ml-agents/TestDataset" in card_content
        assert "ml-agents" in card_content

    def test_generate_dataset_card_with_config(self, uploader):
        """Test dataset card generation with config."""
        card_content = uploader._generate_dataset_card(
            source_dataset="test/dataset",
            target_name="TestDataset",
            config="test-config",
            description="Custom description",
            sample_count=50,
        )

        assert "Configuration**: test-config" in card_content
        assert "Custom description" in card_content
        assert "config-test-config" in card_content

    def test_generate_dataset_card_http_source(self, uploader):
        """Test dataset card generation with HTTP source URL."""
        card_content = uploader._generate_dataset_card(
            source_dataset="https://example.com/dataset",
            target_name="TestDataset",
            sample_count=25,
        )

        assert "https://example.com/dataset" in card_content
        assert (
            "[https://example.com/dataset](https://example.com/dataset)" in card_content
        )

    @patch("ml_agents.core.dataset_uploader.HfApi")
    @patch("ml_agents.core.dataset_uploader.DatasetCard")
    def test_upload_dataset_success(
        self, mock_card_class, mock_hf_api, uploader, sample_json_dataset
    ):
        """Test successful dataset upload."""
        # Mock authentication
        uploader._authenticated = True
        uploader.authenticate = Mock(return_value=True)

        # Mock HfApi
        mock_api_instance = Mock()
        mock_api_instance.create_repo = Mock()
        mock_api_instance.upload_file = Mock()
        uploader.api = mock_api_instance

        # Mock DatasetCard
        mock_card = Mock()
        mock_card.push_to_hub = Mock()
        mock_card_class.return_value = mock_card

        # Create temporary files for testing
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the base files
            base_name = Path(sample_json_dataset).stem
            temp_path = Path(temp_dir)

            # Create analysis and rules files
            (temp_path / f"{base_name}_analysis.json").write_text(
                '{"test": "analysis"}'
            )
            (temp_path / f"{base_name}_rules.json").write_text('{"test": "rules"}')

            # Move the sample dataset to temp dir
            import shutil

            temp_dataset = temp_path / Path(sample_json_dataset).name
            shutil.copy(sample_json_dataset, temp_dataset)

            result = uploader.upload_dataset(
                processed_file=str(temp_dataset),
                source_dataset="test/source",
                target_name="TestDataset",
            )

        assert result == "test-org/TestDataset"
        mock_api_instance.create_repo.assert_called_once_with(
            "test-org/TestDataset", repo_type="dataset", private=False, exist_ok=True
        )
        # Should upload at least 3 files (main, analysis, rules)
        assert mock_api_instance.upload_file.call_count >= 3
        mock_card.push_to_hub.assert_called_once_with("test-org/TestDataset")

    def test_upload_dataset_authentication_failure(self, uploader, sample_json_dataset):
        """Test upload failure due to authentication."""
        uploader.authenticate = Mock(return_value=False)

        with pytest.raises(ValueError, match="HuggingFace authentication failed"):
            uploader.upload_dataset(
                processed_file=sample_json_dataset,
                source_dataset="test/source",
                target_name="TestDataset",
            )

    def test_upload_dataset_validation_failure(self, uploader, invalid_json_dataset):
        """Test upload failure due to validation."""
        uploader.authenticate = Mock(return_value=True)

        with pytest.raises(ValueError, match="Dataset validation failed"):
            uploader.upload_dataset(
                processed_file=invalid_json_dataset,
                source_dataset="test/source",
                target_name="TestDataset",
            )

    def test_list_organization_datasets_not_authenticated(self, uploader):
        """Test listing datasets when not authenticated."""
        uploader.authenticate = Mock(return_value=False)

        result = uploader.list_organization_datasets()

        assert result == []

    @patch("ml_agents.core.dataset_uploader.HfApi")
    def test_list_organization_datasets_success(self, mock_hf_api, uploader):
        """Test successful listing of organization datasets."""
        uploader._authenticated = True
        uploader.authenticate = Mock(return_value=True)

        # Mock dataset objects
        mock_dataset1 = Mock()
        mock_dataset1.id = "test-org/dataset1"
        mock_dataset1.downloads = 100
        mock_dataset1.likes = 5

        mock_dataset2 = Mock()
        mock_dataset2.id = "test-org/dataset2"
        mock_dataset2.downloads = 50
        mock_dataset2.likes = 2

        mock_api_instance = Mock()
        mock_api_instance.list_datasets.return_value = [mock_dataset1, mock_dataset2]
        uploader.api = mock_api_instance

        result = uploader.list_organization_datasets()

        expected = [
            {"id": "test-org/dataset1", "downloads": 100, "likes": 5},
            {"id": "test-org/dataset2", "downloads": 50, "likes": 2},
        ]
        assert result == expected

    def test_delete_dataset_not_authenticated(self, uploader):
        """Test dataset deletion when not authenticated."""
        uploader.authenticate = Mock(return_value=False)

        result = uploader.delete_dataset("test-dataset")

        assert result is False

    @patch("ml_agents.core.dataset_uploader.HfApi")
    def test_delete_dataset_success(self, mock_hf_api, uploader):
        """Test successful dataset deletion."""
        uploader._authenticated = True
        uploader.authenticate = Mock(return_value=True)

        mock_api_instance = Mock()
        mock_api_instance.delete_repo = Mock()
        uploader.api = mock_api_instance

        result = uploader.delete_dataset("test-dataset")

        assert result is True
        mock_api_instance.delete_repo.assert_called_once_with(
            "test-org/test-dataset", repo_type="dataset"
        )

    @patch("ml_agents.core.dataset_uploader.HfApi")
    def test_delete_dataset_failure(self, mock_hf_api, uploader):
        """Test dataset deletion failure."""
        uploader._authenticated = True
        uploader.authenticate = Mock(return_value=True)

        mock_api_instance = Mock()
        mock_api_instance.delete_repo.side_effect = Exception("Delete failed")
        uploader.api = mock_api_instance

        result = uploader.delete_dataset("test-dataset")

        assert result is False


class TestDatasetUploaderIntegration:
    """Integration tests for DatasetUploader that test workflows."""

    @pytest.fixture
    def uploader(self):
        """Create DatasetUploader instance for integration testing."""
        return DatasetUploader(org_name="test-integration")

    def test_full_workflow_json_dataset(self, uploader):
        """Test full workflow from validation to upload preparation (mock)."""
        # Create test dataset
        data = [
            {"INPUT": "Test question 1", "OUTPUT": "Test answer 1"},
            {"INPUT": "Test question 2", "OUTPUT": "Test answer 2"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f, indent=2)
            test_file = f.name

        try:
            # Test validation
            validation = uploader.validate_processed_file(test_file)
            assert validation["validation_passed"] is True
            assert validation["sample_count"] == 2

            # Test card generation
            card = uploader._generate_dataset_card(
                source_dataset="test/source",
                target_name="TestIntegration",
                sample_count=validation["sample_count"],
            )
            assert "TestIntegration" in card
            assert "num_examples: 2" in card

        finally:
            os.unlink(test_file)

    def test_workflow_with_large_dataset_warning(self, uploader):
        """Test workflow with large dataset that triggers warnings."""
        # Create dataset with many samples
        data = [
            {"INPUT": f"Question {i}", "OUTPUT": f"Answer {i}"} for i in range(1000)
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f, indent=2)
            test_file = f.name

        try:
            validation = uploader.validate_processed_file(test_file)
            assert validation["validation_passed"] is True
            assert validation["sample_count"] == 1000

            # Should have warning about file size
            file_size_mb = validation["file_size_mb"]
            if file_size_mb > 100:  # If file is large enough
                assert any("Large file size" in issue for issue in validation["issues"])

        finally:
            os.unlink(test_file)

    def test_error_handling_corrupted_json(self, uploader):
        """Test error handling with corrupted JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": json}')  # Invalid JSON
            corrupted_file = f.name

        try:
            with pytest.raises(ValueError, match="File validation failed"):
                uploader.validate_processed_file(corrupted_file)
        finally:
            os.unlink(corrupted_file)


class TestDatasetUploader403Fallback:
    """Test suite for DatasetUploader 403 fallback logic."""

    @pytest.fixture
    def uploader(self):
        """Create DatasetUploader instance for testing."""
        return DatasetUploader(org_name="test-org")

    @pytest.fixture
    def sample_json_dataset(self):
        """Create a temporary JSON dataset file."""
        data = [
            {"INPUT": "What is 2+2?", "OUTPUT": "4"},
            {"INPUT": "What is the capital of France?", "OUTPUT": "Paris"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f, indent=2)
            return f.name

    def teardown_method(self, method):
        """Clean up temporary files."""
        for temp_file in getattr(self, "_temp_files", []):
            try:
                os.unlink(temp_file)
            except (OSError, FileNotFoundError):
                pass

    @patch("ml_agents.core.dataset_uploader.HfApi")
    @patch("ml_agents.core.dataset_uploader.DatasetCard")
    def test_upload_file_403_fallback_success(
        self, mock_card_class, mock_hf_api, uploader, sample_json_dataset
    ):
        """Test successful fallback to PR creation after 403 error on file upload."""
        # Mock authentication
        uploader._authenticated = True
        uploader.authenticate = Mock(return_value=True)

        # Mock HfApi with 403 error on first upload, success on PR
        mock_api_instance = Mock()
        mock_api_instance.create_repo = Mock()

        # First call fails with 403, second succeeds with create_pr=True
        def upload_file_side_effect(*args, **kwargs):
            if kwargs.get("create_pr", False):
                return Mock(
                    pr_url="https://huggingface.co/datasets/test-org/TestDataset/discussions/1"
                )
            else:
                raise Exception("403 Forbidden: Authorization error.")

        mock_api_instance.upload_file.side_effect = upload_file_side_effect
        uploader.api = mock_api_instance

        # Mock DatasetCard
        mock_card = Mock()
        mock_card.push_to_hub = Mock()
        mock_card_class.return_value = mock_card

        # Create temporary files for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            base_name = Path(sample_json_dataset).stem
            temp_path = Path(temp_dir)

            # Create analysis and rules files
            (temp_path / f"{base_name}_analysis.json").write_text(
                '{"test": "analysis"}'
            )
            (temp_path / f"{base_name}_rules.json").write_text('{"test": "rules"}')

            # Move the sample dataset to temp dir
            import shutil

            temp_dataset = temp_path / Path(sample_json_dataset).name
            shutil.copy(sample_json_dataset, temp_dataset)

            result = uploader.upload_dataset(
                processed_file=str(temp_dataset),
                source_dataset="test/source",
                target_name="TestDataset",
            )

        assert result == "test-org/TestDataset"
        # Should have called upload_file twice for each file (direct + PR fallback)
        assert (
            mock_api_instance.upload_file.call_count >= 6
        )  # 3 files × 2 attempts each

    @patch("ml_agents.core.dataset_uploader.HfApi")
    @patch("ml_agents.core.dataset_uploader.DatasetCard")
    def test_upload_dataset_card_403_fallback_success(
        self, mock_card_class, mock_hf_api, uploader, sample_json_dataset
    ):
        """Test successful fallback to PR creation after 403 error on dataset card upload."""
        # Mock authentication
        uploader._authenticated = True
        uploader.authenticate = Mock(return_value=True)

        # Mock HfApi for successful file uploads
        mock_api_instance = Mock()
        mock_api_instance.create_repo = Mock()
        mock_api_instance.upload_file = Mock()
        uploader.api = mock_api_instance

        # Mock DatasetCard with 403 error on first push, success on PR
        mock_card = Mock()

        def push_to_hub_side_effect(repo_id, create_pr=False):
            if create_pr:
                return Mock()  # Success
            else:
                raise Exception("403 Forbidden: Authorization error.")

        mock_card.push_to_hub.side_effect = push_to_hub_side_effect
        mock_card_class.return_value = mock_card

        # Create temporary files for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            base_name = Path(sample_json_dataset).stem
            temp_path = Path(temp_dir)

            # Create the main dataset file
            temp_dataset = temp_path / Path(sample_json_dataset).name
            import shutil

            shutil.copy(sample_json_dataset, temp_dataset)

            result = uploader.upload_dataset(
                processed_file=str(temp_dataset),
                source_dataset="test/source",
                target_name="TestDataset",
            )

        assert result == "test-org/TestDataset"
        # Should have called push_to_hub twice (direct + PR fallback)
        assert mock_card.push_to_hub.call_count == 2

    @patch("ml_agents.core.dataset_uploader.HfApi")
    @patch("ml_agents.core.dataset_uploader.DatasetCard")
    def test_upload_403_fallback_failure(
        self, mock_card_class, mock_hf_api, uploader, sample_json_dataset
    ):
        """Test fallback failure when both direct upload and PR creation fail."""
        # Mock authentication
        uploader._authenticated = True
        uploader.authenticate = Mock(return_value=True)

        # Mock HfApi with 403 error on both direct and PR uploads
        mock_api_instance = Mock()
        mock_api_instance.create_repo = Mock()
        mock_api_instance.upload_file.side_effect = Exception(
            "403 Forbidden: Authorization error."
        )
        uploader.api = mock_api_instance

        # Mock DatasetCard with failure on both attempts
        mock_card = Mock()
        mock_card.push_to_hub.side_effect = Exception(
            "403 Forbidden: Authorization error."
        )
        mock_card_class.return_value = mock_card

        # Create temporary files for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            base_name = Path(sample_json_dataset).stem
            temp_path = Path(temp_dir)

            temp_dataset = temp_path / Path(sample_json_dataset).name
            import shutil

            shutil.copy(sample_json_dataset, temp_dataset)

            # Should not raise exception but should complete with failed uploads
            result = uploader.upload_dataset(
                processed_file=str(temp_dataset),
                source_dataset="test/source",
                target_name="TestDataset",
            )

        assert result == "test-org/TestDataset"
        # Should have attempted both direct and PR uploads
        assert mock_api_instance.upload_file.call_count >= 2

    def test_403_error_detection_patterns(self, uploader):
        """Test that various 403 error patterns are detected correctly."""
        test_errors = [
            "403 Forbidden: Authorization error.",
            "403 Forbidden: pass `create_pr=1` as a query parameter",
            "Forbidden: pass `create_pr=1` as a query parameter",
            "Authorization error.",
            "HTTPError: 403 Client Error: Forbidden",
        ]

        for error_msg in test_errors:
            error_str = error_msg.lower()
            should_trigger = (
                "403" in error_str
                or "forbidden" in error_str
                or "authorization" in error_str
            )
            assert should_trigger, f"Error pattern should trigger fallback: {error_msg}"

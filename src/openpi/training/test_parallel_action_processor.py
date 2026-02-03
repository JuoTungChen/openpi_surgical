"""
Tests for parallel action processor.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from openpi.training.parallel_action_processor import (
    ParallelActionProcessor,
    ActionTransformBatch,
    ActionTransformResult,
    get_global_parallel_processor,
    shutdown_global_processor,
)


@pytest.fixture
def mock_modality_configs():
    """Mock modality configurations."""
    return {
        "test_embodiment": {
            "action": Mock(),
            "state": Mock(),
        }
    }


@pytest.fixture
def mock_statistics():
    """Mock statistics."""
    return {
        "test_embodiment": {
            "action": {"mean": [0.0], "std": [1.0]},
            "state": {"mean": [0.0], "std": [1.0]},
        }
    }


@pytest.fixture
def sample_actions():
    """Sample action data."""
    return {
        "joint_positions": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "gripper": np.array([[0.5], [0.8]], dtype=np.float32),
    }


@pytest.fixture
def sample_states():
    """Sample state data."""
    return {
        "joint_positions": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "end_effector_pose": np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }


class TestParallelActionProcessor:
    """Test cases for ParallelActionProcessor."""
    
    def test_init(self, mock_modality_configs, mock_statistics):
        """Test processor initialization."""
        processor = ParallelActionProcessor(
            modality_configs=mock_modality_configs,
            statistics=mock_statistics,
            max_workers=2,
            batch_size=4,
        )
        
        assert processor.modality_configs == mock_modality_configs
        assert processor.statistics == mock_statistics
        assert processor.max_workers == 2
        assert processor.batch_size == 4
        assert processor.use_threading is False
    
    def test_init_auto_workers(self, mock_modality_configs, mock_statistics):
        """Test automatic worker count detection."""
        processor = ParallelActionProcessor(
            modality_configs=mock_modality_configs,
            statistics=mock_statistics,
        )
        
        # Should auto-detect based on CPU count
        assert processor.max_workers >= 1
    
    def test_init_threading(self, mock_modality_configs, mock_statistics):
        """Test threading mode initialization."""
        processor = ParallelActionProcessor(
            modality_configs=mock_modality_configs,
            statistics=mock_statistics,
            use_threading=True,
        )
        
        assert processor.use_threading is True
        # Threading should use more workers
        assert processor.max_workers >= 2
    
    @patch('openpi.training.parallel_action_processor.StateActionProcessor')
    def test_get_processor(self, mock_processor_class, mock_modality_configs, mock_statistics):
        """Test processor creation and caching."""
        mock_processor_instance = Mock()
        mock_processor_class.return_value = mock_processor_instance
        
        processor = ParallelActionProcessor(
            modality_configs=mock_modality_configs,
            statistics=mock_statistics,
        )
        
        # First call should create processor
        result1 = processor._get_processor()
        assert result1 == mock_processor_instance
        mock_processor_class.assert_called_once()
        mock_processor_instance.eval.assert_called_once()
        
        # Second call should use cached processor
        result2 = processor._get_processor()
        assert result2 == mock_processor_instance
        # Should not create another processor
        assert mock_processor_class.call_count == 1
    
    @patch('openpi.training.parallel_action_processor.StateActionProcessor')
    def test_process_single_item(self, mock_processor_class, mock_modality_configs, 
                                mock_statistics, sample_actions, sample_states):
        """Test single item processing."""
        mock_processor_instance = Mock()
        mock_processor_class.return_value = mock_processor_instance
        
        # Mock processor methods
        processed_states = {"processed_state": np.array([1.0, 2.0])}
        processed_actions = {"processed_action": np.array([[3.0, 4.0]])}
        mock_processor_instance.apply_state.return_value = processed_states
        mock_processor_instance.apply_action.return_value = processed_actions
        
        processor = ParallelActionProcessor(
            modality_configs=mock_modality_configs,
            statistics=mock_statistics,
        )
        
        result_actions, result_states = processor._process_single_item(
            sample_actions, sample_states, "test_embodiment", "test_stats"
        )
        
        assert result_actions == processed_actions
        assert result_states == processed_states
        
        # Verify processor calls
        mock_processor_instance.apply_state.assert_called_once_with(
            sample_states,
            embodiment_tag="test_embodiment",
            stats_key="test_stats",
        )
        mock_processor_instance.apply_action.assert_called_once_with(
            sample_actions,
            embodiment_tag="test_embodiment",
            state=sample_states,
            stats_key="test_stats",
        )
    
    @patch('openpi.training.parallel_action_processor.StateActionProcessor')
    def test_process_single_item_fallback(self, mock_processor_class, mock_modality_configs, 
                                         mock_statistics, sample_actions, sample_states):
        """Test fallback when processor creation fails."""
        # Mock import error
        mock_processor_class.side_effect = ImportError("GR00T not available")
        
        processor = ParallelActionProcessor(
            modality_configs=mock_modality_configs,
            statistics=mock_statistics,
        )
        
        result_actions, result_states = processor._process_single_item(
            sample_actions, sample_states, "test_embodiment", "test_stats"
        )
        
        # Should return unprocessed data
        assert result_actions == sample_actions
        assert result_states == sample_states
    
    @patch('openpi.training.parallel_action_processor.StateActionProcessor')
    def test_process_batch_small(self, mock_processor_class, mock_modality_configs, 
                                mock_statistics, sample_actions, sample_states):
        """Test batch processing with small batch (synchronous)."""
        mock_processor_instance = Mock()
        mock_processor_class.return_value = mock_processor_instance
        
        # Mock processor methods
        processed_states = {"processed_state": np.array([1.0, 2.0])}
        processed_actions = {"processed_action": np.array([[3.0, 4.0]])}
        mock_processor_instance.apply_state.return_value = processed_states
        mock_processor_instance.apply_action.return_value = processed_actions
        
        processor = ParallelActionProcessor(
            modality_configs=mock_modality_configs,
            statistics=mock_statistics,
        )
        
        # Small batch should be processed synchronously
        actions_list = [sample_actions, sample_actions]
        states_list = [sample_states, sample_states]
        embodiment_tags = ["test_embodiment", "test_embodiment"]
        stats_keys = ["test_stats", "test_stats"]
        
        result_actions, result_states = processor.process_batch(
            actions_list, states_list, embodiment_tags, stats_keys
        )
        
        assert len(result_actions) == 2
        assert len(result_states) == 2
        assert result_actions[0] == processed_actions
        assert result_states[0] == processed_states
    
    def test_process_batch_empty(self, mock_modality_configs, mock_statistics):
        """Test batch processing with empty input."""
        processor = ParallelActionProcessor(
            modality_configs=mock_modality_configs,
            statistics=mock_statistics,
        )
        
        result_actions, result_states = processor.process_batch([], [], [], [])
        
        assert result_actions == []
        assert result_states == []
    
    def test_process_batch_mismatched_lengths(self, mock_modality_configs, mock_statistics):
        """Test batch processing with mismatched input lengths."""
        processor = ParallelActionProcessor(
            modality_configs=mock_modality_configs,
            statistics=mock_statistics,
        )
        
        with pytest.raises(ValueError, match="All input lists must have the same length"):
            processor.process_batch([{}], [], ["tag"], ["stats"])
    
    @patch('openpi.training.parallel_action_processor.StateActionProcessor')
    def test_process_single(self, mock_processor_class, mock_modality_configs, 
                           mock_statistics, sample_actions, sample_states):
        """Test single processing method."""
        mock_processor_instance = Mock()
        mock_processor_class.return_value = mock_processor_instance
        
        # Mock processor methods
        processed_states = {"processed_state": np.array([1.0, 2.0])}
        processed_actions = {"processed_action": np.array([[3.0, 4.0]])}
        mock_processor_instance.apply_state.return_value = processed_states
        mock_processor_instance.apply_action.return_value = processed_actions
        
        processor = ParallelActionProcessor(
            modality_configs=mock_modality_configs,
            statistics=mock_statistics,
        )
        
        result_actions, result_states = processor.process_single(
            sample_actions, sample_states, "test_embodiment", "test_stats"
        )
        
        assert result_actions == processed_actions
        assert result_states == processed_states
    
    def test_performance_stats_empty(self, mock_modality_configs, mock_statistics):
        """Test performance stats with no processing done."""
        processor = ParallelActionProcessor(
            modality_configs=mock_modality_configs,
            statistics=mock_statistics,
        )
        
        stats = processor.get_performance_stats()
        
        assert stats["total_processed"] == 0
        assert stats["average_time_per_item"] == 0.0
        assert stats["total_time"] == 0.0
        assert stats["throughput_items_per_sec"] == 0.0
        assert stats["batch_count"] == 0
        assert stats["average_batch_time"] == 0.0
    
    def test_reset_stats(self, mock_modality_configs, mock_statistics):
        """Test stats reset functionality."""
        processor = ParallelActionProcessor(
            modality_configs=mock_modality_configs,
            statistics=mock_statistics,
        )
        
        # Simulate some processing
        processor._total_processed = 10
        processor._total_time = 5.0
        processor._batch_times = [1.0, 2.0, 2.0]
        
        processor.reset_stats()
        
        assert processor._total_processed == 0
        assert processor._total_time == 0.0
        assert processor._batch_times == []
    
    def test_shutdown(self, mock_modality_configs, mock_statistics):
        """Test shutdown functionality."""
        processor = ParallelActionProcessor(
            modality_configs=mock_modality_configs,
            statistics=mock_statistics,
        )
        
        # Mock executor
        mock_executor = Mock()
        processor._executor = mock_executor
        
        processor.shutdown()
        
        mock_executor.shutdown.assert_called_once_with(wait=True)
        assert processor._executor is None
        assert processor._processor_cache == {}


class TestGlobalProcessor:
    """Test cases for global processor functions."""
    
    def test_get_global_processor(self, mock_modality_configs, mock_statistics):
        """Test global processor creation."""
        # Clean up any existing global processor
        shutdown_global_processor()
        
        processor = get_global_parallel_processor(
            mock_modality_configs, mock_statistics, max_workers=2
        )
        
        assert isinstance(processor, ParallelActionProcessor)
        assert processor.max_workers == 2
        
        # Second call should return same instance
        processor2 = get_global_parallel_processor(
            mock_modality_configs, mock_statistics, max_workers=4
        )
        
        assert processor2 is processor
        # Should still have original settings
        assert processor2.max_workers == 2
    
    def test_shutdown_global_processor(self, mock_modality_configs, mock_statistics):
        """Test global processor shutdown."""
        # Create global processor
        processor = get_global_parallel_processor(
            mock_modality_configs, mock_statistics
        )
        
        # Mock shutdown method
        processor.shutdown = Mock()
        
        shutdown_global_processor()
        
        processor.shutdown.assert_called_once()
        
        # Next call should create new processor
        new_processor = get_global_parallel_processor(
            mock_modality_configs, mock_statistics
        )
        
        assert new_processor is not processor


class TestActionTransformBatch:
    """Test cases for ActionTransformBatch dataclass."""
    
    def test_creation(self):
        """Test batch creation."""
        batch = ActionTransformBatch(
            actions=[{"action1": np.array([1.0])}],
            states=[{"state1": np.array([2.0])}],
            embodiment_tags=["test"],
            stats_keys=["stats"],
            batch_id=0,
        )
        
        assert len(batch.actions) == 1
        assert len(batch.states) == 1
        assert batch.embodiment_tags == ["test"]
        assert batch.stats_keys == ["stats"]
        assert batch.batch_id == 0


class TestActionTransformResult:
    """Test cases for ActionTransformResult dataclass."""
    
    def test_creation(self):
        """Test result creation."""
        result = ActionTransformResult(
            processed_actions=[{"action1": np.array([1.0])}],
            processed_states=[{"state1": np.array([2.0])}],
            batch_id=0,
            processing_time=1.5,
        )
        
        assert len(result.processed_actions) == 1
        assert len(result.processed_states) == 1
        assert result.batch_id == 0
        assert result.processing_time == 1.5
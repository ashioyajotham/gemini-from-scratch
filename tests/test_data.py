"""Tests for data processing modules."""

import tempfile
from pathlib import Path

import pytest
import torch

from src.data.tokenizer import BPETokenizer, load_tokenizer, save_tokenizer
from src.data.dataset import TextDataset, create_dataset, train_val_split
from src.data.dataloader import create_dataloader, collate_fn
from src.data.preprocessing import clean_text, chunk_text, normalize_text


class TestTokenizer:
    """Tests for BPE tokenizer."""

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for tokenizer training."""
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Hello world! This is a test.",
            "Machine learning is fascinating.",
            "Neural networks can learn patterns.",
            "Transformers have revolutionized NLP.",
        ] * 20  # Repeat for enough training data

    @pytest.fixture
    def trained_tokenizer(self, sample_texts, tmp_path):
        """Create a trained tokenizer for testing."""
        tokenizer = BPETokenizer()
        tokenizer.train(sample_texts, vocab_size=100, output_dir=tmp_path)
        return tokenizer

    def test_tokenizer_train(self, sample_texts, tmp_path):
        """Tokenizer should train without errors."""
        tokenizer = BPETokenizer()
        tokenizer.train(sample_texts, vocab_size=100, output_dir=tmp_path)
        assert tokenizer.vocab_size > 0

    def test_tokenizer_encode_decode(self, trained_tokenizer):
        """Tokenizer should encode and decode text."""
        text = "Hello world"
        ids = trained_tokenizer.encode(text)

        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

        decoded = trained_tokenizer.decode(ids)
        assert isinstance(decoded, str)
        # Decoded text should be similar (may have slight differences)
        assert "hello" in decoded.lower() or "world" in decoded.lower()

    def test_tokenizer_special_tokens(self, trained_tokenizer):
        """Tokenizer should handle special tokens."""
        ids = trained_tokenizer.encode("Test", add_bos=True, add_eos=True)

        assert ids[0] == trained_tokenizer.bos_id
        assert ids[-1] == trained_tokenizer.eos_id

    def test_tokenizer_save_load(self, trained_tokenizer, tmp_path):
        """Tokenizer should save and load correctly."""
        save_path = tmp_path / "saved_tokenizer.model"
        trained_tokenizer.save(save_path)

        loaded = load_tokenizer(save_path)
        assert loaded.vocab_size == trained_tokenizer.vocab_size

        # Should encode the same way
        text = "Test encoding"
        assert trained_tokenizer.encode(text) == loaded.encode(text)

    def test_tokenizer_batch_encode(self, trained_tokenizer):
        """Tokenizer should batch encode texts."""
        texts = ["Hello", "World", "Test"]
        batch_ids = trained_tokenizer.encode_batch(texts)

        assert len(batch_ids) == 3
        assert all(isinstance(ids, list) for ids in batch_ids)

    def test_tokenizer_vocab(self, trained_tokenizer):
        """Tokenizer should provide vocabulary access."""
        vocab = trained_tokenizer.get_vocab()
        assert isinstance(vocab, dict)
        assert len(vocab) == trained_tokenizer.vocab_size


class TestDataset:
    """Tests for dataset classes."""

    @pytest.fixture
    def sample_token_ids(self):
        """Sample token IDs for testing."""
        return list(range(100))  # Simple sequential IDs

    def test_text_dataset_length(self, sample_token_ids):
        """TextDataset should have correct length."""
        dataset = TextDataset(sample_token_ids, seq_length=10)
        # With 100 tokens and seq_length=10, stride=10: 9 sequences
        # (need seq_length+1 tokens per sequence)
        expected = (len(sample_token_ids) - 10) // 10 + 1
        assert len(dataset) == expected

    def test_text_dataset_getitem(self, sample_token_ids):
        """TextDataset should return input/target pairs."""
        dataset = TextDataset(sample_token_ids, seq_length=10)
        input_ids, target_ids = dataset[0]

        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(target_ids, torch.Tensor)
        assert input_ids.shape == (10,)
        assert target_ids.shape == (10,)

        # Target should be input shifted by 1
        assert torch.equal(input_ids[1:], target_ids[:-1])

    def test_text_dataset_with_stride(self, sample_token_ids):
        """TextDataset should handle custom stride."""
        dataset = TextDataset(sample_token_ids, seq_length=10, stride=5)
        # With stride=5, should have more sequences
        assert len(dataset) > (len(sample_token_ids) - 10) // 10 + 1

    def test_text_dataset_padding(self):
        """TextDataset should pad short sequences."""
        short_ids = list(range(5))  # Only 5 tokens
        dataset = TextDataset(short_ids, seq_length=10, pad_id=0)
        input_ids, target_ids = dataset[0]

        assert input_ids.shape == (10,)
        assert target_ids.shape == (10,)

    def test_create_dataset(self, tmp_path):
        """create_dataset should create a dataset from texts."""
        # Create a simple tokenizer with diverse text
        tokenizer = BPETokenizer()
        texts = [
            "The quick brown fox jumps over the lazy dog. " * 20,
            "Machine learning is fascinating and powerful. " * 20,
            "Neural networks can learn complex patterns. " * 20,
        ]
        tokenizer.train(texts, vocab_size=100, output_dir=tmp_path)

        dataset = create_dataset("Hello world", tokenizer, seq_length=5)
        assert len(dataset) > 0

    def test_train_val_split(self, sample_token_ids):
        """train_val_split should split dataset correctly."""
        dataset = TextDataset(sample_token_ids, seq_length=10)
        train_ds, val_ds = train_val_split(dataset, val_ratio=0.2)

        total = len(train_ds) + len(val_ds)
        assert total == len(dataset)
        assert len(val_ds) / total == pytest.approx(0.2, abs=0.1)


class TestDataLoader:
    """Tests for dataloader utilities."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset for testing."""
        token_ids = list(range(200))
        return TextDataset(token_ids, seq_length=10)

    def test_create_dataloader(self, sample_dataset):
        """create_dataloader should create a working dataloader."""
        loader = create_dataloader(sample_dataset, batch_size=4, shuffle=False)

        batch = next(iter(loader))
        input_ids, target_ids = batch

        assert input_ids.shape == (4, 10)
        assert target_ids.shape == (4, 10)

    def test_collate_fn(self):
        """collate_fn should batch and pad sequences."""
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])),
            (torch.tensor([5, 6]), torch.tensor([6, 7])),
        ]

        input_ids, target_ids = collate_fn(batch, pad_id=0)

        assert input_ids.shape == (2, 3)  # Padded to longest
        assert target_ids.shape == (2, 3)
        assert input_ids[1, 2] == 0  # Padding

    def test_dataloader_iteration(self, sample_dataset):
        """DataLoader should iterate over all batches."""
        loader = create_dataloader(
            sample_dataset,
            batch_size=4,
            shuffle=False,
            drop_last=True
        )

        batch_count = 0
        for input_ids, target_ids in loader:
            batch_count += 1
            assert input_ids.shape[0] == 4

        assert batch_count == len(sample_dataset) // 4


class TestPreprocessing:
    """Tests for text preprocessing."""

    def test_clean_text_urls(self):
        """clean_text should remove URLs."""
        text = "Check out https://example.com for more"
        cleaned = clean_text(text, remove_urls=True)
        assert "https://" not in cleaned
        assert "example.com" not in cleaned

    def test_clean_text_emails(self):
        """clean_text should remove emails."""
        text = "Contact me at test@example.com"
        cleaned = clean_text(text, remove_emails=True)
        assert "@" not in cleaned

    def test_clean_text_whitespace(self):
        """clean_text should collapse whitespace."""
        text = "Hello    world   test"
        cleaned = clean_text(text, remove_extra_whitespace=True)
        assert cleaned == "Hello world test"

    def test_clean_text_lowercase(self):
        """clean_text should lowercase text."""
        text = "Hello World"
        cleaned = clean_text(text, lowercase=True)
        assert cleaned == "hello world"

    def test_normalize_text(self):
        """normalize_text should normalize unicode."""
        # Test NFC normalization
        text = "cafe\u0301"  # 'e' + combining acute accent
        normalized = normalize_text(text)
        assert len(normalized) <= len(text)

    def test_chunk_text(self):
        """chunk_text should split text into chunks."""
        text = "The quick brown fox jumps over the lazy dog. " * 10
        chunks = chunk_text(text, chunk_size=50, overlap=10)

        assert len(chunks) > 1
        assert all(len(c) <= 50 for c in chunks)

    def test_chunk_text_short(self):
        """chunk_text should handle short text."""
        text = "Short"
        chunks = chunk_text(text, chunk_size=100)
        assert chunks == ["Short"]

    def test_chunk_text_empty(self):
        """chunk_text should handle empty text."""
        chunks = chunk_text("", chunk_size=100)
        assert chunks == []

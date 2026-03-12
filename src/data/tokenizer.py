"""BPE Tokenizer implementation wrapping SentencePiece."""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import sentencepiece as spm


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer wrapping SentencePiece.

    Provides training, encoding, and decoding functionality for
    subword tokenization.

    Special tokens:
        - <pad>: Padding token (id=0)
        - <unk>: Unknown token (id=1)
        - <s>: Beginning of sequence (id=2)
        - </s>: End of sequence (id=3)

    Example:
        >>> tokenizer = BPETokenizer()
        >>> tokenizer.train(texts, vocab_size=8000)
        >>> ids = tokenizer.encode("Hello world!")
        >>> text = tokenizer.decode(ids)
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Initialize tokenizer.

        Args:
            model_path: Path to a pre-trained SentencePiece model.
                       If None, must call train() before using.
        """
        self.sp_model = spm.SentencePieceProcessor()
        self._model_path = None

        if model_path is not None:
            self.load(model_path)

    def train(
        self,
        texts: List[str],
        vocab_size: int = 8000,
        model_prefix: str = "tokenizer",
        output_dir: Optional[Union[str, Path]] = None,
        character_coverage: float = 0.9995,
        model_type: str = "bpe",
    ) -> None:
        """
        Train a new tokenizer on the provided texts.

        Args:
            texts: List of text strings to train on.
            vocab_size: Target vocabulary size.
            model_prefix: Prefix for output model files.
            output_dir: Directory to save model. If None, uses temp directory.
            character_coverage: Character coverage for training.
            model_type: Model type ('bpe', 'unigram', 'word', 'char').

        Example:
            >>> texts = ["Hello world", "This is a test"]
            >>> tokenizer.train(texts, vocab_size=1000)
        """
        # Create output directory
        if output_dir is None:
            output_dir = tempfile.mkdtemp()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write texts to temporary file
        input_file = output_dir / "train_data.txt"
        with open(input_file, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(text.strip() + "\n")

        # Train SentencePiece model
        model_prefix_path = output_dir / model_prefix

        spm.SentencePieceTrainer.train(
            input=str(input_file),
            model_prefix=str(model_prefix_path),
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece=self.PAD_TOKEN,
            unk_piece=self.UNK_TOKEN,
            bos_piece=self.BOS_TOKEN,
            eos_piece=self.EOS_TOKEN,
        )

        # Load the trained model
        model_path = str(model_prefix_path) + ".model"
        self.load(model_path)

        # Cleanup temporary training file
        if input_file.exists():
            input_file.unlink()

    def load(self, model_path: Union[str, Path]) -> None:
        """
        Load a pre-trained SentencePiece model.

        Args:
            model_path: Path to the .model file.

        Raises:
            FileNotFoundError: If model file doesn't exist.
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Tokenizer model not found: {model_path}")

        self.sp_model.load(str(model_path))
        self._model_path = model_path

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the tokenizer model to a new location.

        Args:
            path: Destination path for the .model file.
        """
        if self._model_path is None:
            raise RuntimeError("No model loaded. Train or load a model first.")

        import shutil
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Copy model file (skip if source and destination are the same)
        if self._model_path.resolve() != path.resolve():
            shutil.copy(self._model_path, path)

        # Copy vocab file if it exists
        vocab_path = self._model_path.with_suffix(".vocab")
        dest_vocab = path.with_suffix(".vocab")
        if vocab_path.exists() and vocab_path.resolve() != dest_vocab.resolve():
            shutil.copy(vocab_path, dest_vocab)

        self._model_path = path

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text string to encode.
            add_bos: Add beginning-of-sequence token.
            add_eos: Add end-of-sequence token.

        Returns:
            List of token IDs.

        Example:
            >>> ids = tokenizer.encode("Hello world!")
            >>> ids_with_special = tokenizer.encode("Hello", add_bos=True, add_eos=True)
        """
        ids = self.sp_model.encode(text)

        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs.
            skip_special: If True, skip special tokens in output.

        Returns:
            Decoded text string.

        Example:
            >>> text = tokenizer.decode([123, 456, 789])
        """
        if skip_special:
            special_ids = {self.pad_id, self.bos_id, self.eos_id}
            ids = [i for i in ids if i not in special_ids]

        return self.sp_model.decode(ids)

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False
    ) -> List[List[int]]:
        """
        Encode multiple texts to token IDs.

        Args:
            texts: List of text strings.
            add_bos: Add beginning-of-sequence token.
            add_eos: Add end-of-sequence token.

        Returns:
            List of token ID lists.
        """
        return [self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]

    def decode_batch(self, ids_batch: List[List[int]], skip_special: bool = True) -> List[str]:
        """
        Decode multiple token ID sequences to text.

        Args:
            ids_batch: List of token ID lists.
            skip_special: If True, skip special tokens.

        Returns:
            List of decoded text strings.
        """
        return [self.decode(ids, skip_special=skip_special) for ids in ids_batch]

    def get_vocab(self) -> dict:
        """
        Get the vocabulary as a dictionary.

        Returns:
            Dictionary mapping tokens to IDs.
        """
        return {
            self.sp_model.id_to_piece(i): i
            for i in range(self.sp_model.get_piece_size())
        }

    def id_to_token(self, id: int) -> str:
        """Convert token ID to token string."""
        return self.sp_model.id_to_piece(id)

    def token_to_id(self, token: str) -> int:
        """Convert token string to token ID."""
        return self.sp_model.piece_to_id(token)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.sp_model.get_piece_size()

    @property
    def pad_id(self) -> int:
        """Padding token ID."""
        return self.sp_model.pad_id()

    @property
    def unk_id(self) -> int:
        """Unknown token ID."""
        return self.sp_model.unk_id()

    @property
    def bos_id(self) -> int:
        """Beginning-of-sequence token ID."""
        return self.sp_model.bos_id()

    @property
    def eos_id(self) -> int:
        """End-of-sequence token ID."""
        return self.sp_model.eos_id()


def load_tokenizer(path: Union[str, Path]) -> BPETokenizer:
    """
    Load a tokenizer from a saved model.

    Args:
        path: Path to the .model file.

    Returns:
        Loaded BPETokenizer instance.

    Example:
        >>> tokenizer = load_tokenizer("tokenizer.model")
    """
    return BPETokenizer(model_path=path)


def save_tokenizer(tokenizer: BPETokenizer, path: Union[str, Path]) -> None:
    """
    Save a tokenizer to disk.

    Args:
        tokenizer: BPETokenizer instance.
        path: Destination path for the .model file.

    Example:
        >>> save_tokenizer(tokenizer, "models/tokenizer.model")
    """
    tokenizer.save(path)

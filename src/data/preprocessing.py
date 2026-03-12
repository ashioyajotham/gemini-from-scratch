"""Text preprocessing utilities."""

import re
import unicodedata
from typing import List, Optional


def clean_text(
    text: str,
    lowercase: bool = False,
    remove_punctuation: bool = False,
    remove_numbers: bool = False,
    remove_extra_whitespace: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True
) -> str:
    """
    Clean and normalize text.

    Args:
        text: Input text string.
        lowercase: Convert to lowercase.
        remove_punctuation: Remove punctuation characters.
        remove_numbers: Remove numeric characters.
        remove_extra_whitespace: Collapse multiple whitespaces.
        remove_urls: Remove URLs.
        remove_emails: Remove email addresses.

    Returns:
        Cleaned text string.

    Example:
        >>> text = "Check out  https://example.com for more info!"
        >>> clean_text(text, remove_urls=True)
        'Check out for more info!'
    """
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

    if remove_emails:
        text = re.sub(r'\S+@\S+\.\S+', '', text)

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)

    if remove_numbers:
        text = re.sub(r'\d+', '', text)

    if remove_extra_whitespace:
        text = re.sub(r'\s+', ' ', text).strip()

    return text


def normalize_text(text: str) -> str:
    """
    Unicode normalize text (NFC form).

    Args:
        text: Input text string.

    Returns:
        Normalized text string.

    Example:
        >>> text = "café"  # with combining characters
        >>> normalize_text(text)  # normalized form
    """
    return unicodedata.normalize('NFC', text)


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int = 0,
    separator: str = " "
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text string.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between chunks.
        separator: Preferred split point (tries to break at this character).

    Returns:
        List of text chunks.

    Example:
        >>> text = "The quick brown fox jumps over the lazy dog."
        >>> chunks = chunk_text(text, chunk_size=20, overlap=5)
    """
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If not at the end, try to find a good break point
        if end < len(text):
            # Look for the separator within the last portion of the chunk
            search_start = max(start + chunk_size // 2, start)
            last_sep = text.rfind(separator, search_start, end)

            if last_sep > start:
                end = last_sep + 1  # Include the separator

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start, accounting for overlap
        start = end - overlap if overlap > 0 else end

    return chunks


def chunk_by_sentences(
    text: str,
    max_tokens: int,
    tokenizer=None
) -> List[str]:
    """
    Split text into chunks at sentence boundaries.

    Args:
        text: Input text string.
        max_tokens: Maximum tokens per chunk (approximate).
        tokenizer: Optional tokenizer for accurate token counting.

    Returns:
        List of text chunks.

    Example:
        >>> chunks = chunk_by_sentences(text, max_tokens=512, tokenizer=tokenizer)
    """
    # Simple sentence splitting
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Estimate token count
        if tokenizer:
            sent_tokens = len(tokenizer.encode(sentence))
        else:
            sent_tokens = len(sentence.split())

        # Check if adding this sentence would exceed the limit
        if current_length + sent_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += sent_tokens

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def remove_special_characters(text: str, keep_chars: Optional[str] = None) -> str:
    """
    Remove special characters from text.

    Args:
        text: Input text string.
        keep_chars: String of characters to keep (in addition to alphanumeric).

    Returns:
        Text with special characters removed.

    Example:
        >>> remove_special_characters("Hello, World!", keep_chars=",")
        'Hello, World'
    """
    if keep_chars:
        pattern = f'[^a-zA-Z0-9\\s{re.escape(keep_chars)}]'
    else:
        pattern = r'[^a-zA-Z0-9\s]'

    return re.sub(pattern, '', text)


def truncate_text(
    text: str,
    max_length: int,
    truncation_side: str = "right",
    ellipsis: str = "..."
) -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Input text string.
        max_length: Maximum character length.
        truncation_side: Which side to truncate ('left' or 'right').
        ellipsis: String to append/prepend to indicate truncation.

    Returns:
        Truncated text string.

    Example:
        >>> truncate_text("This is a long sentence.", max_length=15)
        'This is a lo...'
    """
    if len(text) <= max_length:
        return text

    if truncation_side == "right":
        return text[:max_length - len(ellipsis)] + ellipsis
    else:
        return ellipsis + text[-(max_length - len(ellipsis)):]

"""
Vocab files for BERT. This module fetches the requested vocab files and loads
a HuggingFace's tokenizer. Compared to tokenizer-cereal, we only provide a light
loading wrapper over it.

Source: https://github.com/huggingface/tokenizers/issues/59#issuecomment-593184936
"""

import tokenizers

from .dumps import download_to, get_filename_path, is_file

vocabs = {
    'bert-base-uncased':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    'bert-large-uncased':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
    'bert-base-cased':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    'bert-large-cased':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
    'bert-base-multilingual-uncased':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
    'bert-base-multilingual-cased':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
    'bert-base-chinese':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
    'bert-base-german-cased':
        "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-vocab.txt",
    'bert-large-uncased-whole-word-masking':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-vocab.txt",
    'bert-large-cased-whole-word-masking':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-vocab.txt",
    'bert-large-uncased-whole-word-masking-finetuned-squad':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt",
    'bert-large-cased-whole-word-masking-finetuned-squad':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-vocab.txt",
    'bert-base-cased-finetuned-mrpc':
        "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-vocab.txt"
}


def load_vocab(flavour: str) -> str:
    """
    Load a vocab file.

    :param flavour the specific vocab file to open
    :returns the path of the newly downloaded vocab file
    """
    path = f"bert/{flavour}.str"
    real_path = get_filename_path(path)

    assert flavour in vocabs.keys(), f"Flavour not listed in {', '.join(vocabs.keys())}"

    if not is_file(path):
        download_to(vocabs[flavour], path)

    return real_path


def load_tokenizer(flavour: str) -> tokenizers.BertWordPieceTokenizer:
    """
    Load a BertForWordPiece Tokenizer given a flavour.

    :param flavour the specific vocab file to open
    :returns the path of the newly downloaded vocab file
    """
    vocab_path = load_vocab(flavour)
    return tokenizers.BertWordPieceTokenizer(vocab_path)

"""Declared tokenizer selection; character counting is explicitly smoke-only."""
from importlib.metadata import version


def load_tokenizer(configuration):
    name = configuration.get('name', '')
    pinned = configuration.get('version')
    if name == 'characters-smoke':
        if pinned != '1':
            raise ValueError('characters-smoke version must be 1')
        return len
    if not name.startswith('tiktoken:'):
        raise ValueError('supported tokenizers: tiktoken:<encoding> or characters-smoke')
    import tiktoken
    installed = version('tiktoken')
    if pinned != installed:
        raise ValueError(f'tiktoken version mismatch: declared {pinned}, installed {installed}')
    encoding = tiktoken.get_encoding(name.split(':', 1)[1])
    return lambda text: len(encoding.encode(text, disallowed_special=()))

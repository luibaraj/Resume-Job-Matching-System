import warnings

# Suppress deprecation warnings from torchao (used transitively via unsloth)
# See: https://github.com/pytorch/ao/issues/2752
warnings.filterwarnings(
    'ignore',
    message=r'Importing from torchao\.dtypes\.uintx.*is deprecated',
    category=DeprecationWarning,
)
warnings.filterwarnings(
    'ignore',
    message='Importing BlockSparseLayout from torchao.dtypes is deprecated',
    category=DeprecationWarning,
)
warnings.filterwarnings(
    'ignore',
    message='.*builtin type [a-zA-Z_]* has no __module__ attribute',
    category=DeprecationWarning,
)

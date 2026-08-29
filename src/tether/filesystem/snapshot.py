"""Re-export of the FileSnapshot model for the filesystem package.

The canonical implementation lives in ``tether.memory.file_snapshot``;
this module keeps the documented filesystem/ layout working.
"""

from tether.memory.file_snapshot import (  # noqa: F401
    FileSnapshot,
    compute_md5,
    extract_symbols,
    infer_language,
)

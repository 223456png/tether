"""Symbol-level diff utilities used by drift detection."""



def compare_symbols(old_symbols: list[str], new_symbols: list[str]) -> dict[str, object]:
    """Compare two symbol lists.

    Returns added/removed/common symbol names plus a ``changed`` flag that
    is True whenever symbols were added or removed.
    """
    old_set = set(old_symbols)
    new_set = set(new_symbols)
    added = [s for s in new_symbols if s not in old_set]
    removed = [s for s in old_symbols if s not in new_set]
    common = [s for s in old_symbols if s in new_set]
    return {
        "added": added,
        "removed": removed,
        "common": common,
        "changed": bool(added or removed),
    }

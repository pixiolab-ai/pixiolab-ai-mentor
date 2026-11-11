def chunk(tokens: list[str], size: int = 400, overlap: int = 60) -> list[dict]:
    """Fixed-size windows. Neighbors share `overlap` tokens.
    Offsets are kept so a citation can point back into the source."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    if not tokens:
        return []
    step, out = size - overlap, []
    for start in range(0, len(tokens), step):
        end = min(start + size, len(tokens))
        out.append({"start": start, "end": end, "text": tokens[start:end]})
        if end == len(tokens):
            break                     # last window reached the end
    return out

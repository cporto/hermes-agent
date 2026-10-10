_JSON_MISSING_ESCAPE = {
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\b": "\\b",
    "\f": "\\f",
}


def escape_raw_control_chars(raw: str) -> str:
    """Escape raw control characters that appear INSIDE JSON string values.

    LLM tool-call payloads frequently carry literal newlines/tabs inside string
    fields instead of the ``\\n`` escape the JSON spec requires. ``json.loads``
    rejects those as unescaped control characters. This recovery only rewrites the
    control chars inside string values and leaves the rest of the document
    untouched (a real ``\\n`` escape is a two-char ``\\``+``n`` sequence and is left
    alone). A literal backslash sets `escaped` so the next char (even a control
    char) is copied verbatim — this also correctly preserves a literal ``\\``
    followed by a newline.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in raw:
        if escaped:
            out.append(ch)
            escaped = False
        elif ch == "\\":
            out.append(ch)
            escaped = True
        elif ch == '"':
            in_string = not in_string
            out.append(ch)
        elif in_string and ch in _JSON_MISSING_ESCAPE:
            out.append(_JSON_MISSING_ESCAPE[ch])
        else:
            out.append(ch)
    return "".join(out)

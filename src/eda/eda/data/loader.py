"""Parse FabCon / SQLCON session markdown files into pandas DataFrames."""
from __future__ import annotations

import re
from pathlib import Path

import frontmatter
import pandas as pd

_WIKI_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_wikilinks(value: object) -> object:
    """Unwrap ``[[wiki links]]`` inside strings or lists of strings."""
    if isinstance(value, str):
        links = _WIKI_RE.findall(value)
        if len(links) == 1:
            return links[0]
        if links:
            return links
        return value
    if isinstance(value, list):
        return [_extract_wikilinks(v) for v in value]
    return value


def _parse_description(body: str) -> str:
    """Extract text after a ``## Description`` heading in the markdown body."""
    m = re.search(
        r"(?:^|\n)#{1,3}\s+Description\s*\n+(.*)",
        body,
        re.DOTALL,
    )
    return m.group(1).strip() if m else body.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_vault(
    vault_path: str | Path,
    *,
    include_workshops: bool = True,
    include_sessions: bool = True,
) -> pd.DataFrame:
    """Load all session / workshop markdown files from the vault.

    Parameters
    ----------
    vault_path:
        Root of the Obsidian vault (contains ``Sessions/`` and ``Workshops/``).
    include_workshops:
        Whether to recurse into ``Workshops/``.
    include_sessions:
        Whether to recurse into ``Sessions/``.

    Returns
    -------
    pd.DataFrame
        One row per session.  YAML front-matter keys become columns.
        Additional columns: ``file``, ``file_path``, ``description``, ``text``.
    """
    vault = Path(vault_path).resolve()
    files: list[Path] = []

    if include_sessions and (vault / "Sessions").is_dir():
        files.extend(sorted((vault / "Sessions").rglob("*.md")))
    if include_workshops and (vault / "Workshops").is_dir():
        files.extend(sorted((vault / "Workshops").rglob("*.md")))

    rows: list[dict] = []
    for path in files:
        try:
            post = frontmatter.load(str(path))
        except Exception:  # malformed YAML — skip silently
            continue

        row: dict = {"file": path.name, "file_path": str(path)}

        for key, value in post.metadata.items():
            row[key] = _extract_wikilinks(value)

        description = _parse_description(post.content)
        row["description"] = description
        row["text"] = f"{row.get('title', '')} {description}".strip()

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ---- type coercions ------------------------------------------------
    for col in ("level", "duration", "interest"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Strip " Track" suffix from track labels (e.g. "Power BI Track" → "Power BI")
    if "track" in df.columns:
        df["track"] = df["track"].apply(
            lambda v: re.sub(r"\s+Track$", "", v) if isinstance(v, str) else v
        )

    # Ensure list-typed columns are always lists
    for col in ("speakers", "audience", "tags"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: v
                if isinstance(v, list)
                else ([str(v)] if (v is not None and v != "") else [])
            )

    return df.reset_index(drop=True)

"""Generate the configuration reference page from ``config.yaml``.

The settings are already documented once, in the comments of ``config.yaml``,
and those comments are written for the people who actually edit that file.
Copying them into a docs page by hand would create a second copy to keep in
step, and the two would drift. So the page is generated from the file itself
at build time, and ``tests/test_docs_config.py`` fails the build if a setting
exists in the dataclass but not in the YAML.

The parser is deliberately small. It relies on the layout conventions already
used in ``config.yaml``:

* ``# ---- Title ----`` starts a section.
* ``# ==== ... ====`` around a title starts a major part (used for ADVANCED).
* Comment lines directly above a ``key: value`` line document that key.
* Comment lines after a section header and followed by a blank line document
  the section as a whole.
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

SECTION_RE = re.compile(r"^#\s*-{2,}\s*(.+?)\s*-{2,}\s*$")
BANNER_RE = re.compile(r"^#\s*={4,}\s*$")
KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z_0-9]*)\s*:\s*(.*?)\s*$")
COMMENT_RE = re.compile(r"^#\s?(.*)$")


class Setting(NamedTuple):
    """One documented entry, which may cover more than one key.

    Some settings only make sense as a pair and are commented once in
    config.yaml, e.g. ``night_start_hour`` / ``night_end_hour``. Adjacent keys
    with no blank line between them are treated as sharing that comment rather
    than leaving the second one undocumented.
    """

    names: List[str]
    defaults: List[str]
    doc: str

    @property
    def name(self) -> str:
        return self.names[0]


class Section(NamedTuple):
    title: str
    intro: str
    settings: List[Setting]
    advanced: bool


def _flow(lines: List[str]) -> str:
    """Join comment lines into paragraphs, preserving blank-line breaks."""
    paragraphs: List[List[str]] = [[]]
    for line in lines:
        if line.strip():
            paragraphs[-1].append(line.strip())
        elif paragraphs[-1]:
            paragraphs.append([])
    return "\n\n".join(" ".join(p) for p in paragraphs if p)


def parse_config_yaml(path: Path) -> List[Section]:
    """Read config.yaml into sections of documented settings."""
    sections: List[Section] = []
    current: Optional[Section] = None
    pending: List[str] = []
    in_header = False
    advanced = False
    banner_open = False
    previous_was_key = False

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()

        if BANNER_RE.match(line):
            # A '# ====' rule; the title sits between two of them.
            banner_open = not banner_open
            pending = []
            continue

        section_match = SECTION_RE.match(line)
        if section_match:
            current = Section(section_match.group(1), "", [], advanced)
            sections.append(current)
            pending = []
            in_header = True
            previous_was_key = False
            continue

        comment_match = COMMENT_RE.match(line)
        if comment_match:
            text = comment_match.group(1)
            if banner_open and text.strip():
                # Title inside a '# ====' banner, e.g. ADVANCED.
                advanced = advanced or text.strip().upper() == "ADVANCED"
                continue
            pending.append(text)
            previous_was_key = False
            continue

        if not line.strip():
            if in_header and pending and current is not None:
                idx = sections.index(current)
                current = current._replace(intro=_flow(pending))
                sections[idx] = current
                in_header = False
            pending = []
            previous_was_key = False
            continue

        key_match = KEY_RE.match(line)
        if key_match and current is not None:
            name, default = key_match.group(1), key_match.group(2) or "~"
            if previous_was_key and current.settings:
                # Directly under the previous key with nothing between them:
                # they share one comment, so document them as one entry.
                last = current.settings[-1]
                last.names.append(name)
                last.defaults.append(default)
            else:
                current.settings.append(Setting([name], [default], _flow(pending)))
            pending = []
            in_header = False
            previous_was_key = True
            continue

        pending = []
        previous_was_key = False

    return [s for s in sections if s.settings]


def setting_names(sections: List[Section]) -> set:
    """Every individual key documented across all sections."""
    return {
        name
        for section in sections
        for setting in section.settings
        for name in setting.names
    }


def _defaults_from_dataclass() -> Dict[str, str]:
    from cardiacprep.config import Config

    return {f.name: repr(getattr(Config(), f.name)) for f in fields(Config)}


def render(sections: List[Section]) -> str:
    """Render the parsed sections as a MyST Markdown page."""
    out: List[str] = [
        "<!-- GENERATED FILE - DO NOT EDIT.",
        "     Produced from config.yaml by docs/source/_generate.py at build",
        "     time. Edit the comments in config.yaml instead. -->",
        "",
        "# Configuration reference",
        "",
        "Every setting lives in `config.yaml` at the root of the project. All of",
        "them are optional: delete a line and the pipeline falls back to the",
        "default shown here. You should never need to edit a `.py` file.",
        "",
        "Command-line flags override the file, which overrides these defaults:",
        "",
        "```text",
        "dataclass defaults  ->  config.yaml  ->  command-line flags",
        "```",
        "",
        "A mistake in the file stops the run immediately, before any recording is",
        "processed, with a message naming the setting at fault.",
        "",
    ]

    basic = [s for s in sections if not s.advanced]
    advanced = [s for s in sections if s.advanced]

    def emit(group: List[Section]) -> None:
        for section in group:
            out.append(f"### {section.title}")
            out.append("")
            if section.intro:
                out.append(section.intro)
                out.append("")
            for setting in section.settings:
                out.append(", ".join(f"`{n}`" for n in setting.names))
                out.append("")
                defaults = ", ".join(
                    f"`{n}: {d}`" for n, d in zip(setting.names, setting.defaults)
                )
                out.append(f": **Default:** {defaults}")
                out.append("")
                if setting.doc:
                    for paragraph in setting.doc.split("\n\n"):
                        out.append(f"  {paragraph}")
                        out.append("")

    out.append("## Everyday settings")
    out.append("")
    emit(basic)

    if advanced:
        out.append("## Advanced settings")
        out.append("")
        out.append(
            "These control the signal processing itself. The defaults match the "
            "conditions the bundled QRS detector was trained under, so changing "
            "them changes what the model sees. Check the effect on detection "
            "before trusting the results."
        )
        out.append("")
        out.append(
            "The exception is `mains_hz`, which you should set correctly for "
            "your country regardless."
        )
        out.append("")
        emit(advanced)

    return "\n".join(out) + "\n"


def write_configuration_page(source_dir: Path, config_path: Path) -> Path:
    sections = parse_config_yaml(config_path)
    documented = setting_names(sections)

    try:
        defaults = _defaults_from_dataclass()
    except Exception:  # noqa: BLE001 - docs build must not depend on imports
        defaults = {}

    missing = sorted(set(defaults) - documented)
    if missing:
        raise RuntimeError(
            "These settings exist in Config but are absent from config.yaml, "
            "so they would be missing from the documentation: "
            + ", ".join(missing)
        )

    target = source_dir / "configuration.md"
    target.write_text(render(sections), encoding="utf-8")
    return target

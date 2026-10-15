"""List and read evaluation skills without selecting one for the model."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9-]+$")
FRONTMATTER_PATTERN = re.compile(r"\A---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|\Z)", re.DOTALL)


class SkillReaderError(ValueError):
    """Report invalid skill metadata or a disallowed skill path."""


class SkillReader:
    """Provide read-only access to Markdown files inside evaluation skills."""

    def __init__(self, skills_root: Path):
        """Initialize the skill directory boundary.

        Arg:
            skills_root: Directory containing one folder per evaluation skill.

        Returns:
            None.
        """
        self.skills_root = skills_root.resolve()
        if not self.skills_root.is_dir():
            raise SkillReaderError(f"Skills directory not found: {self.skills_root}")

    def list_skills(self) -> list[dict[str, str]]:
        """List skill names and descriptions without recommending a skill.

        Arg:
            None.

        Returns:
            Skill metadata sorted by skill name.
        """
        skills: list[dict[str, str]] = []
        for skill_directory in self.skills_root.iterdir():
            if not skill_directory.is_dir():
                continue
            skill_path = skill_directory / "SKILL.md"
            if not skill_path.is_file():
                continue
            metadata = self._read_frontmatter(skill_path)
            skill_name = metadata["name"]
            if skill_name != skill_directory.name:
                raise SkillReaderError(
                    f"Skill name '{skill_name}' does not match folder "
                    f"'{skill_directory.name}'."
                )
            skills.append(
                {
                    "name": skill_name,
                    "description": metadata["description"],
                }
            )
        return sorted(skills, key=lambda item: item["name"])

    def read_skill(
        self,
        skill_name: str,
        resource: str = "SKILL.md",
    ) -> dict[str, str]:
        """Read one Markdown file selected by the caller.

        Arg:
            skill_name: Exact evaluation skill folder name.
            resource: Skill-relative Markdown path, defaulting to SKILL.md.

        Returns:
            Skill name, normalized resource path, and Markdown content.
        """
        skill_directory = self._resolve_skill_directory(skill_name)
        requested = Path(str(resource))
        if requested.is_absolute():
            raise SkillReaderError("Absolute resource paths are not allowed.")
        if requested.suffix.lower() != ".md":
            raise SkillReaderError("Only Markdown skill resources are allowed.")

        candidate = (skill_directory / requested).resolve()
        try:
            candidate.relative_to(skill_directory)
        except ValueError as exc:
            raise SkillReaderError(
                "A skill resource must remain inside its skill directory."
            ) from exc
        if not candidate.is_file():
            raise SkillReaderError(
                f"Skill resource not found: {skill_name}/{requested.as_posix()}"
            )

        return {
            "skill_name": skill_name,
            "resource": candidate.relative_to(skill_directory).as_posix(),
            "content": candidate.read_text(encoding="utf-8"),
        }

    def _resolve_skill_directory(self, skill_name: str) -> Path:
        """Resolve and validate one exact skill folder.

        Arg:
            skill_name: Requested skill name.

        Returns:
            Resolved skill directory below the skills root.
        """
        normalized_name = str(skill_name).strip()
        if not SKILL_NAME_PATTERN.fullmatch(normalized_name):
            raise SkillReaderError(
                "Skill names may contain only lowercase letters, digits, and hyphens."
            )

        candidate = (self.skills_root / normalized_name).resolve()
        try:
            candidate.relative_to(self.skills_root)
        except ValueError as exc:
            raise SkillReaderError("Skill path must remain inside the skills directory.") from exc
        if not candidate.is_dir() or not (candidate / "SKILL.md").is_file():
            raise SkillReaderError(f"Skill not found: {normalized_name}")
        return candidate

    def _read_frontmatter(self, skill_path: Path) -> dict[str, str]:
        """Parse the required name and description from SKILL.md.

        Arg:
            skill_path: Path to one skill's main Markdown file.

        Returns:
            Validated name and description strings.
        """
        text = skill_path.read_text(encoding="utf-8")
        match = FRONTMATTER_PATTERN.match(text)
        if match is None:
            raise SkillReaderError(f"Missing YAML frontmatter: {skill_path}")

        try:
            metadata = yaml.safe_load(match.group(1))
        except yaml.YAMLError as exc:
            raise SkillReaderError(f"Invalid YAML frontmatter: {skill_path}") from exc
        if not isinstance(metadata, dict):
            raise SkillReaderError(f"Frontmatter must be a mapping: {skill_path}")

        name = metadata.get("name")
        description = metadata.get("description")
        if not isinstance(name, str) or not SKILL_NAME_PATTERN.fullmatch(name):
            raise SkillReaderError(f"Invalid skill name in frontmatter: {skill_path}")
        if not isinstance(description, str) or not description.strip():
            raise SkillReaderError(f"Missing skill description: {skill_path}")
        return {
            "name": name,
            "description": description.strip(),
        }


class ListSkillsTool:
    """Expose SkillReader.list_skills as a provider-neutral tool."""

    def __init__(self, reader: SkillReader):
        """Initialize the list-skills tool.

        Arg:
            reader: Configured evaluation skill reader.

        Returns:
            None.
        """
        self.reader = reader

    @property
    def name(self) -> str:
        """Return the provider-neutral tool name.

        Arg:
            None.

        Returns:
            Tool name used in model tool calls.
        """
        return "list_skills"

    @property
    def description(self) -> str:
        """Return the provider-neutral tool description.

        Arg:
            None.

        Returns:
            Tool instructions shown to each model.
        """
        return "List available raw-data skills and their descriptions."

    @property
    def input_schema(self) -> dict[str, Any]:
        """Return the provider-neutral JSON input schema.

        Arg:
            None.

        Returns:
            JSON schema with no accepted arguments.
        """
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    def run(self) -> list[dict[str, str]]:
        """List available evaluation skills.

        Arg:
            None.

        Returns:
            Skill names and descriptions.
        """
        return self.reader.list_skills()


class ReadSkillTool:
    """Expose caller-selected skill Markdown as a provider-neutral tool."""

    def __init__(self, reader: SkillReader):
        """Initialize the read-skill tool.

        Arg:
            reader: Configured evaluation skill reader.

        Returns:
            None.
        """
        self.reader = reader

    @property
    def name(self) -> str:
        """Return the provider-neutral tool name.

        Arg:
            None.

        Returns:
            Tool name used in model tool calls.
        """
        return "read_skill"

    @property
    def description(self) -> str:
        """Return the provider-neutral tool description.

        Arg:
            None.

        Returns:
            Tool instructions shown to each model.
        """
        return (
            "Read the main SKILL.md or one Markdown reference from a skill "
            "selected by the model."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        """Return the provider-neutral JSON input schema.

        Arg:
            None.

        Returns:
            JSON schema for a skill name and optional resource path.
        """
        return {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Exact skill name returned by list_skills.",
                },
                "resource": {
                    "type": "string",
                    "description": (
                        "Skill-relative Markdown path. Omit to read SKILL.md."
                    ),
                    "default": "SKILL.md",
                },
            },
            "required": ["skill_name"],
            "additionalProperties": False,
        }

    def run(
        self,
        skill_name: str,
        resource: str = "SKILL.md",
    ) -> dict[str, str]:
        """Read one caller-selected skill Markdown file.

        Arg:
            skill_name: Exact evaluation skill name.
            resource: Skill-relative Markdown path.

        Returns:
            Selected skill resource and its content.
        """
        return self.reader.read_skill(skill_name, resource)

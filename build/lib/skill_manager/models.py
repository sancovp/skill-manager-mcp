"""Pydantic models for skill manager."""

from typing import Optional, Literal
from pydantic import BaseModel


# Skill categories determine how the agent should use the skill
SkillCategory = Literal["understand", "preflight", "single_turn_process"]


class ClaudeSkillFrontmatter(BaseModel):
    """Claude's official skill frontmatter fields (SKILL.md YAML)."""
    name: str  # max 64 chars, lowercase/numbers/hyphens only
    description: str  # max 1024 chars - THIS IS THE TRIGGER
    allowed_tools: Optional[str] = None  # comma-separated tool names
    model: Optional[str] = None  # e.g. claude-sonnet-4-20250514


class GnosysSkillMetadata(BaseModel):
    """GNOSYS extensions stored in _metadata.json."""
    domain: str  # PAIAB, SANCTUM, CAVE
    subdomain: Optional[str] = None
    category: Optional[SkillCategory] = None
    # Typed description fields - concatenated into Claude's description
    what: str  # What this skill does
    when: str  # When to use it (trigger condition)
    # how is the skill body for preflight, not stored here


class Skill(BaseModel):
    """A skill combining Claude's format + GNOSYS extensions."""
    # Claude's fields
    name: str
    description: str  # Built from WHAT/WHEN (and HOW for preflight is the body)
    allowed_tools: Optional[str] = None
    model: Optional[str] = None
    # GNOSYS extensions
    domain: str
    subdomain: Optional[str] = None
    category: Optional[SkillCategory] = None
    what: str
    when: str
    # Skill body content
    content: str


class Skillset(BaseModel):
    """A named group of skills with its own domain."""
    name: str
    domain: str
    subdomain: Optional[str] = None
    description: str
    skills: list[str]  # skill names


class Persona(BaseModel):
    """A composable persona bundling frame, MCP set, skillset, and identity."""
    name: str
    domain: str
    subdomain: Optional[str] = None
    description: str
    frame: str  # cognitive frame / prompt text
    mcp_set: Optional[str] = None  # strata set name (aspirational)
    skillset: Optional[str] = None  # skillset name (aspirational)
    carton_identity: Optional[str] = None  # CartON identity for observations

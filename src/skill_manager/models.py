"""Pydantic models for skill manager."""

from typing import Optional, Literal
from pydantic import BaseModel


# Skill categories determine how the agent should use the skill
SkillCategory = Literal["understand", "preflight", "single_turn_process"]


class Skill(BaseModel):
    """A skill with content and metadata."""
    name: str
    domain: str
    subdomain: Optional[str] = None
    description: str
    content: str
    category: Optional[SkillCategory] = None  # understand | preflight | single_turn_process


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

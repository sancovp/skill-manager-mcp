"""Skill Manager MCP Server - three-tier architecture: global/equipped/sets."""

import logging
from mcp.server.fastmcp import FastMCP

from .core import SkillManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("skill-manager")
manager = SkillManager()


# === Global catalog ===

@mcp.tool()
def list_skills() -> str:
    """List all skills in global catalog."""
    skills = manager.list_skills()
    if not skills:
        return "No skills in catalog. Create with create_skill."

    lines = []
    for s in skills:
        path = f"{s['domain']}::{s['subdomain']}" if s['subdomain'] else s['domain']
        lines.append(f"{s['name']}: {path} - {s['description']}")
    return "\n".join(lines)


@mcp.tool()
def list_domains() -> str:
    """List all available skill domains."""
    domains = manager.list_domains()
    if not domains:
        return "No domains. Create skills to populate."
    return "Domains: " + ", ".join(domains)


@mcp.tool()
def list_by_domain(domain: str) -> str:
    """List all skills and skillsets in a domain."""
    result = manager.list_by_domain(domain)
    lines = [f"Domain: {domain}"]

    if result["skillsets"]:
        lines.append("\nSkillsets:")
        for ss in result["skillsets"]:
            lines.append(f"  {ss['name']}: {ss['description']} ({ss['skill_count']} skills)")

    if result["skills"]:
        lines.append("\nSkills:")
        for s in result["skills"]:
            lines.append(f"  {s['name']}: {s['description']}")

    if not result["skillsets"] and not result["skills"]:
        lines.append("(empty)")

    return "\n".join(lines)


@mcp.tool()
def get_skill(name: str) -> str:
    """Get full content of a skill."""
    skill = manager.get_skill(name)
    if not skill:
        return f"Skill '{name}' not found"

    path = f"{skill.domain}::{skill.subdomain}" if skill.subdomain else skill.domain
    return f"# {skill.name}\nDomain: {path}\n\n{skill.content}"


@mcp.tool()
def create_skill(name: str, domain: str, content: str, description: str, subdomain: str = "") -> str:
    """Create a skill in global catalog.

    Args:
        name: Skill name (kebab-case)
        domain: Primary domain
        content: Skill content/instructions
        description: Brief description
        subdomain: Optional subdomain
    """
    skill = manager.create_skill(name, domain, content, description, subdomain or None)
    path = f"{skill.domain}::{skill.subdomain}" if skill.subdomain else skill.domain
    return f"Created skill '{skill.name}' in {path}"


@mcp.tool()
def search_skills(query: str, n_results: int = 5) -> str:
    """Search skills and skillsets using RAG."""
    matches = manager.search_skills(query, n_results)
    if not matches:
        return "No matches"

    lines = []
    for m in matches:
        path = f"{m['domain']}::{m['subdomain']}" if m['subdomain'] else m['domain']
        lines.append(f"[{m['score']:.2f}] {m['name']} ({m['type']}) - {path}")
    return "\n".join(lines)


# === Equipped state ===

@mcp.tool()
def list_equipped() -> str:
    """List currently equipped skills. Call this to see what knowledge is loaded."""
    equipped = manager.list_equipped()
    if not equipped:
        return "No skills equipped. Use equip(name) to load skills."

    lines = ["Equipped skills:"]
    for s in equipped:
        path = f"{s['domain']}::{s['subdomain']}" if s['subdomain'] else s['domain']
        lines.append(f"  {s['name']}: {path}")
    return "\n".join(lines)


@mcp.tool()
def get_equipped_content() -> str:
    """Get full content of all equipped skills."""
    return manager.get_equipped_content()


@mcp.tool()
def equip(name: str) -> str:
    """Equip a skill or skillset. Loads it into working memory.

    Args:
        name: Skill or skillset name to equip
    """
    result = manager.equip(name)
    if "error" in result:
        return result["error"]

    if result["type"] == "skillset":
        return f"Equipped skillset '{name}' ({result['domain']}): {', '.join(result['skills'])}"
    return f"Equipped skill '{name}' ({result['domain']})"


@mcp.tool()
def unequip(name: str) -> str:
    """Unequip a skill."""
    result = manager.unequip(name)
    if "error" in result:
        return result["error"]
    return f"Unequipped '{name}'"


@mcp.tool()
def unequip_all() -> str:
    """Clear all equipped skills."""
    result = manager.unequip_all()
    return f"Unequipped {result['unequipped_count']} skills"


# === Skillsets ===

@mcp.tool()
def list_skillsets() -> str:
    """List all skillsets."""
    skillsets = manager.list_skillsets()
    if not skillsets:
        return "No skillsets. Create with create_skillset."

    lines = []
    for ss in skillsets:
        path = f"{ss['domain']}::{ss['subdomain']}" if ss['subdomain'] else ss['domain']
        lines.append(f"{ss['name']}: {path} - {ss['description']} ({ss['skill_count']} skills)")
    return "\n".join(lines)


@mcp.tool()
def create_skillset(name: str, domain: str, description: str, skills: str, subdomain: str = "") -> str:
    """Create a skillset with domain.

    Args:
        name: Skillset name
        domain: Primary domain
        description: What this skillset is for
        skills: Comma-separated skill names
        subdomain: Optional subdomain
    """
    skill_list = [s.strip() for s in skills.split(",") if s.strip()]
    ss = manager.create_skillset(name, domain, description, skill_list, subdomain or None)
    path = f"{ss.domain}::{ss.subdomain}" if ss.subdomain else ss.domain
    return f"Created skillset '{ss.name}' in {path} with {len(ss.skills)} skills"


@mcp.tool()
def add_to_skillset(skillset_name: str, skill_name: str) -> str:
    """Add a skill to a skillset."""
    result = manager.add_to_skillset(skillset_name, skill_name)
    if "error" in result:
        return result["error"]
    return f"Added '{skill_name}' to '{skillset_name}'"


# === SkillLog matching ===

@mcp.tool()
def match_skilllog(prediction: str) -> str:
    """Match a SkillLog prediction against catalog.

    Args:
        prediction: SkillLog like "domain::subdomain::specific"
    """
    result = manager.match_skilllog(prediction)

    lines = [f"SkillLog: {result['prediction']}"]

    if result['has_match']:
        lines.append("Matches:")
        for m in result['matches'][:3]:
            lines.append(f"  [{m['score']:.2f}] {m['name']} ({m['type']}, {m['domain']})")
    else:
        lines.append("No strong matches.")

    if result['available_domains']:
        lines.append(f"Domains: {', '.join(result['available_domains'])}")

    return "\n".join(lines)


# === Personas ===

@mcp.tool()
def list_personas() -> str:
    """List all personas."""
    personas = manager.list_personas()
    if not personas:
        return "No personas. Create with create_persona."

    lines = []
    for p in personas:
        path = f"{p['domain']}::{p['subdomain']}" if p['subdomain'] else p['domain']
        lines.append(f"{p['name']}: {path} - {p['description']}")
        if p['mcp_set']:
            lines.append(f"  MCP set: {p['mcp_set']}")
        if p['skillset']:
            lines.append(f"  Skillset: {p['skillset']}")
    return "\n".join(lines)


@mcp.tool()
def create_persona(name: str, domain: str, description: str, frame: str,
                   mcp_set: str = "", skillset: str = "",
                   carton_identity: str = "", subdomain: str = "") -> str:
    """Create a persona bundling frame, MCP set, skillset, and identity.

    Args:
        name: Persona name
        domain: Primary domain
        description: What this persona is for
        frame: Cognitive frame / prompt text (how to think)
        mcp_set: Strata MCP set name (aspirational - doesn't need to exist)
        skillset: Skillset name (aspirational - doesn't need to exist)
        carton_identity: CartON identity for observations (defaults to name)
        subdomain: Optional subdomain
    """
    p = manager.create_persona(
        name, domain, description, frame,
        mcp_set=mcp_set or None,
        skillset=skillset or None,
        carton_identity=carton_identity or None,
        subdomain=subdomain or None
    )
    path = f"{p.domain}::{p.subdomain}" if p.subdomain else p.domain
    return f"Created persona '{p.name}' in {path}"


@mcp.tool()
def equip_persona(name: str) -> str:
    """Equip a persona - loads frame, attempts skillset, reports MCP set needs.

    Missing components are aspirational - they signal what needs to be created.
    """
    result = manager.equip_persona(name)
    if "error" in result:
        return result["error"]

    lines = [f"Persona '{name}' equipped:"]
    lines.append(f"✓ Frame loaded")
    lines.append(f"✓ Identity: {result['carton_identity']}")

    if result['skillset'] == "equipped":
        lines.append(f"✓ Skillset equipped: {', '.join(result['equipped_skills'])}")

    if result['mcp_set']:
        lines.append(f"→ MCP set '{result['mcp_set']['name']}': {result['mcp_set']['action']}")

    if result['missing']:
        lines.append("\nMissing (aspirational):")
        for m in result['missing']:
            lines.append(f"  ✗ {m['type']}: {m['name']} - {m['suggestion']}")

    lines.append(f"\n--- Frame ---\n{result['frame_content']}")
    return "\n".join(lines)


@mcp.tool()
def get_active_persona() -> str:
    """Get the currently active persona."""
    result = manager.get_active_persona()
    if not result:
        return "No active persona."
    return f"Active: {result['name']} ({result['domain']})"


@mcp.tool()
def deactivate_persona() -> str:
    """Deactivate current persona and unequip all skills."""
    result = manager.deactivate_persona()
    if "status" in result:
        return result["status"]
    return f"Deactivated '{result['deactivated']}', skills unequipped"


def main():
    """Run the MCP server."""
    logger.info("Starting Skill Manager MCP")
    mcp.run()


if __name__ == "__main__":
    main()

"""Core skill management logic with three-tier architecture: global/equipped/sets."""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings

from .models import Skill, Skillset, Persona

logger = logging.getLogger(__name__)

# Claude Code's native skills directory - we mirror equipped skills here
CLAUDE_SKILLS_DIR = Path.home() / ".claude" / "skills"


class SkillManager:
    """Manages skills with global catalog, equipped state, and skillsets."""

    def __init__(self, skills_dir: Optional[str] = None, chroma_dir: Optional[str] = None):
        # Use HEAVEN_DATA_DIR env var, fallback to ~/.heaven_data
        heaven_data = os.environ.get("HEAVEN_DATA_DIR", os.path.expanduser("~/.heaven_data"))
        self.skills_dir = Path(skills_dir or os.path.join(heaven_data, "skills"))
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Skills directory: {self.skills_dir}")

        # Skillsets and personas config
        self.skillsets_file = self.skills_dir / "_skillsets.json"
        self.personas_file = self.skills_dir / "_personas.json"

        # Defaults and quarantine
        self.defaults_file = self.skills_dir / "_defaults.json"
        self.quarantine_dir = self.skills_dir / "_quarantine"

        # Equipped state (in-memory for session)
        self.equipped: dict[str, Skill] = {}
        self.active_persona: Optional[Persona] = None

        # ChromaDB for RAG
        chroma_path = chroma_dir or os.path.join(heaven_data, "skill_chroma")
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="skills",
            metadata={"hnsw:space": "cosine"}
        )

        # Sync on startup
        self.sync_on_startup()

    # === File paths ===

    def _skill_path(self, name: str) -> Path:
        return self.skills_dir / name

    def _skill_md_path(self, name: str) -> Path:
        return self._skill_path(name) / "SKILL.md"

    def _metadata_path(self, name: str) -> Path:
        return self._skill_path(name) / "_metadata.json"

    # === Skill CRUD ===

    def _parse_skill_md(self, content: str) -> dict:
        """Parse SKILL.md frontmatter. Returns dict with Claude's fields + body.

        Claude's official frontmatter fields:
        - name (required)
        - description (required) - THIS IS THE TRIGGER
        - allowed-tools (optional)
        - model (optional)
        """
        lines = content.strip().split("\n")
        if lines[0] != "---":
            return {"name": "", "description": "", "body": content}

        end_idx = None
        for i, line in enumerate(lines[1:], 1):
            if line == "---":
                end_idx = i
                break

        if not end_idx:
            return {"name": "", "description": "", "body": content}

        frontmatter = "\n".join(lines[1:end_idx])
        body = "\n".join(lines[end_idx + 1:]).strip()

        result = {"name": "", "description": "", "allowed_tools": None, "model": None, "body": body}
        for line in frontmatter.split("\n"):
            if line.startswith("name:"):
                result["name"] = line.split(":", 1)[1].strip()
            elif line.startswith("description:"):
                result["description"] = line.split(":", 1)[1].strip()
            elif line.startswith("allowed-tools:"):
                result["allowed_tools"] = line.split(":", 1)[1].strip()
            elif line.startswith("model:"):
                result["model"] = line.split(":", 1)[1].strip()

        return result

    def _build_description(self, what: str, when: str) -> str:
        """Build Claude's description string from typed fields."""
        return f"WHAT: {what}\nWHEN: {when}"

    def create_skill(self, name: str, domain: str, content: str,
                     what: str, when: str,
                     subdomain: Optional[str] = None,
                     category: Optional[str] = None,
                     allowed_tools: Optional[str] = None,
                     model: Optional[str] = None) -> dict:
        """Create a new skill in global catalog with full resource structure.

        Args:
            what: What this skill does
            when: When to use it (trigger condition)
            category: One of 'understand', 'preflight', 'single_turn_process'
                - understand: pure context, no flight (read and absorb)
                - preflight: primes with concepts, body IS the HOW
                - single_turn_process: context + immediate action, no flight needed
            allowed_tools: Claude's allowed-tools field (comma-separated)
            model: Claude's model field
        """
        logger.info(f"Creating skill: {name} in {domain}::{subdomain or ''} (category={category})")

        skill_dir = self._skill_path(name)
        skill_dir.mkdir(parents=True, exist_ok=True)

        # Create resource directories
        (skill_dir / "scripts").mkdir(exist_ok=True)
        (skill_dir / "templates").mkdir(exist_ok=True)

        # Build description from typed fields
        description = self._build_description(what, when)

        # Write SKILL.md with ONLY Claude's fields in frontmatter
        allowed_line = f"\nallowed-tools: {allowed_tools}" if allowed_tools else ""
        model_line = f"\nmodel: {model}" if model else ""
        skill_md = f"""---
name: {name}
description: |
  {description.replace(chr(10), chr(10) + '  ')}{allowed_line}{model_line}
---

# {name.replace('-', ' ').replace('_', ' ').title()}

{content}
"""
        (skill_dir / "SKILL.md").write_text(skill_md)

        # Write empty reference.md
        (skill_dir / "reference.md").write_text(f"# {name} Reference\n\nAdd extended documentation here.\n")

        # Write GNOSYS extensions to _metadata.json
        metadata = {
            "domain": domain,
            "subdomain": subdomain,
            "category": category,
            "what": what,
            "when": when
        }
        self._metadata_path(name).write_text(json.dumps(metadata, indent=2))

        # Index in ChromaDB
        self._index_skill(name, domain, subdomain, description, content, category)

        skill = Skill(name=name, domain=domain, subdomain=subdomain,
                      content=content, description=description,
                      what=what, when=when,
                      category=category, allowed_tools=allowed_tools, model=model)

        return {
            "skill": skill,
            "path": str(skill_dir),
            "structure": {
                "SKILL.md": "main content (body is HOW for preflight)",
                "scripts/": "add executable scripts here",
                "templates/": "add reusable templates here",
                "reference.md": "add extended documentation here"
            }
        }

    def _index_skill(self, name: str, domain: str, subdomain: Optional[str],
                     description: str, content: str, category: Optional[str] = None):
        """Add skill to RAG index."""
        doc_id = f"skill:{name}"
        search_text = f"{domain} {subdomain or ''} {name} {description} {category or ''} {content}"
        self.collection.upsert(
            ids=[doc_id],
            documents=[search_text],
            metadatas=[{"name": name, "domain": domain, "subdomain": subdomain or "",
                        "type": "skill", "category": category or ""}]
        )

    def _scan_resources(self, skill_dir: Path) -> dict:
        """Scan skill directory for resources."""
        resources = {
            "scripts": [],
            "templates": [],
            "reference": None
        }

        scripts_dir = skill_dir / "scripts"
        if scripts_dir.exists():
            resources["scripts"] = [f.name for f in scripts_dir.iterdir() if f.is_file()]

        templates_dir = skill_dir / "templates"
        if templates_dir.exists():
            resources["templates"] = [f.name for f in templates_dir.iterdir() if f.is_file()]

        reference_path = skill_dir / "reference.md"
        if reference_path.exists():
            size = reference_path.stat().st_size
            resources["reference"] = f"{size} bytes" if size > 50 else "empty"

        return resources

    def get_skill(self, name: str) -> Optional[dict]:
        """Get a skill from global catalog with resource info."""
        skill_md_path = self._skill_md_path(name)
        if not skill_md_path.exists():
            return None

        content = skill_md_path.read_text()
        parsed = self._parse_skill_md(content)

        # Defaults
        domain, subdomain, category = "unknown", None, None
        what, when = "", ""

        # Load GNOSYS extensions from _metadata.json
        if self._metadata_path(name).exists():
            meta = json.loads(self._metadata_path(name).read_text())
            domain = meta.get("domain", "unknown")
            subdomain = meta.get("subdomain")
            category = meta.get("category")
            what = meta.get("what", "")
            when = meta.get("when", "")

        skill = Skill(
            name=parsed["name"] or name,
            description=parsed["description"],
            allowed_tools=parsed.get("allowed_tools"),
            model=parsed.get("model"),
            domain=domain,
            subdomain=subdomain,
            category=category,
            what=what,
            when=when,
            content=parsed["body"]
        )

        skill_dir = self._skill_path(name)
        resources = self._scan_resources(skill_dir)

        return {
            "skill": skill,
            "path": str(skill_dir),
            "resources": resources
        }

    def list_skills(self) -> list[dict]:
        """List all skills in global catalog."""
        skills = []
        for skill_dir in self.skills_dir.iterdir():
            if skill_dir.is_dir() and not skill_dir.name.startswith("_"):
                result = self.get_skill(skill_dir.name)
                if result:
                    skill = result["skill"]
                    skills.append({
                        "name": skill.name,
                        "domain": skill.domain,
                        "subdomain": skill.subdomain,
                        "description": skill.description,
                        "category": skill.category
                    })
        return skills

    def list_domains(self) -> list[str]:
        """List all available domains."""
        skills = self.list_skills()
        skillsets = self.list_skillsets()
        domains = set(s["domain"] for s in skills)
        domains.update(ss["domain"] for ss in skillsets)
        return sorted(domains)

    def list_by_domain(self, domain: str) -> dict:
        """List all skills and skillsets in a domain."""
        skills = [s for s in self.list_skills() if s["domain"] == domain]
        skillsets = [ss for ss in self.list_skillsets() if ss["domain"] == domain]
        return {"domain": domain, "skills": skills, "skillsets": skillsets}

    # === Claude skills directory mirroring ===

    def _mirror_to_claude(self, name: str):
        """Copy skill to Claude's native skills dir for hot-reload pickup."""
        src = self._skill_path(name)
        dst = CLAUDE_SKILLS_DIR / name
        if src.exists():
            CLAUDE_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            logger.info(f"Mirrored skill to Claude: {dst}")

    def _remove_from_claude(self, name: str):
        """Remove skill from Claude's native skills dir."""
        dst = CLAUDE_SKILLS_DIR / name
        if dst.exists():
            shutil.rmtree(dst)
            logger.info(f"Removed skill from Claude: {dst}")

    # === Defaults and sync ===

    def _load_defaults(self) -> list[str]:
        """Load default skill names from config."""
        if not self.defaults_file.exists():
            return []
        try:
            data = json.loads(self.defaults_file.read_text())
            return data.get("defaults", [])
        except (json.JSONDecodeError, KeyError):
            return []

    def _quarantine_skill(self, name: str, source: str, reason: str = "duplicate"):
        """Move skill to quarantine with metadata."""
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine_path = self.quarantine_dir / f"{name}_{timestamp}"

        # Determine source path
        if source == "claude":
            src = CLAUDE_SKILLS_DIR / name
        else:
            src = self.skills_dir / name

        if not src.exists():
            logger.warning(f"Cannot quarantine {name}: source {src} does not exist")
            return

        # Copy to quarantine
        shutil.copytree(src, quarantine_path)

        # Write metadata
        metadata = {
            "source": source,
            "original_name": name,
            "reason": reason,
            "timestamp": timestamp,
            "original_path": str(src)
        }
        (quarantine_path / "_quarantine_metadata.json").write_text(json.dumps(metadata, indent=2))
        logger.info(f"Quarantined skill: {name} from {source} -> {quarantine_path}")

    def list_quarantine(self) -> list[dict]:
        """List all quarantined skills."""
        if not self.quarantine_dir.exists():
            return []

        quarantined = []
        for p in self.quarantine_dir.iterdir():
            if p.is_dir():
                meta_path = p / "_quarantine_metadata.json"
                if meta_path.exists():
                    try:
                        metadata = json.loads(meta_path.read_text())
                        quarantined.append({
                            "path": str(p),
                            "metadata": metadata
                        })
                    except json.JSONDecodeError:
                        quarantined.append({
                            "path": str(p),
                            "metadata": {"error": "invalid metadata"}
                        })
        return quarantined

    def sync_on_startup(self):
        """Sync claude/skills to heaven, then clean claude to defaults only.

        1. Any skill in claude but not heaven -> copy to heaven (new from plugin)
        2. Any skill in claude AND heaven but not in defaults -> quarantine claude version
        3. Clean claude to defaults only
        4. Ensure defaults are present in claude
        """
        CLAUDE_SKILLS_DIR.mkdir(parents=True, exist_ok=True)

        # Get current state
        claude_skills = set(
            d.name for d in CLAUDE_SKILLS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        ) if CLAUDE_SKILLS_DIR.exists() else set()

        heaven_skills = set(
            d.name for d in self.skills_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        )

        defaults = set(self._load_defaults())

        logger.info(f"Sync on startup: claude={len(claude_skills)}, heaven={len(heaven_skills)}, defaults={len(defaults)}")

        # 1. Sync: claude -> heaven (new skills from plugins)
        for skill in claude_skills:
            if skill not in heaven_skills:
                # New skill from plugin - copy to heaven
                src = CLAUDE_SKILLS_DIR / skill
                dst = self.skills_dir / skill
                shutil.copytree(src, dst)
                logger.info(f"Synced new skill from claude to heaven: {skill}")
                # Index it
                self._try_index_skill(skill)
            elif skill not in defaults:
                # Duplicate! Quarantine the claude version
                self._quarantine_skill(skill, source="claude", reason="duplicate")

        # 2. Clean claude to defaults only
        for skill in claude_skills:
            if skill not in defaults:
                skill_path = CLAUDE_SKILLS_DIR / skill
                if skill_path.exists():
                    shutil.rmtree(skill_path)
                    logger.info(f"Removed non-default skill from claude: {skill}")

        # 3. Ensure defaults are present in claude
        for skill in defaults:
            if skill in heaven_skills:
                self._mirror_to_claude(skill)

    def _try_index_skill(self, name: str):
        """Try to index a skill that was synced from claude."""
        result = self.get_skill(name)
        if result:
            skill = result["skill"]
            self._index_skill(
                name, skill.domain, skill.subdomain,
                skill.description, skill.content, skill.category
            )

    # === Equipped state ===

    def equip(self, name: str) -> dict:
        """Equip a skill or skillset."""
        # Try as skillset first
        skillset = self.get_skillset(name)
        if skillset:
            return self.equip_skillset(name)

        # Try as skill
        result = self.get_skill(name)
        if result:
            skill = result["skill"]
            self.equipped[name] = skill
            self._mirror_to_claude(name)
            logger.info(f"Equipped skill: {name}")
            return {"equipped": name, "type": "skill", "domain": skill.domain}

        return {"error": f"'{name}' not found as skill or skillset"}

    def equip_skillset(self, name: str) -> dict:
        """Equip all skills in a skillset."""
        skillset = self.get_skillset(name)
        if not skillset:
            return {"error": f"Skillset '{name}' not found"}

        equipped_names = []
        for skill_name in skillset.skills:
            result = self.get_skill(skill_name)
            if result:
                self.equipped[skill_name] = result["skill"]
                self._mirror_to_claude(skill_name)
                equipped_names.append(skill_name)

        logger.info(f"Equipped skillset: {name} ({len(equipped_names)} skills)")
        return {
            "equipped": name,
            "type": "skillset",
            "domain": skillset.domain,
            "skills": equipped_names
        }

    def unequip(self, name: str) -> dict:
        """Unequip a skill."""
        if name in self.equipped:
            del self.equipped[name]
            self._remove_from_claude(name)
            return {"unequipped": name}
        return {"error": f"'{name}' not equipped"}

    def unequip_all(self) -> dict:
        """Clear all equipped skills."""
        count = len(self.equipped)
        for name in list(self.equipped.keys()):
            self._remove_from_claude(name)
        self.equipped.clear()
        return {"unequipped_count": count}

    def list_equipped(self) -> list[dict]:
        """List currently equipped skills."""
        return [
            {
                "name": s.name,
                "domain": s.domain,
                "subdomain": s.subdomain,
                "description": s.description
            }
            for s in self.equipped.values()
        ]

    def get_equipped_content(self) -> str:
        """Get full content of all equipped skills."""
        if not self.equipped:
            return "No skills equipped."

        lines = []
        for skill in self.equipped.values():
            lines.append(f"## {skill.name} ({skill.domain}::{skill.subdomain or ''})")
            lines.append(skill.content)
            lines.append("")
        return "\n".join(lines)

    # === RAG search ===

    def search_skills(self, query: str, n_results: int = 5,
                      category: Optional[str] = None) -> list[dict]:
        """Search skills and skillsets using RAG, optionally filtered by category."""
        where_filter = {"category": category} if category else None
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )

        matches = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                matches.append({
                    "name": metadata.get("name", doc_id.split(":", 1)[-1]),
                    "domain": metadata.get("domain", "unknown"),
                    "subdomain": metadata.get("subdomain", ""),
                    "type": metadata.get("type", "skill"),
                    "category": metadata.get("category", ""),
                    "score": 1 - distance
                })
        return matches

    def _parse_skilllog_prediction(self, prediction: str) -> dict:
        """Parse SkillLog prediction into components."""
        parts = prediction.split("::")
        return {
            "domain": parts[0] if parts else "",
            "subdomain": parts[1] if len(parts) > 1 else "",
            "specific": parts[2] if len(parts) > 2 else ""
        }

    def match_skilllog(self, prediction: str) -> dict:
        """Match a SkillLog prediction against catalog."""
        logger.info(f"Matching SkillLog: {prediction}")
        parsed = self._parse_skilllog_prediction(prediction)
        query = f"{parsed['domain']} {parsed['subdomain']} {parsed['specific']}".strip()

        matches = self.search_skills(query, n_results=5)
        exact_domain = [m for m in matches if m["domain"] == parsed["domain"]]

        return {
            "prediction": prediction,
            "parsed": parsed,
            "matches": matches,
            "exact_domain_matches": exact_domain,
            "available_domains": self.list_domains(),
            "has_match": len(matches) > 0 and matches[0]["score"] > 0.5
        }

    # === Skillset management ===

    def _load_skillsets(self) -> dict[str, Skillset]:
        if not self.skillsets_file.exists():
            return {}
        data = json.loads(self.skillsets_file.read_text())
        return {name: Skillset(**ss) for name, ss in data.items()}

    def _save_skillsets(self, skillsets: dict[str, Skillset]):
        data = {name: ss.model_dump() for name, ss in skillsets.items()}
        self.skillsets_file.write_text(json.dumps(data, indent=2))

    def create_skillset(self, name: str, domain: str, description: str,
                        skills: list[str], subdomain: Optional[str] = None) -> Skillset:
        """Create a skillset with domain and index with aggregated member domains."""
        logger.info(f"Creating skillset: {name} in {domain}::{subdomain or ''}")

        skillsets = self._load_skillsets()
        ss = Skillset(name=name, domain=domain, subdomain=subdomain,
                      description=description, skills=skills)
        skillsets[name] = ss
        self._save_skillsets(skillsets)

        # Index with aggregated domains from member skills
        self._index_skillset(ss)

        return ss

    def _index_skillset(self, ss: Skillset):
        """Index skillset with its own domain + all member skill domains."""
        # Start with skillset's own domain
        search_parts = [ss.domain, ss.subdomain or "", ss.name, ss.description]

        # Add all member skill domains
        for skill_name in ss.skills:
            result = self.get_skill(skill_name)
            if result:
                skill = result["skill"]
                search_parts.extend([skill.domain, skill.subdomain or "", skill.name])

        search_text = " ".join(search_parts)
        doc_id = f"skillset:{ss.name}"

        self.collection.upsert(
            ids=[doc_id],
            documents=[search_text],
            metadatas=[{"name": ss.name, "domain": ss.domain,
                        "subdomain": ss.subdomain or "", "type": "skillset"}]
        )

    def get_skillset(self, name: str) -> Optional[Skillset]:
        skillsets = self._load_skillsets()
        return skillsets.get(name)

    def list_skillsets(self) -> list[dict]:
        skillsets = self._load_skillsets()
        return [
            {
                "name": ss.name,
                "domain": ss.domain,
                "subdomain": ss.subdomain,
                "description": ss.description,
                "skill_count": len(ss.skills)
            }
            for ss in skillsets.values()
        ]

    def add_to_skillset(self, skillset_name: str, skill_name: str) -> dict:
        """Add a skill to a skillset and reindex."""
        skillsets = self._load_skillsets()
        if skillset_name not in skillsets:
            return {"error": f"Skillset '{skillset_name}' not found"}

        ss = skillsets[skillset_name]
        if skill_name not in ss.skills:
            ss.skills.append(skill_name)
            self._save_skillsets(skillsets)
            self._index_skillset(ss)  # Reindex with new skill

        return {"success": True, "skillset": skillset_name, "skills": ss.skills}

    # === Persona management ===

    def _load_personas(self) -> dict[str, Persona]:
        if not self.personas_file.exists():
            return {}
        data = json.loads(self.personas_file.read_text())
        return {name: Persona(**p) for name, p in data.items()}

    def _save_personas(self, personas: dict[str, Persona]):
        data = {name: p.model_dump() for name, p in personas.items()}
        self.personas_file.write_text(json.dumps(data, indent=2))

    def create_persona(self, name: str, domain: str, description: str, frame: str,
                       mcp_set: Optional[str] = None, skillset: Optional[str] = None,
                       carton_identity: Optional[str] = None,
                       subdomain: Optional[str] = None) -> Persona:
        """Create a persona with aspirational MCP set, skillset, and identity."""
        logger.info(f"Creating persona: {name} in {domain}::{subdomain or ''}")

        personas = self._load_personas()
        persona = Persona(
            name=name, domain=domain, subdomain=subdomain,
            description=description, frame=frame,
            mcp_set=mcp_set, skillset=skillset,
            carton_identity=carton_identity or name
        )
        personas[name] = persona
        self._save_personas(personas)

        # Index persona in RAG
        self._index_persona(persona)

        return persona

    def _index_persona(self, p: Persona):
        """Index persona for RAG search."""
        search_text = f"persona {p.domain} {p.subdomain or ''} {p.name} {p.description} {p.frame}"
        doc_id = f"persona:{p.name}"
        self.collection.upsert(
            ids=[doc_id],
            documents=[search_text],
            metadatas=[{"name": p.name, "domain": p.domain,
                        "subdomain": p.subdomain or "", "type": "persona"}]
        )

    def get_persona(self, name: str) -> Optional[Persona]:
        personas = self._load_personas()
        return personas.get(name)

    def list_personas(self) -> list[dict]:
        personas = self._load_personas()
        return [
            {
                "name": p.name,
                "domain": p.domain,
                "subdomain": p.subdomain,
                "description": p.description,
                "mcp_set": p.mcp_set,
                "skillset": p.skillset,
                "carton_identity": p.carton_identity
            }
            for p in personas.values()
        ]

    def _try_equip_skillset_for_persona(self, persona: Persona, report: dict):
        """Attempt to equip persona's skillset, update report with status."""
        if not persona.skillset:
            return
        skillset = self.get_skillset(persona.skillset)
        if skillset:
            equip_result = self.equip_skillset(persona.skillset)
            report["skillset"] = "equipped"
            report["equipped_skills"] = equip_result.get("skills", [])
        else:
            report["missing"].append({
                "type": "skillset",
                "name": persona.skillset,
                "suggestion": f"Create skillset '{persona.skillset}' with create_skillset()"
            })

    def _build_mcp_set_status(self, persona: Persona) -> Optional[dict]:
        """Build MCP set status for persona report."""
        if not persona.mcp_set:
            return None
        return {
            "name": persona.mcp_set,
            "status": "requires_strata",
            "action": f"Use strata: connect_set('{persona.mcp_set}')"
        }

    def equip_persona(self, name: str) -> dict:
        """Equip a persona - activate frame, attempt MCP set and skillset."""
        persona = self.get_persona(name)
        if not persona:
            return {"error": f"Persona '{name}' not found"}

        logger.info(f"Equipping persona: {name}")

        report = {
            "persona": name,
            "frame": "loaded",
            "frame_content": persona.frame,
            "mcp_set": None,
            "skillset": None,
            "carton_identity": persona.carton_identity,
            "missing": [],
            "equipped_skills": []
        }

        self._try_equip_skillset_for_persona(persona, report)
        report["mcp_set"] = self._build_mcp_set_status(persona)

        self.active_persona = persona
        return report

    def get_active_persona(self) -> Optional[dict]:
        """Get the currently active persona."""
        if not self.active_persona:
            return None
        return {
            "name": self.active_persona.name,
            "domain": self.active_persona.domain,
            "frame": self.active_persona.frame,
            "mcp_set": self.active_persona.mcp_set,
            "skillset": self.active_persona.skillset,
            "carton_identity": self.active_persona.carton_identity
        }

    def deactivate_persona(self) -> dict:
        """Deactivate current persona and unequip all skills."""
        if not self.active_persona:
            return {"status": "no active persona"}

        name = self.active_persona.name
        self.active_persona = None
        self.unequip_all()
        return {"deactivated": name, "skills_unequipped": True}

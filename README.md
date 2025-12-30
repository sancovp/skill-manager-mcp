# Skill Manager

Self-steering skill management MCP with RAG-based prediction matching.

Part of the [GNOSYS](https://github.com/sancovp/gnosys) compound intelligence ecosystem.

## Features

- Create/manage skills compatible with Claude Code's skill format
- Skillsets for grouping related skills
- RAG-based skill search using ChromaDB
- SkillLog prediction matching for self-steering

## Usage

```bash
pip install skill-manager-mcp
```

Add to your MCP config:
```json
{
  "skill-manager": {
    "command": "skill-manager"
  }
}
```

## Tools

- `list_skills` - List all available skills
- `get_skill` - Get full skill content
- `create_skill` - Create a new skill
- `search_skills` - Semantic search for skills
- `match_skilllog` - Match SkillLog predictions
- `list_skillsets` - List skill groups
- `create_skillset` - Create skill group
- `activate_skillset` - Load all skills in a group
- `add_to_skillset` - Add skill to group

# Claude Code Skill YAML Format (Official)

## SKILL.md Frontmatter Schema

| Field | Required | Type | Constraints |
|-------|----------|------|-------------|
| `name` | Yes | string | max 64 chars, lowercase/numbers/hyphens only |
| `description` | Yes | string | max 1024 chars |
| `allowed-tools` | No | string | comma-separated tool names |
| `model` | No | string | e.g. `claude-sonnet-4-20250514` |

## Example

```yaml
---
name: my-skill-name
description: Claude uses this to decide when to apply the Skill
allowed-tools: Read, Write, Bash
model: claude-sonnet-4-20250514
---
```

## Notes

- `description` is what Claude uses to decide when to trigger the skill
- That's it. Four fields. Two required, two optional.

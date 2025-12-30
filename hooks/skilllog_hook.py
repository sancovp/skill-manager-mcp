#!/usr/bin/env python3
"""SkillLog hook - parses SkillLog from assistant responses and injects skill matches."""

import json
import logging
import re
import sys
from skill_manager.core import SkillManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_skilllog(text: str) -> str | None:
    """Extract SkillLog prediction from text.

    Format: 🎯 domain::subdomain::specific 🎯
    """
    pattern = r'🎯\s*([^🎯]+)\s*🎯'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None


def get_last_assistant_text(transcript: list) -> str:
    """Extract text from most recent assistant message."""
    for msg in reversed(transcript):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        for block in content:
            if block.get("type") == "text":
                return block.get("text", "")
    return ""


def format_match_output(result: dict) -> str:
    """Format skill match results for injection."""
    lines = [f"🎯 SkillLog detected: {result['prediction']}"]

    if result['has_match']:
        lines.append("Matching skills available:")
        for m in result['matches'][:3]:
            domain_path = f"{m['domain']}::{m['subdomain']}" if m['subdomain'] else m['domain']
            lines.append(f"  - {m['name']} ({domain_path}) [score: {m['score']:.2f}]")
        lines.append("Use get_skill(name) to load, or activate_skillset() for groups.")
    else:
        lines.append("No strong matches found.")
        if result['available_domains']:
            lines.append(f"Available domains: {', '.join(result['available_domains'])}")
        lines.append("Consider creating a skill for this domain.")

    return "\n".join(lines)


def main():
    """Hook entry point."""
    input_data = json.load(sys.stdin)
    transcript = input_data.get("transcript", [])

    assistant_text = get_last_assistant_text(transcript)
    if not assistant_text:
        logger.debug("No assistant message found")
        sys.exit(0)

    prediction = parse_skilllog(assistant_text)
    if not prediction:
        logger.debug("No SkillLog found in response")
        sys.exit(0)

    logger.info(f"SkillLog detected: {prediction}")
    manager = SkillManager()
    result = manager.match_skilllog(prediction)

    output = format_match_output(result)
    sys.stdout.write(json.dumps({"result": output}))


if __name__ == "__main__":
    main()

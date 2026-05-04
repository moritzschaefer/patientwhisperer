"""
Bootstrap the OpenScientist SQLite database and seed domain skills.

Run once before dispatching patient agents:
    python bootstrap.py [--db openscientist.db]

Creates all tables and inserts CAR T domain + workflow skills.
"""
import asyncio
import hashlib
import os
import sys
from pathlib import Path
from uuid import uuid4

# Wire up OpenScientist on sys.path
VENDOR_DIR = Path(__file__).resolve().parent.parent.parent / "vendor" / "openscientist" / "src"
sys.path.insert(0, str(VENDOR_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

SKILLS_DIR = VENDOR_DIR.parent / "skills"


def _set_database_url(db_path: str) -> None:
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"


async def create_tables() -> None:
    """Create all database tables from SQLAlchemy models."""
    from openscientist.database.base import Base
    from openscientist.database.engine import get_engine

    # Import all models so Base.metadata is populated
    import openscientist.database.models  # noqa: F401

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print(f"Created tables in {engine.url}")


async def seed_skills() -> None:
    """Insert domain and workflow skills from markdown files."""
    from openscientist.database.models.skill import Skill
    from openscientist.database.session import AsyncSessionLocal

    skills_to_seed: list[dict] = []
    for category_dir in SKILLS_DIR.iterdir():
        if not category_dir.is_dir():
            continue
        category = category_dir.name  # "domain" or "workflow"
        for skill_dir in category_dir.iterdir():
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            content = skill_file.read_text(encoding="utf-8")
            # Parse YAML frontmatter
            name = skill_dir.name
            description = ""
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    for line in parts[1].strip().splitlines():
                        if line.startswith("name:"):
                            name = line.split(":", 1)[1].strip()
                        elif line.startswith("description:"):
                            description = line.split(":", 1)[1].strip()
                    content = parts[2].strip()

            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            skills_to_seed.append({
                "name": name,
                "slug": skill_dir.name,
                "category": category,
                "description": description,
                "content": content,
                "content_hash": content_hash,
            })

    async with AsyncSessionLocal(thread_safe=True) as session:
        for skill_data in skills_to_seed:
            skill = Skill(
                id=uuid4(),
                name=skill_data["name"],
                slug=skill_data["slug"],
                category=skill_data["category"],
                description=skill_data["description"],
                content=skill_data["content"],
                content_hash=skill_data["content_hash"],
                tags=[skill_data["category"]],
                is_enabled=True,
            )
            session.add(skill)
        await session.commit()
    print(f"Seeded {len(skills_to_seed)} skills")


async def main(db_path: str) -> None:
    _set_database_url(db_path)
    await create_tables()
    await seed_skills()
    print("Bootstrap complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bootstrap OpenScientist DB")
    parser.add_argument("--db", default="openscientist.db", help="SQLite database path")
    args = parser.parse_args()

    asyncio.run(main(args.db))

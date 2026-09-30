"""Neo4j access - the knowledge-graph half of retrieval.

We drive Neo4j through the raw driver rather than a LlamaIndex graph store so
that entity canonicalisation, the uniqueness constraint and the chunk_id backlink
into Qdrant stay explicit and debuggable.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from neo4j import GraphDatabase, Driver

from src import config

log = logging.getLogger(__name__)

_driver: Driver | None = None


def driver() -> Driver:
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
            max_connection_lifetime=600,
        )
    return _driver


def close() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def verify() -> None:
    """Fail fast with remediation text if Neo4j or APOC is unavailable."""
    try:
        with driver().session() as session:
            session.run("RETURN 1").consume()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Neo4j at {config.NEO4J_URI}.\n"
            "  Start it:  docker compose up -d neo4j\n"
            f"  Detail:    {exc}"
        ) from exc

    try:
        with driver().session() as session:
            session.run("RETURN apoc.version()").consume()
    except Exception as exc:
        raise RuntimeError(
            "Neo4j is up but APOC is not loaded. Triple insertion needs "
            "apoc.merge.relationship because Cypher cannot parameterise "
            "relationship types.\n"
            "  Check NEO4J_PLUGINS=[\"apoc\"] in docker-compose.yml, then:\n"
            "  docker compose down && docker compose up -d\n"
            f"  Detail: {exc}"
        ) from exc


def ensure_schema() -> None:
    """Idempotent constraint creation.

    The uniqueness constraint on `key` is what makes MERGE safe to re-run: without
    it, a second --build-index pass duplicates every entity.
    """
    with driver().session() as session:
        session.run(
            "CREATE CONSTRAINT entity_key IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.key IS UNIQUE"
        ).consume()


_UPSERT_CYPHER = """
UNWIND $triples AS t
MERGE (s:Entity {key: t.skey})
  ON CREATE SET s.name = t.sname
MERGE (o:Entity {key: t.tkey})
  ON CREATE SET o.name = t.tname
SET s.chunk_ids = apoc.coll.toSet(coalesce(s.chunk_ids, []) + [$chunk_id]),
    o.chunk_ids = apoc.coll.toSet(coalesce(o.chunk_ids, []) + [$chunk_id])
WITH s, o, t
CALL apoc.merge.relationship(s, t.rel, {}, {}, o, {}) YIELD rel
SET rel.chunk_ids = apoc.coll.toSet(coalesce(rel.chunk_ids, []) + [$chunk_id])
RETURN count(rel) AS written
"""


def upsert_triples(triples: list[dict[str, str]], chunk_id: str) -> int:
    """Write a chunk's triples, linking every touched node back to its vector.

    `triples` entries carry skey/sname/rel/tkey/tname, already normalised by
    text_utils. Returns the number of relationships merged.
    """
    if not triples:
        return 0
    with driver().session() as session:
        record = session.run(
            _UPSERT_CYPHER, triples=triples, chunk_id=chunk_id
        ).single()
        return int(record["written"]) if record else 0


def find_paths(keys: Iterable[str]) -> list[str]:
    """Return human-readable graph paths radiating from the query's entities.

    The hop count is interpolated rather than parameterised because Cypher does
    not accept a parameter inside a variable-length pattern. It is clamped to a
    small integer range so this can never become injectable.
    """
    key_list = [k for k in keys if k]
    if not key_list:
        return []

    hops = max(1, min(int(config.GRAPH_MAX_HOPS), 3))
    cypher = (
        f"MATCH p = (n:Entity)-[*1..{hops}]-(m:Entity) "
        "WHERE n.key IN $keys "
        "RETURN p LIMIT $limit"
    )

    rendered: list[str] = []
    seen: set[str] = set()
    with driver().session() as session:
        for record in session.run(
            cypher, keys=key_list, limit=int(config.GRAPH_PATH_LIMIT)
        ):
            line = _render_path(record["p"])
            if line and line not in seen:
                seen.add(line)
                rendered.append(line)
    return rendered


def _render_path(path: Any) -> str:
    """Serialise a neo4j Path as 'A -[REL]-> B -[REL]-> C', with edge direction."""
    try:
        nodes = list(path.nodes)
        rels = list(path.relationships)
    except AttributeError:
        return ""

    parts = [nodes[0].get("name") or nodes[0].get("key") or "?"]
    for i, rel in enumerate(rels):
        nxt = nodes[i + 1]
        label = nxt.get("name") or nxt.get("key") or "?"
        # A path can traverse a relationship against its stored direction.
        forward = rel.start_node.element_id == nodes[i].element_id
        arrow = f"-[{rel.type}]->" if forward else f"<-[{rel.type}]-"
        parts.append(f"{arrow} {label}")
    return " ".join(parts)


def chunk_ids_for_keys(keys: Iterable[str], limit: int = 20) -> list[str]:
    """Vector chunk IDs attached to the matched entities.

    This is the graph -> vector direction of the link: entities found by name can
    pull back the exact passages they were extracted from.
    """
    key_list = [k for k in keys if k]
    if not key_list:
        return []
    with driver().session() as session:
        result = session.run(
            "MATCH (n:Entity) WHERE n.key IN $keys "
            "UNWIND coalesce(n.chunk_ids, []) AS cid "
            "RETURN DISTINCT cid LIMIT $limit",
            keys=key_list,
            limit=limit,
        )
        return [r["cid"] for r in result]


def stats() -> dict[str, int]:
    with driver().session() as session:
        nodes = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
        rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        orphan = session.run(
            "MATCH (n:Entity) WHERE n.chunk_ids IS NULL RETURN count(n) AS c"
        ).single()["c"]
    return {"entities": nodes, "relationships": rels, "unlinked_entities": orphan}


def wipe() -> None:
    """Delete every entity and relationship. Used by --reset."""
    with driver().session() as session:
        session.run("MATCH (n:Entity) DETACH DELETE n").consume()

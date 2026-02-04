# -*- coding: utf-8 -*-
"""
Database module for RAG_BE_02.

This module provides PostgreSQL database access for the RAG service.
It handles:
1. File hash caching (for rebuild detection)
2. LLM summary caching (to avoid re-generating expensive summaries)

ARCHITECTURE:
- `papers` table (owned by Backend) is the SINGLE SOURCE OF TRUTH for:
  - Paper metadata (title, abstract, authors)
  - Processing status
  - File URL

- RAG service owns these minimal tables:
  - `rag_paper_cache`: File hash for rebuild detection
  - `paper_content_summaries`: Cached LLM summaries

- RAG reads metadata from `papers` table when needed (no duplication)
"""

import os
import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

DATABASE_URL = os.getenv("RAG_DATABASE_URL", os.getenv("DATABASE_URL", ""))


def parse_database_url(url: str) -> Dict[str, Any]:
    """Parse PostgreSQL connection URL into connection parameters."""
    if not url:
        raise ValueError("DATABASE_URL environment variable is not set")

    # Handle postgres:// or postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    from urllib.parse import urlparse
    parsed = urlparse(url)

    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "database": parsed.path.lstrip("/") if parsed.path else "rag_db",
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
    }


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PaperMetadata:
    """
    Paper metadata read from the `papers` table (owned by Backend).
    RAG service reads this for building prompts.
    """
    rag_paper_id: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    authors: Optional[str] = None  # JSON array string
    status: str = "PENDING"
    node_count: int = 0
    table_count: int = 0
    image_count: int = 0


@dataclass
class ContentSummaryRecord:
    """
    Cached LLM-generated summary for a table or image.
    Stored in `paper_content_summaries` table.
    """
    id: Optional[int] = None
    rag_paper_id: str = ""
    content_type: str = ""  # 'table' or 'image'
    content_index: int = 0
    content_hash: str = ""
    summary_text: str = ""
    created_at: Optional[datetime] = None


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

class RAGDatabase:
    """
    PostgreSQL database interface for RAG_BE_02.

    RAG service owns:
    - rag_paper_cache: File hash for rebuild detection
    - paper_content_summaries: Cached LLM summaries

    RAG service reads from (Backend-owned):
    - papers: Metadata, status, file URL
    """

    def __init__(self, database_url: Optional[str] = None):
        """Initialize database connection."""
        self._conn_params = parse_database_url(database_url or DATABASE_URL)
        self._init_schema()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with automatic cleanup."""
        conn = psycopg2.connect(**self._conn_params)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self):
        """Initialize RAG-owned tables if not exists."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Minimal cache table for RAG processing
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS rag_paper_cache (
                        rag_paper_id VARCHAR(100) PRIMARY KEY,
                        file_content_hash VARCHAR(64),
                        last_processed_at TIMESTAMPTZ,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # Table for cached summaries
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS paper_content_summaries (
                        id SERIAL PRIMARY KEY,
                        rag_paper_id VARCHAR(100) NOT NULL,
                        content_type VARCHAR(20) NOT NULL,
                        content_index INTEGER NOT NULL,
                        content_hash VARCHAR(64) NOT NULL,
                        summary_text TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT paper_content_summaries_unique
                            UNIQUE (rag_paper_id, content_type, content_index),
                        CONSTRAINT paper_content_summaries_content_type_check
                            CHECK (content_type IN ('table', 'image'))
                    );

                    CREATE INDEX IF NOT EXISTS idx_summaries_paper
                        ON paper_content_summaries(rag_paper_id);
                    CREATE INDEX IF NOT EXISTS idx_summaries_type
                        ON paper_content_summaries(rag_paper_id, content_type);
                """)

    # =========================================================================
    # READ FROM PAPERS TABLE (Backend-owned, single source of truth)
    # =========================================================================

    def get_paper_metadata(self, rag_paper_id: str) -> Optional[PaperMetadata]:
        """
        Read paper metadata from the `papers` table (owned by Backend).
        Used for building prompts with paper context.
        """
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT rag_file_id, title, abstract, authors, status,
                           node_count, table_count, image_count
                    FROM papers
                    WHERE rag_file_id = %s
                """, (rag_paper_id,))
                row = cur.fetchone()

                if not row:
                    return None

                return PaperMetadata(
                    rag_paper_id=row['rag_file_id'],
                    title=row['title'],
                    abstract=row['abstract'],
                    authors=row['authors'],
                    status=row['status'],
                    node_count=row['node_count'] or 0,
                    table_count=row['table_count'] or 0,
                    image_count=row['image_count'] or 0,
                )

    def get_paper_status(self, rag_paper_id: str) -> Optional[str]:
        """
        Get paper status from the `papers` table.
        Falls back to checking `rag_paper_cache` for guest mode (no papers record).

        Returns: 'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED' or None
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # First check papers table (for authenticated users)
                cur.execute(
                    "SELECT status FROM papers WHERE rag_file_id = %s",
                    (rag_paper_id,)
                )
                result = cur.fetchone()
                if result:
                    return result[0]

                # Fallback: check rag_paper_cache for guest mode
                # If file hash exists, it means ingest completed successfully
                cur.execute(
                    "SELECT file_content_hash FROM rag_paper_cache WHERE rag_paper_id = %s",
                    (rag_paper_id,)
                )
                cache_result = cur.fetchone()
                if cache_result and cache_result[0]:
                    return 'COMPLETED'  # Guest file was ingested successfully

                return None

    # =========================================================================
    # FILE HASH OPERATIONS (rag_paper_cache table)
    # =========================================================================

    def get_file_hash(self, rag_paper_id: str) -> Optional[str]:
        """Get stored file hash for rebuild detection."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT file_content_hash FROM rag_paper_cache WHERE rag_paper_id = %s",
                    (rag_paper_id,)
                )
                result = cur.fetchone()
                return result[0] if result else None

    def save_file_hash(self, rag_paper_id: str, file_hash: str) -> None:
        """Save file hash after successful processing."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO rag_paper_cache (rag_paper_id, file_content_hash, last_processed_at)
                    VALUES (%s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (rag_paper_id) DO UPDATE SET
                        file_content_hash = EXCLUDED.file_content_hash,
                        last_processed_at = CURRENT_TIMESTAMP
                """, (rag_paper_id, file_hash))

    def needs_rebuild(self, rag_paper_id: str, current_hash: str) -> bool:
        """Check if vector store needs rebuild (file changed)."""
        stored_hash = self.get_file_hash(rag_paper_id)
        return stored_hash is None or stored_hash != current_hash

    # =========================================================================
    # SUMMARY CACHE OPERATIONS (paper_content_summaries table)
    # =========================================================================

    def get_cached_summaries(
        self,
        rag_paper_id: str,
        content_type: str,
    ) -> Optional[List[str]]:
        """
        Get cached summaries for tables or images.
        Returns list of summary strings ordered by content_index, or None if not cached.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT content_index, summary_text
                    FROM paper_content_summaries
                    WHERE rag_paper_id = %s AND content_type = %s
                    ORDER BY content_index
                """, (rag_paper_id, content_type))

                rows = cur.fetchall()
                if not rows:
                    return None

                # Build list maintaining order by content_index
                max_index = max(row[0] for row in rows)
                summaries = [""] * (max_index + 1)
                for content_index, summary_text in rows:
                    summaries[content_index] = summary_text

                return summaries

    def save_cached_summaries(
        self,
        rag_paper_id: str,
        content_type: str,
        summaries: List[str],
        content_hashes: Optional[List[str]] = None,
    ) -> None:
        """Save summaries to cache."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Delete existing summaries for this paper and type
                cur.execute("""
                    DELETE FROM paper_content_summaries
                    WHERE rag_paper_id = %s AND content_type = %s
                """, (rag_paper_id, content_type))

                # Insert new summaries
                for idx, summary in enumerate(summaries):
                    if not summary:
                        continue

                    content_hash = ""
                    if content_hashes and idx < len(content_hashes):
                        content_hash = content_hashes[idx]

                    cur.execute("""
                        INSERT INTO paper_content_summaries
                        (rag_paper_id, content_type, content_index, content_hash, summary_text)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (rag_paper_id, content_type, idx, content_hash, summary))

    def clear_paper_cache(self, rag_paper_id: str) -> None:
        """Clear all cached data for a paper (used when re-ingesting)."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM paper_content_summaries WHERE rag_paper_id = %s",
                    (rag_paper_id,)
                )
                cur.execute(
                    "DELETE FROM rag_paper_cache WHERE rag_paper_id = %s",
                    (rag_paper_id,)
                )

    def delete_summaries(self, rag_paper_id: str) -> int:
        """Delete all content summaries for a paper. Returns count deleted."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM paper_content_summaries WHERE rag_paper_id = %s",
                    (rag_paper_id,)
                )
                return cur.rowcount

    def delete_paper_cache(self, rag_paper_id: str) -> bool:
        """Delete paper cache entry. Returns True if deleted."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM rag_paper_cache WHERE rag_paper_id = %s",
                    (rag_paper_id,)
                )
                return cur.rowcount > 0

    def get_orphaned_guest_files(self, max_age_hours: int = 24) -> List[str]:
        """
        Find guest files that are older than max_age_hours.
        Guest files exist in rag_paper_cache but NOT in papers table.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT r.rag_paper_id
                    FROM rag_paper_cache r
                    LEFT JOIN papers p ON r.rag_paper_id = p.rag_file_id
                    WHERE p.id IS NULL
                    AND r.created_at < NOW() - INTERVAL '%s hours'
                """, (max_age_hours,))
                return [row[0] for row in cur.fetchall()]

    # =========================================================================
    # UTILITY OPERATIONS
    # =========================================================================

    def paper_exists_in_cache(self, rag_paper_id: str) -> bool:
        """Check if paper exists in RAG cache."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM rag_paper_cache WHERE rag_paper_id = %s",
                    (rag_paper_id,)
                )
                return cur.fetchone() is not None


# =============================================================================
# GLOBAL DATABASE INSTANCE
# =============================================================================

_db_instance: Optional[RAGDatabase] = None


def get_database() -> RAGDatabase:
    """Get the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = RAGDatabase()
    return _db_instance


# =============================================================================
# CONVENIENCE FUNCTIONS (for backward compatibility with api.py)
# =============================================================================

def get_status(rag_paper_id: str) -> Optional[str]:
    """
    Get paper status - reads from Backend's `papers` table.
    Returns lowercase status for backward compatibility.
    """
    db = get_database()
    status = db.get_paper_status(rag_paper_id)

    if status is None:
        return None

    # Convert Backend status to RAG format
    status_map = {
        'PENDING': 'pending',
        'PROCESSING': 'processing',
        'COMPLETED': 'completed',
        'FAILED': 'failed',
    }
    return status_map.get(status, status.lower())


def load_metadata(rag_paper_id: str) -> Optional[Any]:
    """
    Load paper metadata from Backend's `papers` table.
    Falls back to minimal metadata for guest mode (no papers record).
    Returns object compatible with IngestionResult interface.
    """
    db = get_database()
    record = db.get_paper_metadata(rag_paper_id)

    @dataclass
    class MetadataResult:
        paper_id: str
        title: str
        abstract: str
        node_count: int
        table_count: int
        image_count: int

    if record is None:
        # Fallback for guest mode: check if file exists in rag_paper_cache
        file_hash = db.get_file_hash(rag_paper_id)
        if file_hash:
            # File was ingested (guest mode), return minimal metadata
            return MetadataResult(
                paper_id=rag_paper_id,
                title="Guest Document",  # Default title for guest
                abstract="",
                node_count=0,
                table_count=0,
                image_count=0,
            )
        return None

    return MetadataResult(
        paper_id=record.rag_paper_id,
        title=record.title or "",
        abstract=record.abstract or "",
        node_count=record.node_count,
        table_count=record.table_count,
        image_count=record.image_count,
    )

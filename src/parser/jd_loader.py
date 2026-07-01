"""Load and validate job-description records from JSON files.

This module provides the input-boundary utilities for job-description data.
It performs structural validation only and leaves all matching, scoring, and
reasoning behavior to the downstream pipeline modules.
"""

from __future__ import annotations

import json
import logging
import re
from zipfile import BadZipFile, ZipFile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from src.parser.schema_loader import get_job_required_fields


logger = logging.getLogger(__name__)

REQUIRED_FIELDS = get_job_required_fields()
DOCX_WORD_NAMESPACE = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
}
REQUIRED_SKILL_TERMS = (
    "embeddings",
    "retrieval",
    "ranking",
    "llms",
    "fine-tuning",
    "python",
    "vector databases",
    "hybrid search",
    "pinecone",
    "weaviate",
    "qdrant",
    "milvus",
    "opensearch",
    "elasticsearch",
    "faiss",
    "ndcg",
    "mrr",
    "map",
    "a/b testing",
    "learning-to-rank",
    "xgboost",
    "nlp",
    "information retrieval",
)


class JobDescriptionSchemaError(ValueError):
    """Raised when a job-description file has an invalid schema."""


def _load_json(file_path: str | Path) -> tuple[Path, Any]:
    """Load raw JSON from a file.

    Args:
        file_path: Path to a UTF-8 encoded JSON file.

    Returns:
        The normalized path and decoded JSON value.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist or is not a file.
        json.JSONDecodeError: If the file does not contain valid JSON.
        OSError: If the file cannot be read.
    """
    path = Path(file_path)
    if not path.is_file():
        message = f"Job-description file does not exist or is not a file: {path}"
        logger.error(message)
        raise FileNotFoundError(message)

    try:
        with path.open("r", encoding="utf-8") as job_file:
            return path, json.load(job_file)
    except json.JSONDecodeError as exc:
        logger.error(
            "Invalid JSON in job-description file %s at line %d, column %d: %s",
            path,
            exc.lineno,
            exc.colno,
            exc.msg,
        )
        raise json.JSONDecodeError(
            f"Invalid JSON in job-description file '{path}': {exc.msg}",
            exc.doc,
            exc.pos,
        ) from exc
    except OSError:
        logger.exception("Unable to read job-description file: %s", path)
        raise


def _load_docx_text(file_path: str | Path) -> tuple[Path, list[str]]:
    """Load text paragraphs from a Word ``.docx`` file.

    Args:
        file_path: Path to a Word document.

    Returns:
        The normalized path and non-empty text paragraphs.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist or is not a file.
        JobDescriptionSchemaError: If the document cannot be parsed.
        OSError: If the file cannot be read.
    """
    path = Path(file_path)
    if not path.is_file():
        message = f"Job-description file does not exist or is not a file: {path}"
        logger.error(message)
        raise FileNotFoundError(message)

    try:
        with ZipFile(path) as docx_file:
            document_xml = docx_file.read("word/document.xml")
    except KeyError as exc:
        _raise_schema_error(path, "the DOCX file does not contain document text")
        raise AssertionError("unreachable") from exc
    except BadZipFile as exc:
        message = f"Invalid DOCX file '{path}': unable to open document archive"
        logger.error(message)
        raise JobDescriptionSchemaError(message) from exc
    except OSError:
        logger.exception("Unable to read job-description file: %s", path)
        raise

    try:
        root = ElementTree.fromstring(document_xml)
    except ElementTree.ParseError as exc:
        message = f"Invalid DOCX XML in '{path}': {exc}"
        logger.error(message)
        raise JobDescriptionSchemaError(message) from exc

    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", DOCX_WORD_NAMESPACE):
        text = "".join(
            node.text or ""
            for node in paragraph.findall(".//w:t", DOCX_WORD_NAMESPACE)
        ).strip()
        if text:
            paragraphs.append(text)

    if not paragraphs:
        _raise_schema_error(path, "the DOCX file contains no readable text")

    return path, paragraphs


def _job_from_docx(path: Path, paragraphs: list[str]) -> dict[str, Any]:
    """Build a job-description dictionary from DOCX text.

    Args:
        path: Source DOCX path.
        paragraphs: Non-empty document paragraphs.

    Returns:
        A job-description dictionary matching the project schema.
    """
    first_line = paragraphs[0]
    title = re.sub(r"^job description:\s*", "", first_line, flags=re.I).strip()
    if not title:
        title = path.stem.replace("_", " ").replace("-", " ").title()

    description = "\n".join(paragraphs)
    normalized_description = description.casefold()
    required_skills = [
        skill
        for skill in REQUIRED_SKILL_TERMS
        if skill.casefold() in normalized_description
    ]

    return {
        "job_id": path.stem,
        "title": title,
        "description": description,
        "required_skills": required_skills,
    }


def load_job_description(file_path: str | Path) -> dict[str, Any]:
    """Load one valid job description from a JSON file.

    The file must contain a JSON object with every required top-level field.
    Additional fields are preserved.

    Args:
        file_path: Path to a UTF-8 encoded JSON job-description file.

    Returns:
        The validated job-description dictionary.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist or is not a file.
        json.JSONDecodeError: If the file does not contain valid JSON.
        JobDescriptionSchemaError: If the JSON root is not an object or the
            job description is missing a required field.
        OSError: If the file cannot be read.
    """
    path = Path(file_path)
    if path.suffix.casefold() == ".docx":
        path, paragraphs = _load_docx_text(path)
        job_description = _job_from_docx(path, paragraphs)
    else:
        path, job_description = _load_json(path)

    if not isinstance(job_description, dict):
        _raise_schema_error(path, "the JSON root must be an object")

    if not validate_job_description(job_description):
        _raise_schema_error(
            path,
            "the job description is missing one or more required fields",
        )

    logger.info("Loaded job description %r from %s", job_description["job_id"], path)
    return job_description


def load_job_descriptions(file_path: str | Path) -> list[dict[str, Any]]:
    """Load valid job descriptions from a JSON file.

    The file must contain a JSON array. Entries that are not objects or do not
    contain every required top-level field are excluded from the result.

    Args:
        file_path: Path to a UTF-8 encoded JSON job-description file.

    Returns:
        Valid job descriptions in their original order.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist or is not a file.
        json.JSONDecodeError: If the file does not contain valid JSON.
        JobDescriptionSchemaError: If the JSON root is not an array.
        OSError: If the file cannot be read.
    """
    path, job_descriptions = _load_json(file_path)
    if not isinstance(job_descriptions, list):
        _raise_schema_error(path, "the JSON root must be an array")

    valid_jobs = [
        job
        for job in job_descriptions
        if isinstance(job, dict) and validate_job_description(job)
    ]
    logger.info(
        "Loaded %d valid job description(s) from %s; discarded %d invalid "
        "record(s)",
        len(valid_jobs),
        path,
        len(job_descriptions) - len(valid_jobs),
    )
    return valid_jobs


def validate_job_description(job_description: dict[str, Any]) -> bool:
    """Check whether a job description has all required top-level fields.

    Args:
        job_description: Job-description record to validate.

    Returns:
        ``True`` if the value is a dictionary containing every required field;
        otherwise ``False``.
    """
    if not isinstance(job_description, dict):
        logger.warning(
            "Rejected job-description record: expected a dictionary, received %s",
            type(job_description).__name__,
        )
        return False

    missing_fields = REQUIRED_FIELDS.difference(job_description)
    if missing_fields:
        logger.warning(
            "Rejected job-description record %r: missing required field(s): %s",
            job_description.get("job_id", "<unknown>"),
            ", ".join(sorted(missing_fields)),
        )
        return False

    return True


def get_job_by_id(
    job_descriptions: list[dict[str, Any]], job_id: str
) -> dict[str, Any] | None:
    """Return the first job description with the requested identifier.

    Args:
        job_descriptions: Job-description records to search.
        job_id: Exact job identifier to match.

    Returns:
        The first matching job-description dictionary, or ``None`` if no match
        is found. Malformed entries are ignored.
    """
    for job_description in job_descriptions:
        is_match = (
            isinstance(job_description, dict)
            and job_description.get("job_id") == job_id
        )
        if is_match:
            return job_description

    return None


def _raise_schema_error(path: Path, reason: str) -> None:
    """Log and raise a schema error for a job-description file.

    Args:
        path: Path of the file containing the invalid schema.
        reason: Human-readable description of the schema violation.

    Raises:
        JobDescriptionSchemaError: Always raised with file context.
    """
    message = f"Invalid job-description schema in '{path}': {reason}"
    logger.error(message)
    raise JobDescriptionSchemaError(message)

# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

"""LLM-based metadata suggestions activity."""

import logging
import re
import unicodedata
from datetime import timedelta
from difflib import SequenceMatcher

from idutils.normalizers import normalize_doi
from idutils.validators import is_doi
from pydantic import BaseModel, Field
from temporalio import activity
from temporalio.common import RetryPolicy

from app.agent import build_agent
from app.config import get_settings
from app.observability import propagate_langfuse_context
from app.schemas.metadata_suggestions import ExtractedMetadata
from app.workflows.specs import WorkflowContext

logger = logging.getLogger(__name__)

EXTRACT_METADATA_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=5),
    backoff_coefficient=2,
    maximum_interval=timedelta(seconds=20),
    maximum_attempts=3,
)


class ExtractMetadataRequest(BaseModel):
    """Request to generate metadata suggestions from document text."""

    text: str = Field(description="Document text to analyze")


INSTRUCTIONS = (
    "You extract bibliographic metadata from the document text. "
    "Only include information clearly stated in the text; "
    "leave anything you cannot determine empty. "
    "If a piece of information is not present in the document text, leave that "
    "field null or empty. Never guess, invent, or use placeholder values."
)

# Below this many non-whitespace-stripped chars the extraction is effectively empty
# (broken/image-only PDFs). Models fabricate whole records, output schema
# placeholders/examples, loop in reasoning, or crash tool-call parsers on empty input.
MIN_TEXT_CHARS = 50

# Values the model returns for these fields must (fuzzily) be present in the source
# text, else they're skipped.
_PRESENCE_THRESHOLDS = {
    "title": 0.85,
    "description": 0.80,
}


def _coverage(value: str, text: str) -> float:
    """Fraction of `value` found in `text` (ordered matching blocks)."""
    sm = SequenceMatcher(None, value, text, autojunk=False)
    matched = sum(b.size for b in sm.get_matching_blocks())
    return matched / len(value) if value else 1.0


_SUP_SUB_TAGS = re.compile(r"</?su[bp]>")
_NON_ALNUM = re.compile(r"[\W_]+")


def _squash(s: str) -> str:
    """Reduce text to its letters and digits for the presence check.

    A faithful copy may still differ from the PDF text in ways that don't change
    its content: repaired spacing and line-break hyphens, ^/_ or <sup> markup,
    ligatures and math-italic letters (NFKC maps 'ﬁ' -> 'fi', '𝑏' -> 'b'),
    quotes, dashes, and punctuation. Comparing only letters and digits keeps
    those copies while invented text still fails.
    """
    s = unicodedata.normalize("NFKC", str(s or ""))
    return _NON_ALNUM.sub("", _SUP_SUB_TAGS.sub("", s).casefold())


def _clear_absent_fields(output: ExtractedMetadata, text: str) -> None:
    """Null title/description/doi whose values aren't present in the source text."""
    squashed_text = _squash(text)
    for field, threshold in _PRESENCE_THRESHOLDS.items():
        value = getattr(output, field)
        if not value:
            continue

        squashed_value = _squash(value)
        if squashed_value in squashed_text:
            continue
        coverage = _coverage(squashed_value, squashed_text)
        if coverage >= threshold:
            continue
        logger.warning(
            "Dropping %s not found in source text (coverage %.2f < %.2f): %.120r",
            field,
            coverage,
            threshold,
            value,
        )
        setattr(output, field, None)

    # DOIs wrap across lines and may carry a URL or 'doi:' prefix: normalize to
    # the bare identifier and match it whitespace-insensitively in the source.
    # A non-DOI is fabricated by construction (and would crash normalize_doi).
    if output.doi:
        if not is_doi(output.doi):
            output.doi = None
        else:
            squashed = re.sub(r"\s+", "", text).casefold()
            if normalize_doi(output.doi).casefold() not in squashed:
                output.doi = None


async def extract_metadata_record(
    text: str,
    # EXPERIMENT (remove before the PR): ablation hooks used by app/evals/ablation.py
    output_type: type[ExtractedMetadata] = ExtractedMetadata,
    presence_check: bool = True,
) -> ExtractedMetadata:
    """Shared metadata extraction helper without Temporal dependencies."""
    if len(text.strip()) < MIN_TEXT_CHARS:
        # No usable text: skip the LLM entirely rather than let it fabricate.
        return output_type()

    agent = build_agent(get_settings().llm, output_type, INSTRUCTIONS)
    result = await agent.run(text)
    if presence_check:  # EXPERIMENT: the condition only; always check in the PR
        _clear_absent_fields(result.output, text)
    return result.output


@activity.defn
async def extract_metadata_with_llm(
    request: ExtractMetadataRequest,
    context: WorkflowContext,
) -> ExtractedMetadata:
    """Extract raw metadata suggestions using an LLM."""
    with propagate_langfuse_context(context, trace_name="extract_metadata"):
        return await extract_metadata_record(request.text)

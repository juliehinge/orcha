# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

"""EXPERIMENT (temporary, remove before the PR): ablation of the abstract changes.

Three runs in ``RUNS`` separate the two LLM-side changes: baseline -> extraction
(new pdfplumber settings) -> full (plus the new description prompt). The
presence check needs no runs of its own: the model's raw output is kept and
every variant in ``CHECKS`` is replayed on it. The reported output uses C2, the
check the app ships, so the regular scores mean what they always do.
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from statistics import mean
from typing import Any

from pydantic import Field, create_model

import app.extractors.pdfplumber as plumber
from app.activities import extract_metadata as em
from app.schemas.extracted_metadata import ExtractedMetadata

from .evaluators import Evaluation, build_comparison_payload

EXPERIMENT_NAME = "abstract-ablation"

TEXT_VARIANTS: dict[str, dict[str, Any]] = {
    "old": {"x_tolerance": 2},
    "new": {"x_tolerance_ratio": 0.15, "use_text_flow": True},  # what the app ships
}

PROMPTS = {
    "old": (
        "Abstract or executive summary of the document, copied word-for-word, "
        "complete and unchanged; never paraphrase, shorten, or write a new "
        "summary. Null if the document has no abstract or summary."
    ),
    "new": ExtractedMetadata.model_fields["description"].description,  # shipped
}

# run: (text variant, prompt)
RUNS = {
    "baseline": ("old", "old"),
    "extraction": ("new", "old"),
    "full": ("new", "new"),
}


@dataclass(frozen=True)
class AblationRun:
    """One cell of the matrix, and which repeat of it this is."""

    run: str
    repeat: int = 1
    experiment_name: str = EXPERIMENT_NAME

    def __post_init__(self):
        """Reject cells that aren't in the matrix."""
        if self.run not in RUNS:
            raise ValueError(
                f"Unknown ablation run {self.run!r}; choose from {list(RUNS)}"
            )

    @property
    def text(self) -> str:
        """Text-settings variant of this cell."""
        return RUNS[self.run][0]

    @property
    def prompt(self) -> str:
        """Description prompt of this cell."""
        return RUNS[self.run][1]

    @property
    def label(self) -> str:
        """Short name, e.g. 'full-rep1'."""
        return f"{self.run}-rep{self.repeat}"

    def run_name(self) -> str:
        """Unique Langfuse run name for this cell and repeat."""
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{self.label} ({self.text}/{self.prompt}) {stamp}"

    def metadata(self) -> dict[str, str]:
        """Settings of this run, as Langfuse run metadata."""
        return {
            "ablation_run": self.run,
            "repeat": str(self.repeat),
            "text_settings": f"{self.text} {TEXT_VARIANTS[self.text]}",
            "description_prompt": self.prompt,
        }

    def configure(self) -> type[ExtractedMetadata]:
        """Apply this run's text settings; return its output schema."""
        plumber._TEXT_SETTINGS = TEXT_VARIANTS[self.text]
        return schema_for(self.prompt)


@cache
def schema_for(prompt: str) -> type[ExtractedMetadata]:
    """ExtractedMetadata with the `description` field's text replaced."""
    field = ExtractedMetadata.model_fields["description"]
    return create_model(
        "ExtractedMetadata",
        __base__=ExtractedMetadata,
        description=(
            str | None,
            Field(default=None, description=PROMPTS[prompt], examples=field.examples),
        ),
    )


# --- presence-check variants ---------------------------------------------------


def _norm_c0(s):  # original: whitespace-collapsed, casefolded
    return re.sub(r"\s+", " ", str(s or "").casefold()).strip()


def _norm_c1(s):  # first fix: also drop whitespace, ^, _, -, <sup>/<sub>
    return re.sub(r"</?su[bp]>|[\s^_\-]+", "", str(s or "").casefold())


def _check_with(norm):
    def check(output, text):
        normalized_text = norm(text)
        for field, threshold in em._PRESENCE_THRESHOLDS.items():
            value = getattr(output, field)
            if value:
                v = norm(value)
                if not (
                    v in normalized_text
                    or em._coverage(v, normalized_text) >= threshold
                ):
                    setattr(output, field, None)
        # The DOI check didn't change between variants: reuse the shipped one.
        doi_only = output.model_copy(update={"title": None, "description": None})
        em._clear_absent_fields(doi_only, text)
        output.doi = doi_only.doi

    return check


CHECKS = {
    "C0": _check_with(_norm_c0),
    "C1": _check_with(_norm_c1),
    "C2": em._clear_absent_fields,  # the shipped check
}
BUCKETS = ["GT/null", "GT/<0.8", "GT/>=0.8", "noGT/invented", "noGT/null"]


def bucket(desc: dict[str, Any]) -> str:
    """Classify one description comparison."""
    if desc["gt_present"]:
        if not desc["predicted"]:
            return "GT/null"
        return "GT/>=0.8" if desc["score"] >= 0.8 else "GT/<0.8"
    return "noGT/invented" if desc["predicted"] else "noGT/null"


def replay_checks(
    raw: ExtractedMetadata, text: str, expected: dict[str, Any]
) -> tuple[ExtractedMetadata, dict[str, Any]]:
    """Apply every check to copies of the raw output.

    Returns the C2-checked output (what the app would return) and the
    per-check description results.
    """
    checks = {}
    final = raw
    for name, check in CHECKS.items():
        checked = raw.model_copy()
        check(checked, text)
        fields = build_comparison_payload(checked.model_dump(mode="json"), expected)[
            "fields"
        ]
        checks[name] = {
            "description": fields["description"],
            "bucket": bucket(fields["description"]),
            "title_score": fields["title"]["score"],
        }
        if name == "C2":
            final = checked
    info = {
        "raw_description": raw.description,
        "raw_title": raw.title,
        "checks": checks,
    }
    return final, info


# --- Langfuse evaluators -------------------------------------------------------


def ablation_item_evaluator(output: dict[str, Any], **_: Any) -> list[Evaluation]:
    """Per item: description score under each check, and its bucket."""
    info = output.get("ablation")
    if not info:
        return []
    checks = info["checks"]
    evals = [
        Evaluation(name=f"description_{name}", value=row["description"]["score"])
        for name, row in checks.items()
    ]
    evals.append(Evaluation(name="description_bucket", value=checks["C2"]["bucket"]))
    evals.append(
        Evaluation(
            name="description_dropped_by_check",
            value=bool(info["raw_description"])
            and not checks["C2"]["description"]["predicted"],
        )
    )
    return evals


def ablation_run_evaluator(item_results: list[Any], **_: Any) -> list[Evaluation]:
    """Per run: mean description score under each check, and bucket counts."""
    infos = [r.output["ablation"] for r in item_results if r.output.get("ablation")]
    if not infos:
        return []
    evals = [
        Evaluation(
            name=f"run_description_{name}",
            value=round(
                mean(i["checks"][name]["description"]["score"] for i in infos), 4
            ),
        )
        for name in CHECKS
    ]
    for b in BUCKETS:
        evals.append(
            Evaluation(
                name=f"run_bucket_{b}",
                value=sum(i["checks"]["C2"]["bucket"] == b for i in infos),
            )
        )
    return evals

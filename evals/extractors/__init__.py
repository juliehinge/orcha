# SPDX-FileCopyrightText: 2026 CERN.
# SPDX-License-Identifier: MIT

"""PDF extraction modules."""

from .grobid import GrobidExtractor
from .pdfplumber import PdfplumberExtractor
from .pymupdf import PymupdfExtractor


def get_extractor(extractor: str = "pdfplumber"):
    """Get an extractor instance by type.

    Args:
        extractor: Either "pdfplumber", "pymupdf", or "grobid"

    Returns:
        Extractor instance

    Raises:
        ValueError: If unknown extractor type is specified
    """
    if extractor == "pdfplumber":
        return PdfplumberExtractor()
    elif extractor == "pymupdf":
        return PymupdfExtractor()
    elif extractor == "grobid":
        return GrobidExtractor()
    else:
        raise ValueError(
            f"Unknown extractor: {extractor}. Supported extractors: pdfplumber, pymupdf, grobid"
        )

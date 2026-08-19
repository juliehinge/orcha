"""GROBID-based PDF extractor.

Pipeline:
1. Extract raw page text and PDF link annotations (single pypdf parse).
2. Call GROBID for TEI header metadata (title, DOI, date, description,
   authors/affiliations/ORCID), tolerating GROBID being unavailable.
3. Try a deterministic parser for the "Name<superscript-digits>, ... /
   N Affiliation text" byline convention common in physics/HEP papers.
   When it matches, it's preferred over GROBID's own author-affiliation
   linking model, since it reproduces the page's affiliation text
   verbatim and resolves multi-affiliation authors exactly.
4. Reconcile ORCID IDs from three sources: GROBID's TEI idno records,
   PDF link annotations (clickable ORCID URLs), and a plain-text regex
   scan (many papers print ORCIDs as bare text with no hyperlink).
5. Because the eval pipeline that consumes this extractor only reads
   `full_text` (not `extra`), the resolved author/affiliation/ORCID
   data is written back into `full_text` as labeled blocks — title/DOI/
   date/description are deliberately NOT injected this way, since those
   are fields a model should read directly off the page rather than be
   handed as a pre-computed answer.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
import warnings
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Any, Dict, List, Optional

import requests
from pypdf import PdfReader

from .base import BaseExtractor
from .utils import resolve_pages

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

logger = logging.getLogger(__name__)

# Only literal superscript unicode characters are normalized. This is
# unambiguous (no ordinary text contains U+00B9 etc.) unlike a regex that
# guesses at plain-ASCII "letter followed by digit" patterns, which
# rewrites far too much ordinary prose (e.g. "page 10" -> "page[10]").
_SUPERSCRIPT_MAP = str.maketrans(
    {
        "¹": "[1]", "²": "[2]", "³": "[3]", "⁴": "[4]", "⁵": "[5]",
        "⁶": "[6]", "⁷": "[7]", "⁸": "[8]", "⁹": "[9]", "⁰": "[0]",
    }
)

# Bare ORCID pattern, e.g. "0000-0002-1825-0097". Many papers print this
# as plain text next to an author's name with no hyperlink, so relying on
# link annotations alone misses a large share of them.
_ORCID_TEXT_PATTERN = re.compile(r"\b(\d{4}-\d{4}-\d{4}-\d{3}[\dX])\b")


class GrobidExtractor(BaseExtractor):
    """Extract raw text plus resolved metadata candidates."""

    def extract(
        self,
        pdf_bytes: bytes,
        pages: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = PdfReader(BytesIO(pdf_bytes), strict=False)

        page_count = len(reader.pages)
        pages_extracted = (
            resolve_pages(pages, page_count)
            if pages is not None
            else list(range(1, min(2, page_count) + 1))
        )

        full_text = self._extract_page_text(reader, pages_extracted)
        links = self._extract_link_annotations(reader, pages_extracted)

        candidates: Dict[str, Any] = {
            "title": "",
            "description": "",
            "doi": "",
            "publication_date": "",
            "authors": [],
            "orcid_ids": [],
        }
        grobid_error: Optional[str] = None

        try:
            tei_xml = self._fetch_tei(pdf_bytes)
            root = ET.fromstring(tei_xml)
        except (requests.RequestException, ET.ParseError) as exc:
            grobid_error = str(exc)
            root = None
            logger.warning("GROBID extraction failed, continuing without candidates: %s", exc)

        if root is not None:
            candidates["title"] = self._extract_title(root)
            candidates["description"] = self._extract_description(root)
            candidates["doi"] = self._extract_doi(root, full_text)
            candidates["publication_date"] = self._extract_publication_date(root)
            candidates["authors"] = self._extract_authors(root)

        # Prefer the deterministic numbered-marker byline parser when it
        # matches this paper's format — it resolves author<->affiliation
        # linking exactly (including multi-affiliation authors) and
        # reproduces the page's affiliation text verbatim, which GROBID's
        # own linking model gets wrong often enough to matter. Falls back
        # to whatever GROBID found when the paper doesn't use this
        # convention (named superscripts, footnote symbols, inline
        # affiliations, etc).
        numbered_authors = self._extract_numbered_byline_authors(full_text)
        if numbered_authors:
            candidates["authors"] = self._merge_authors_with_orcid(
                numbered_authors, candidates["authors"]
            )

        candidates["orcid_ids"] = self._reconcile_orcid_ids(
            candidates["authors"], links, full_text
        )

        # The eval pipeline only reads `full_text`, not `extra`, so the
        # resolved author/affiliation/ORCID structure has to actually
        # reach the model this way.
        if candidates["orcid_ids"]:
            full_text = self._insert_orcid_block(full_text, candidates["orcid_ids"])
        if candidates["authors"]:
            full_text = self._insert_author_block(full_text, candidates["authors"])

        return {
            "full_text": full_text,
            "page_count": page_count,
            "pages_extracted": pages_extracted,
            "links": links,
            "extra": {
                "candidates": candidates,
                "grobid_url": self._grobid_url(),
                "grobid_error": grobid_error,
            },
        }

    # ------------------------------------------------------------------
    # GROBID call
    # ------------------------------------------------------------------

    def _grobid_url(self) -> str:
        return os.getenv(
            "GROBID_URL", "http://localhost:8070/api/processHeaderDocument"
        )

    def _fetch_tei(self, pdf_bytes: bytes) -> str:
        response = requests.post(
            self._grobid_url(),
            headers={"Accept": "application/xml"},
            files={"input": ("document.pdf", pdf_bytes, "application/pdf")},
            data={
                "consolidateHeader": "1",
                # Ask GROBID to also return the literal, unparsed
                # affiliation string per author (tei:note[@type='raw_affiliation'])
                # alongside its structured orgName/address breakdown. The
                # structured version reorders and fragments the original
                # text, which tends to score worse against an expected
                # affiliation string than the raw text does.
                "includeRawAffiliations": "1",
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.text

    # ------------------------------------------------------------------
    # TEI field extraction
    # ------------------------------------------------------------------

    def _extract_title(self, root: ET.Element) -> str:
        for xpath in (
            ".//tei:sourceDesc//tei:titleStmt/tei:title",
            ".//tei:titleStmt/tei:title",
            ".//tei:fileDesc/tei:titleStmt/tei:title",
            ".//tei:analytic/tei:title",
        ):
            title = root.findtext(xpath, default="", namespaces=TEI_NS)
            if title:
                return self._clean_text(title)
        return ""

    def _extract_doi(self, root: ET.Element, page_text: str) -> str:
        for xpath in (".//tei:idno[@type='DOI']", ".//tei:idno[@type='doi']"):
            doi = root.findtext(xpath, default="", namespaces=TEI_NS)
            doi = self._extract_doi_from_text(doi)
            if doi:
                return doi

        xml_text = " ".join(root.itertext())
        doi = self._extract_doi_from_text(xml_text)
        if doi:
            return doi

        return self._extract_doi_from_text(page_text)

    def _extract_description(self, root: ET.Element) -> str:
        abstract_xpaths = (
            ".//tei:profileDesc/tei:abstract",
            ".//tei:abstract",
            ".//tei:div[@type='abstract']",
            ".//tei:note[@type='abstract']",
        )
        for xpath in abstract_xpaths:
            abstract_el = root.find(xpath, TEI_NS)
            if abstract_el is None:
                continue

            paragraphs = [
                self._clean_text("".join(paragraph.itertext()))
                for paragraph in abstract_el.findall(".//tei:p", TEI_NS)
            ]
            paragraphs = [p for p in paragraphs if p]
            if paragraphs:
                return "\n\n".join(paragraphs)

            description = self._clean_text("".join(abstract_el.itertext()))
            if description:
                return description

        return ""

    def _extract_publication_date(self, root: ET.Element) -> str:
        candidate_xpaths = (
            ".//tei:publicationStmt/tei:date[@when]",
            ".//tei:publicationStmt/tei:date",
            ".//tei:sourceDesc//tei:date[@when]",
            ".//tei:sourceDesc//tei:date",
            ".//tei:profileDesc//tei:date[@when]",
            ".//tei:profileDesc//tei:date",
        )
        for xpath in candidate_xpaths:
            for date_el in root.findall(xpath, TEI_NS):
                raw_date = date_el.get("when") or (date_el.text or "")
                normalized = self._normalize_date_value(raw_date)
                if normalized:
                    return normalized
        return ""

    def _extract_authors(self, root: ET.Element) -> List[Dict[str, Any]]:
        authors: List[Dict[str, Any]] = []
        for author_el in root.findall(".//tei:sourceDesc//tei:author", TEI_NS):
            pers_name = author_el.find("tei:persName", TEI_NS)
            if pers_name is None:
                continue

            forename = pers_name.findtext("tei:forename", default="", namespaces=TEI_NS)
            surname = pers_name.findtext("tei:surname", default="", namespaces=TEI_NS)
            name = f"{forename} {surname}".strip()

            affiliations: List[str] = []
            for affil_el in author_el.findall("tei:affiliation", TEI_NS):
                # Prefer the raw, unparsed affiliation string (requires
                # includeRawAffiliations=1 on the request) — it matches
                # the original page wording, whereas the structured
                # orgName/address breakdown below often reorders and
                # fragments it in ways that score poorly against an
                # expected affiliation string.
                raw_note = affil_el.find("tei:note[@type='raw_affiliation']", TEI_NS)
                raw_text = self._clean_text(
                    "".join(raw_note.itertext()) if raw_note is not None else None
                )
                if raw_text:
                    raw_text = re.sub(r"^\[?\d{1,2}\]?[\s,.:]*", "", raw_text).strip()
                    if raw_text:
                        affiliations.append(raw_text)
                        continue

                org_parts = [
                    self._clean_text(org.text)
                    for org in affil_el.findall("tei:orgName", TEI_NS)
                    if self._clean_text(org.text)
                ]
                address_el = affil_el.find("tei:address", TEI_NS)
                if address_el is not None:
                    addr_parts = [
                        self._clean_text(node.text)
                        for node in address_el
                        if self._clean_text(node.text)
                    ]
                    org_parts.extend(addr_parts)
                if org_parts:
                    affiliations.append(", ".join(org_parts))

            orcid_el = author_el.find("tei:idno[@type='ORCID']", TEI_NS)
            orcid = self._clean_text(orcid_el.text if orcid_el is not None else None)

            authors.append(
                {
                    "name": name,
                    "affiliations": affiliations,
                    "orcid": orcid or None,
                }
            )
        return authors

    # ------------------------------------------------------------------
    # Numbered-marker byline parser
    #
    # Handles the common convention:
    #   Nicolas Berger1, Claudia Bertella2,3, ... and Hongtao Yang12
    #   ...
    #   1 LAPP, Annecy, France
    #   2 CERN, Geneva, Switzerland
    # where a digit (or comma-separated digits) glued directly onto each
    # author's name refers to a numbered affiliation list elsewhere on
    # the page. This is deterministic and format-specific: it returns an
    # empty list (rather than a partial/garbage match) for papers that
    # use a different convention, so callers should fall back to GROBID's
    # own author/affiliation extraction when this returns nothing.
    # ------------------------------------------------------------------

    _AFFIL_LIST_LINE = re.compile(r"^(\d{1,3})\s+[A-Za-zÀ-ÖØ-öø-ÿ]")
    _AFFIL_LIST_ENTRY = re.compile(r"^(\d{1,3})\s+(.*)$")
    _BYLINE_END_MARKERS = re.compile(r"^(Abstract|Part of|published in)", re.IGNORECASE)
    _BYLINE_LINE_END = re.compile(r"[A-Za-z][.,]?\d{1,2}(,\d{1,2})*,?\s*$")
    _AUTHOR_ENTRY = re.compile(r"^(.*?)\s*((?:\d{1,2},)*\d{1,2})$")

    def _extract_numbered_affiliation_map(self, lines: List[str]) -> Dict[int, str]:
        aff_start = None
        for i, line in enumerate(lines):
            if self._AFFIL_LIST_LINE.match(line.strip()):
                aff_start = i
                break
        if aff_start is None:
            return {}

        aff_map: Dict[int, str] = {}
        current_num: Optional[int] = None
        current_text: List[str] = []
        for line in lines[aff_start:]:
            stripped = line.strip()
            match = self._AFFIL_LIST_ENTRY.match(stripped)
            if match:
                if current_num is not None:
                    aff_map[current_num] = self._clean_text(" ".join(current_text)).strip(" ,")
                current_num = int(match.group(1))
                current_text = [match.group(2)]
            elif current_num is not None and stripped and not re.fullmatch(r"\d{1,3}", stripped):
                current_text.append(stripped)
            elif re.fullmatch(r"\d{1,3}", stripped):
                break  # page-number footer
        if current_num is not None:
            aff_map[current_num] = self._clean_text(" ".join(current_text)).strip(" ,")
        return aff_map

    def _extract_numbered_byline_authors(self, full_text: str) -> List[Dict[str, Any]]:
        lines = full_text.splitlines()

        start = None
        end = None
        for i, line in enumerate(lines):
            if start is None and i > 0 and self._BYLINE_LINE_END.search(line.strip()):
                start = i
            if start is not None and self._BYLINE_END_MARKERS.match(line.strip()):
                end = i
                break
        if start is None:
            return []

        aff_map = self._extract_numbered_affiliation_map(lines)
        if not aff_map:
            return []

        block = " ".join(l.strip() for l in lines[start:end])
        block = re.sub(r"\s+,", ",", block)
        # Protect "2,3"-style affiliation-number separators (digit,digit,
        # no space) from being mistaken for author-separating commas.
        block = re.sub(r"(?<=\d),(?=\d)", "\uE000", block)

        raw_entries = re.split(r",\s*(?:and\s+)?|\s+and\s+", block)
        authors: List[Dict[str, Any]] = []
        for entry in raw_entries:
            entry = entry.strip().rstrip(".")
            if not entry:
                continue
            entry = entry.replace("\uE000", ",")
            match = self._AUTHOR_ENTRY.match(entry)
            if not match:
                continue
            name = re.sub(r"\s+\.", ".", match.group(1)).strip()
            if not name:
                continue
            numbers = [int(n) for n in match.group(2).split(",")]
            affiliations = [aff_map[n] for n in numbers if n in aff_map]
            if not affiliations:
                continue
            authors.append({"name": name, "affiliations": affiliations, "orcid": None})

        return authors

    def _merge_authors_with_orcid(
        self,
        numbered_authors: List[Dict[str, Any]],
        grobid_authors: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Attach ORCID from GROBID's author records onto the
        numbered-parser results, matched by surname (last word)."""
        orcid_by_surname: Dict[str, str] = {}
        for author in grobid_authors:
            orcid = author.get("orcid")
            name = (author.get("name") or "").strip()
            if orcid and name:
                surname = name.split()[-1].lower()
                orcid_by_surname[surname] = orcid

        merged: List[Dict[str, Any]] = []
        for author in numbered_authors:
            name = author["name"]
            surname = name.split()[-1].lower() if name else ""
            merged.append(
                {
                    "name": name,
                    "affiliations": author["affiliations"],
                    "orcid": orcid_by_surname.get(surname),
                }
            )
        return merged

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _clean_text(self, value: Optional[str]) -> str:
        return re.sub(r"\s+", " ", value or "").strip()

    def _extract_doi_from_text(self, value: Optional[str]) -> str:
        text = self._clean_text(value)
        if not text:
            return ""
        match = re.search(
            r"(?:https?://(?:dx\.)?doi\.org/|doi:\s*|DOI:\s*)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
            text,
            flags=re.IGNORECASE,
        )
        return match.group(1).rstrip(".);, ") if match else ""

    def _normalize_date_value(self, value: Optional[str]) -> str:
        text = self._clean_text(value)
        if not text:
            return ""
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            return text
        if re.fullmatch(r"\d{4}-\d{2}", text):
            return text
        if re.fullmatch(r"\d{4}", text):
            return text

        from datetime import datetime

        try:
            return datetime.fromisoformat(text).date().isoformat()
        except ValueError:
            pass

        for fmt in ("%Y/%m/%d", "%d %B %Y", "%B %d %Y", "%d %b %Y", "%b %d %Y", "%B %Y", "%b %Y"):
            try:
                parsed = datetime.strptime(text, fmt)
                return parsed.strftime("%Y-%m") if fmt in {"%B %Y", "%b %Y"} else parsed.date().isoformat()
            except ValueError:
                continue
        return text

    # ------------------------------------------------------------------
    # Link / ORCID handling
    # ------------------------------------------------------------------

    def _extract_link_annotations(
        self, reader: PdfReader, pages: List[int]
    ) -> List[Dict[str, Any]]:
        links: List[Dict[str, Any]] = []
        for page_number in pages:
            page_index = page_number - 1
            if page_index < 0 or page_index >= len(reader.pages):
                continue

            page = reader.pages[page_index]
            annotations = page.get("/Annots") or []
            for annotation_ref in annotations:
                try:
                    annotation = annotation_ref.get_object()
                except Exception:
                    continue
                if annotation.get("/Subtype") != "/Link":
                    continue

                action = annotation.get("/A")
                uri = str(action.get("/URI")) if action and action.get("/URI") else ""
                if not uri:
                    continue

                links.append({"page": page_number, "uri": uri, "type": self._classify_link(uri)})
        return links

    def _extract_orcid_ids_from_text(self, text: str) -> List[str]:
        seen: List[str] = []
        for match in _ORCID_TEXT_PATTERN.finditer(text or ""):
            candidate = match.group(1)
            if candidate not in seen:
                seen.append(candidate)
        return seen

    def _reconcile_orcid_ids(
        self,
        authors: List[Dict[str, Any]],
        links: List[Dict[str, Any]],
        full_text: str,
    ) -> List[str]:
        """Merge ORCID IDs found via TEI author records, PDF link URIs,
        and a plain-text scan of the extracted page text (many papers
        print ORCIDs as bare text with no hyperlink)."""
        orcid_ids: List[str] = []

        for author in authors:
            orcid = author.get("orcid")
            if orcid and orcid not in orcid_ids:
                orcid_ids.append(orcid)

        for link in links:
            uri = link.get("uri") or ""
            if "orcid.org" not in uri.lower():
                continue
            orcid_id = self._extract_orcid_id(uri)
            if orcid_id and orcid_id not in orcid_ids:
                orcid_ids.append(orcid_id)

        for orcid_id in self._extract_orcid_ids_from_text(full_text):
            if orcid_id not in orcid_ids:
                orcid_ids.append(orcid_id)

        return orcid_ids

    def _extract_orcid_id(self, url: str) -> Optional[str]:
        match = re.search(r"orcid\.org/(\d{4}-\d{4}-\d{4}-\d{3}[\dX])", url)
        return match.group(1) if match else None

    # ------------------------------------------------------------------
    # full_text enrichment
    #
    # The eval pipeline only reads `full_text` (see evals/run.py), not
    # `extra`, so resolved author/affiliation/ORCID structure has to be
    # written back into the text to actually reach the model. Title/DOI/
    # description/date are deliberately NOT injected this way — those
    # are fields the model should read directly off the page, not be
    # handed as a pre-computed answer.
    # ------------------------------------------------------------------

    def _insert_author_block(self, text: str, authors: List[Dict[str, Any]]) -> str:
        if not authors:
            return text

        lines = text.splitlines()
        insert_at = 0
        for index, line in enumerate(lines):
            if line.startswith("#"):
                insert_at = index + 1
                break

        block_lines = ["Extracted authors (resolved author/affiliation/ORCID linking):"]
        for i, author in enumerate(authors, start=1):
            parts = [author.get("name") or ""]
            affiliations = author.get("affiliations") or []
            if affiliations:
                parts.append("affiliations: " + "; ".join(affiliations))
            orcid = author.get("orcid")
            if orcid:
                parts.append(f"orcid: {orcid}")
            block_lines.append(f"- Author {i}: " + " | ".join(part for part in parts if part))

        block_lines.append("")
        return "\n".join(lines[:insert_at] + block_lines + lines[insert_at:])

    def _insert_orcid_block(self, text: str, orcid_ids: List[str]) -> str:
        if not orcid_ids:
            return text

        lines = text.splitlines()
        insert_at = 0
        for index, line in enumerate(lines):
            if line.startswith("#"):
                insert_at = index + 1
                break

        block = ["ORCID IDs found in document: " + ", ".join(orcid_ids), ""]
        return "\n".join(lines[:insert_at] + block + lines[insert_at:])

    # ------------------------------------------------------------------
    # Page text
    # ------------------------------------------------------------------

    def _normalize_markers_in_text(self, text: str) -> str:
        """Normalize literal superscript unicode only. Deliberately does
        NOT try to guess ASCII "letter+digit" superscript patterns, since
        that rewrites ordinary text like "page 10" into "page[10]"."""
        t = unicodedata.normalize("NFKC", text or "")
        t = t.translate(_SUPERSCRIPT_MAP)
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()

    def _extract_page_text(self, reader: PdfReader, pages: List[int]) -> str:
        page_texts: List[str] = []
        for page_number in pages:
            page_index = page_number - 1
            if page_index < 0 or page_index >= len(reader.pages):
                continue

            text = reader.pages[page_index].extract_text() or ""
            text = self._normalize_markers_in_text(text).strip()
            if text:
                page_texts.append(text)

        return "\n\n".join(page_texts).rstrip() + ("\n" if page_texts else "")
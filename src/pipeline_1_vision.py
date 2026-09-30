"""Pipeline 1 - visual extraction.

Parses a PDF, crops every chart and table into an image, has llama3.2-vision
describe each one, splices those descriptions back into the page text, and caches
the result. Then evicts the vision model from RAM so pipeline 2 can run.

This is the only stage that loads an 11B model; nothing else may run alongside it.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

import pdfplumber
from tqdm import tqdm

from src import config, ollama_client, state
from src.text_utils import file_hash

log = logging.getLogger(__name__)

VISION_PROMPT = (
    "Analyze this visual element. If it's a chart, describe the trends and "
    "specific numerical data points. If it's a table, format it as Markdown."
)

# Flush the cache to disk every N pages so a crash costs at most N pages of
# vision inference.
FLUSH_EVERY_PAGES = 5


def _clamp_bbox(
    bbox: tuple[float, float, float, float],
    page_bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    """Intersect a region with the page, or return None if unusable.

    pdfplumber routinely reports image bboxes that extend beyond the page
    (negative tops, x1 past the media box); passing those to within_bbox raises
    ValueError and kills the run. Clamping is not optional.
    """
    x0, top, x1, bottom = bbox
    px0, ptop, px1, pbottom = page_bbox

    x0 = max(float(x0), float(px0))
    top = max(float(top), float(ptop))
    x1 = min(float(x1), float(px1))
    bottom = min(float(bottom), float(pbottom))

    if x1 - x0 < config.MIN_CROP_POINTS or bottom - top < config.MIN_CROP_POINTS:
        # Rules, bullets, logos. Not worth an 11B forward pass.
        return None
    return (x0, top, x1, bottom)


def _collect_regions(page: pdfplumber.page.Page) -> list[tuple[str, tuple]]:
    """Bounding boxes of visual elements on a page, de-overlapped and capped."""
    regions: list[tuple[str, tuple]] = []

    for img in page.images or []:
        bbox = _clamp_bbox(
            (img["x0"], img["top"], img["x1"], img["bottom"]), page.bbox
        )
        if bbox:
            regions.append(("image", bbox))

    # find_tables() gives coordinates; extract_tables() would give cell text, but
    # we want the rendered region so the vision model sees merged cells, headers
    # and footnotes as laid out.
    try:
        for table in page.find_tables():
            bbox = _clamp_bbox(tuple(table.bbox), page.bbox)
            if bbox:
                regions.append(("table", bbox))
    except Exception as exc:  # pdfplumber's table finder is fragile on odd PDFs
        log.debug("Table detection failed on page %s: %s", page.page_number, exc)

    # Drop regions almost entirely contained in an earlier one (an image inside a
    # detected table, typically) so we don't pay twice for the same pixels.
    deduped: list[tuple[str, tuple]] = []
    for kind, bbox in regions:
        if not any(_mostly_inside(bbox, kept) for _, kept in deduped):
            deduped.append((kind, bbox))

    # Drop text columns that find_tables() mistook for tables. An embedded image
    # is kept regardless - a chart legitimately contains few digits as text.
    kept: list[tuple[str, tuple]] = []
    for kind, bbox in deduped:
        if kind == "image":
            kept.append((kind, bbox))
            continue
        try:
            density = _digit_density(page.within_bbox(bbox).extract_text() or "")
        except Exception:
            density = 0.0
        if density >= config.MIN_TABLE_DIGIT_PCT:
            kept.append((kind, bbox))
        else:
            log.debug(
                "p%s: skipping %s region (%.1f%% digits - prose, not a table)",
                page.page_number, kind, density,
            )

    # Detection missed it entirely: the page is numeric but nothing survived.
    # Send the whole page rather than lose the table.
    if not kept:
        page_text = page.extract_text(x_tolerance=config.X_TOLERANCE) or ""
        if _digit_density(page_text) >= config.PAGE_TABLE_DIGIT_PCT:
            log.debug("p%s: no region found but page is numeric - sending full page",
                      page.page_number)
            kept.append(("full-page", tuple(page.bbox)))

    return kept[: config.MAX_VISUALS_PER_PAGE]


def _digit_density(text: str) -> float:
    """Percentage of non-space characters that are digits.

    The cheap discriminator between a numeric table (>= ~25%) and body prose
    (typically < 5%) on this kind of document.
    """
    compact = "".join(text.split())
    if not compact:
        return 0.0
    return 100.0 * sum(c.isdigit() for c in compact) / len(compact)


def _mostly_inside(inner: tuple, outer: tuple, threshold: float = 0.85) -> bool:
    ix0, itop, ix1, ibottom = inner
    ox0, otop, ox1, obottom = outer
    ow = max(0.0, min(ix1, ox1) - max(ix0, ox0))
    oh = max(0.0, min(ibottom, obottom) - max(itop, otop))
    inter = ow * oh
    area = max(1e-6, (ix1 - ix0) * (ibottom - itop))
    return inter / area >= threshold


def _crop_to_base64(page: pdfplumber.page.Page, bbox: tuple) -> str:
    """Rasterise a page region to a base64 PNG held only in memory."""
    cropped = page.within_bbox(bbox)
    rendered = cropped.to_image(resolution=config.CROP_DPI)
    buffer = io.BytesIO()
    rendered.original.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _describe(image_b64: str) -> str:
    return ollama_client.generate(
        config.VISION_MODEL,
        VISION_PROMPT,
        images=[image_b64],
        num_ctx=config.VISION_NUM_CTX,
        temperature=0.0,
        timeout=config.VISION_TIMEOUT_S,
    )


def ingest(pdf_path: Path, *, force: bool = False) -> dict[str, Any]:
    """Parse one PDF into data/processed/document_cache.json.

    Returns a summary dict. Skips work already done unless `force`.
    """
    config.ensure_dirs()
    pdf_path = Path(pdf_path).expanduser().resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"No such PDF: {pdf_path}")

    ollama_client.require_models(config.VISION_MODEL)

    doc_hash = file_hash(pdf_path)
    run_state = state.load_state()

    if state.document_is_complete(run_state, doc_hash) and not force:
        log.info(
            "%s is already ingested (hash %s). Use --force to redo it.",
            pdf_path.name,
            doc_hash[:12],
        )
        return {"doc_hash": doc_hash, "skipped": True, **run_state["documents"][doc_hash]}

    cache = state.load_cache()
    pages_out: list[dict[str, Any]] = []
    visual_total = 0

    log.info("Ingesting %s (hash %s)", pdf_path.name, doc_hash[:12])

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            total = len(pdf.pages)
            for index, page in enumerate(tqdm(pdf.pages, desc="pages", unit="pg")):
                page_no = index + 1
                page_text = (
                    page.extract_text(x_tolerance=config.X_TOLERANCE) or ""
                ).strip()
                visuals: list[dict[str, Any]] = []

                for slot, (kind, bbox) in enumerate(_collect_regions(page), start=1):
                    try:
                        description = _describe(_crop_to_base64(page, bbox))
                    except Exception as exc:
                        # One bad crop must not kill a 200-page run.
                        log.warning("Vision failed p%d region %d: %s", page_no, slot, exc)
                        state.record_failure(
                            "vision",
                            {
                                "source": str(pdf_path),
                                "doc_hash": doc_hash,
                                "page": page_no,
                                "bbox": list(bbox),
                                "kind": kind,
                                "error": str(exc),
                            },
                        )
                        continue

                    if description:
                        visuals.append(
                            {"slot": slot, "kind": kind, "bbox": list(bbox),
                             "description": description}
                        )
                        visual_total += 1

                # Splice descriptions into the page text under explicit markers so
                # provenance survives chunking and reaches the final answer.
                combined = page_text
                for v in visuals:
                    combined += (
                        f"\n\n[VISUAL ELEMENT {v['slot']} - {v['kind']}]\n"
                        f"{v['description']}"
                    )

                pages_out.append(
                    {
                        "page_no": page_no,
                        "text": combined.strip(),
                        "visuals": visuals,
                    }
                )

                if page_no % FLUSH_EVERY_PAGES == 0 or page_no == total:
                    cache[doc_hash] = {
                        "source": str(pdf_path),
                        "filename": pdf_path.name,
                        "page_count": total,
                        "pages": pages_out,
                    }
                    state.save_cache(cache)
                    state.mark_document(
                        run_state, doc_hash, source=str(pdf_path),
                        pages=len(pages_out), status="partial", visuals=visual_total,
                    )
                    state.save_state(run_state)

        state.mark_document(
            run_state, doc_hash, source=str(pdf_path),
            pages=len(pages_out), status="complete", visuals=visual_total,
        )
        state.save_state(run_state)
        log.info(
            "Ingested %d pages, %d visual elements described.",
            len(pages_out), visual_total,
        )
    finally:
        # Non-negotiable: the 11B model must leave RAM before anything else runs,
        # including on the failure path.
        ollama_client.unload(config.VISION_MODEL)

    return {
        "doc_hash": doc_hash,
        "skipped": False,
        "pages": len(pages_out),
        "visuals": visual_total,
        "source": str(pdf_path),
    }

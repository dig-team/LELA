import os
import argparse
import gc
import importlib.util
import logging
import tempfile
import threading
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

try:
    import gradio as gr
except ImportError as e:
    raise ImportError(
        "The LELA web UI requires gradio. Install it with:\n"
        "    pip install \"lela[ui]\"\n"
        "or with uv:\n"
        "    uv sync --extra ui"
    ) from e
import torch

# Global cancellation event for cooperative cancellation
_cancel_event = threading.Event()

from lela import Lela
from lela.memory import get_system_resources
from lela.defaults import (
    AVAILABLE_LLM_MODELS as LLM_MODEL_CHOICES,
    AVAILABLE_EMBEDDING_MODELS as EMBEDDING_MODEL_CHOICES,
    AVAILABLE_CROSS_ENCODER_MODELS as CROSS_ENCODER_MODEL_CHOICES,
    AVAILABLE_VLLM_RERANKER_MODELS as VLLM_RERANKER_MODEL_CHOICES,
    DEFAULT_VLLM_RERANKER_MODEL,
    DEFAULT_GLINER_MODEL,
    DEFAULT_GLINER_VRAM_GB,
    DEFAULT_MAX_MODEL_LEN,
    get_model_vram_gb,
)
from lela.llm_pool import clear_all_models

_BADGE_STYLE = (
    "display:inline-flex;align-items:center;gap:4px;padding:3px 10px;"
    "border-radius:6px;background:#f3f4f6;color:#1f2937;text-decoration:none;"
    "font-size:0.85em;border:1px solid #d1d5db;"
)

_GITHUB_SVG = '<svg height="16" width="16" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>'

_ARXIV_SVG = '<svg height="16" width="12" viewBox="0 0 17.732 24.269" xmlns="http://www.w3.org/2000/svg"><path d="M573.549,280.916l2.266,2.738,6.674-7.84c.353-.47.52-.717.353-1.117a1.218,1.218,0,0,0-1.061-.748h0a.953.953,0,0,0-.712.262Z" transform="translate(-566.984 -271.548)" fill="#bdb9b4"/><path d="M579.525,282.225l-10.606-10.174a1.413,1.413,0,0,0-.834-.5,1.09,1.09,0,0,0-1.027.66c-.167.4-.047.681.319,1.206l8.44,10.242h0l-6.282,7.716a1.336,1.336,0,0,0-.323,1.3,1.114,1.114,0,0,0,1.04.69A.992.992,0,0,0,571,293l8.519-7.92A1.924,1.924,0,0,0,579.525,282.225Z" transform="translate(-566.984 -271.548)" fill="#b31b1b"/><path d="M584.32,293.912l-8.525-10.275,0,0L573.53,280.9l-1.389,1.254a2.063,2.063,0,0,0,0,2.965l10.812,10.419a.925.925,0,0,0,.742.282,1.039,1.039,0,0,0,.953-.667A1.261,1.261,0,0,0,584.32,293.912Z" transform="translate(-566.984 -271.548)" fill="#bdb9b4"/></svg>'


def _badge(href: str, icon: str, label: str) -> str:
    return (
        f'<a href="{href}" target="_blank" style="{_BADGE_STYLE}">'
        f"{icon}\n      {label}\n    </a>"
    )


TITLE_HTML = f"""
<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:4px;">
  <h1 style="margin:0;font-size:1.8em;">LELA: An End-to-End LLM-Based Entity Linking Framework with Zero-Shot Domain Adaptation</h1>
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
    {_badge("https://github.com/dig-team/LELA", _GITHUB_SVG, "GitHub")}
    {_badge("https://arxiv.org/abs/2605.26956", _ARXIV_SVG, "Demo paper (IJCAI-ECAI 2026)")}
    {_badge("https://arxiv.org/abs/2601.05192", _ARXIV_SVG, "Full paper (ISWC 2026)")}
  </div>
</div>
"""

LOGO = """
<div style="display: flex; justify-content: center; align-items: center; gap: 40px; margin-top: 40px;">
    <img src="https://www.telecom-paris.fr/wp-content-EvDsK19/uploads/2024/01/logo_telecom_ipparis_rvb_fond_h-768x359.png" alt="Telecom Paris Logo" style="height: 80px;">
    <img src="https://www.ip-paris.fr/sites/default/files/image002.png" alt="IP Paris Logo" style="height: 80px;">
    <img src="https://yago-knowledge.org/assets/images/logo.png" alt="YAGO Logo" style="height: 80px;">
</div>
"""


def _is_vllm_usable() -> bool:
    """Check if vllm is installed and CUDA is available for it to run."""
    vllm_installed = importlib.util.find_spec("vllm") is not None
    cuda_available = False
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    # Log for debugging
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"vLLM check: installed={vllm_installed}, cuda={cuda_available}")

    return vllm_installed and cuda_available


def get_available_components() -> Dict[str, List[str]]:
    """Get list of available spaCy pipeline components."""
    # These map to spaCy factories registered in lela.spacy_components
    available_disambiguators = [
        "none",
        "first",
        "vllm",
        "transformers",
        "openai_api",
    ]

    return {
        "loaders": ["text", "pdf", "docx", "html", "json", "jsonl"],
        "ner": ["regex", "spacy", "gliner"],
        "candidates": ["none", "fuzzy", "bm25", "dense", "openai_api_dense"],
        "rerankers": [
            "none",
            "cross_encoder",
            "vllm_api_client",
            "llama_server",
            # "embedder_transformers",
            # "embedder_vllm",
            "cross_encoder_vllm",
        ],
        "disambiguators": available_disambiguators,
        "knowledge_bases": ["jsonl"],
    }


GRAY_COLOR = "#D1D5DB"  # Tailwind gray-300 (light gray)

# Color palette for consistent entity colors
# Based on D3 Category20 / Tableau 20 - industry standard for categorical data visualization
# First 10: saturated colors for primary distinction
# Next 10: lighter variants for additional categories
# Source: https://d3js.org/d3-scale-chromatic/categorical
ENTITY_COLORS = [
    "#1F77B4",  # Blue
    "#FF7F0E",  # Orange
    "#2CA02C",  # Green
    "#D62728",  # Red
    "#9467BD",  # Purple
    "#8C564B",  # Brown
    "#E377C2",  # Pink
    "#BCBD22",  # Olive
    "#17BECF",  # Cyan
    "#AEC7E8",  # Light Blue
    "#FFBB78",  # Light Orange
    "#98DF8A",  # Light Green
    "#FF9896",  # Light Red
    "#C5B0D5",  # Light Purple
    "#C49C94",  # Light Brown
    "#F7B6D2",  # Light Pink
    "#DBDB8D",  # Light Olive
    "#9EDAE5",  # Light Cyan
]

UNLINKED_COLOR = "#9E9E9E"  # Gray for entities not linked to KB


ERROR_COLOR = "#DC2626"  # Tailwind red-600
MIN_VLLM_CONTEXT_LEN = 512
MAX_VLLM_CONTEXT_LEN = 32768
VLLM_CONTEXT_LEN_STEP = 256
DEFAULT_WEB_VLLM_CONTEXT_LEN = min(4096, DEFAULT_MAX_MODEL_LEN)


def highlighted_to_html(
    highlighted: List[Tuple[str, Optional[str], Optional[Dict]]],
    color_map: Dict[str, str],
    show_legend: bool = True,
) -> str:
    """Convert highlighted text data to HTML with inline styles, interactive hover, and popups.

    This bypasses Gradio's buggy HighlightedText component.
    When hovering a legend item, only entities of that type are highlighted; others turn gray.
    When hovering an entity, a popup shows detailed information.
    """
    import html
    import hashlib
    import uuid
    import json

    def label_to_class(label: str) -> str:
        """Convert label to a valid CSS class name."""
        return "ent-" + hashlib.md5(label.encode()).hexdigest()[:8]

    def escape_js_string(s: str) -> str:
        """Escape a string for use in JavaScript within HTML attributes."""
        if s is None:
            return ""
        s = str(s)
        # Escape backslashes first, then other special chars
        s = s.replace("\\", "\\\\")
        s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        # Escape single quotes for JS BEFORE html.escape.
        # html.escape converts \' to \&#x27;, and the browser's HTML parser
        # decodes &#x27; back to ', yielding \' in the JS string — a valid escape.
        s = s.replace("'", "\\'")
        s = html.escape(s)
        return s

    # Generate unique container ID for this render
    container_id = f"ner-output-{uuid.uuid4().hex[:8]}"
    popup_id = f"popup-{container_id}"

    # Collect entity info by label for legend popups (first instance + count)
    label_entity_info = {}  # label -> entity_info (first instance)
    label_counts = {}  # label -> count of occurrences

    # Build text with entity marks
    parts = []
    for item in highlighted:
        text, label = item[0], item[1]
        entity_info = item[2] if len(item) > 2 else None

        escaped_text = html.escape(text)
        if label is None:
            parts.append(escaped_text)
        else:
            # Use per-instance color if available, otherwise fall back to color_map
            color = (
                entity_info.get("display_color") if entity_info else None
            ) or color_map.get(label, "#808080")
            css_class = label_to_class(label)

            # Track count and store first entity info for each label (for legend popup)
            label_counts[label] = label_counts.get(label, 0) + 1
            if label not in label_entity_info and entity_info:
                label_entity_info[label] = entity_info

            # Build popup content for this entity
            popup_lines = []
            if entity_info:
                if entity_info.get("kb_title"):
                    popup_lines.append(
                        f"<strong style=&quot;color:#333;&quot;>{escape_js_string(entity_info['kb_title'])}</strong>"
                    )
                if entity_info.get("kb_id"):
                    popup_lines.append(
                        f"<em style=&quot;color:#555;&quot;>ID: {escape_js_string(entity_info['kb_id'])}</em>"
                    )
                if entity_info.get("type"):
                    popup_lines.append(
                        f"<span style=&quot;color:#333;&quot;>Type: {escape_js_string(entity_info['type'])}</span>"
                    )
                if entity_info.get("mention") and entity_info.get(
                    "mention"
                ) != entity_info.get("kb_title"):
                    popup_lines.append(
                        f"<span style=&quot;color:#333;&quot;>Mention: &quot;{escape_js_string(entity_info['mention'])}&quot;</span>"
                    )
                if entity_info.get("kb_description"):
                    desc = entity_info["kb_description"]
                    if len(desc) > 150:
                        desc = desc[:150] + "..."
                    popup_lines.append(
                        f"<div style=&quot;margin-top:0.3em;font-size:0.9em;color:#666;&quot;>{escape_js_string(desc)}</div>"
                    )

            popup_content = (
                "<br>".join(popup_lines) if popup_lines else escape_js_string(label)
            )

            # JavaScript for showing/hiding popup (absolute positioning relative to container)
            # Account for container scroll position and use viewport for bounds checking
            show_popup_js = (
                f"var p=document.getElementById('{popup_id}');"
                f"p.innerHTML='{popup_content}';"
                f"var cont=document.getElementById('{container_id}');"
                f"var r=this.getBoundingClientRect();"
                f"var c=cont.getBoundingClientRect();"
                f"var left=r.left-c.left+cont.scrollLeft;"
                f"var top=r.bottom-c.top+cont.scrollTop+5;"
                f"p.style.left=left+'px';p.style.top=top+'px';"
                f"p.style.display='block';"
            )
            hide_popup_js = (
                f"document.getElementById('{popup_id}').style.display='none';"
            )

            parts.append(
                f'<mark class="entity-mark {css_class}" '
                f'data-color="{color}" '
                f'style="background-color: {color}; padding: 0.1em 0.2em; '
                f"border-radius: 0.2em; margin: 0 0.1em; cursor: pointer; "
                f'transition: background-color 0.2s ease, opacity 0.2s ease;" '
                f'onmouseenter="{show_popup_js}" '
                f'onmouseleave="{hide_popup_js}">'
                f"{escaped_text}</mark>"
            )

    # JavaScript functions for legend hover (highlight/dim entities)
    hover_in_js = (
        "var c=document.getElementById('{cid}');"
        "c.querySelectorAll('.entity-mark').forEach(function(m){{"
        "if(m.classList.contains('{cls}')){{m.style.opacity='1';}}"
        "else{{m.style.backgroundColor='#E5E7EB';m.style.opacity='0.6';}}"
        "}});"
    )
    hover_out_js = (
        "var c=document.getElementById('{cid}');"
        "c.querySelectorAll('.entity-mark').forEach(function(m){{"
        "m.style.backgroundColor=m.getAttribute('data-color');"
        "m.style.opacity='1';"
        "}});"
    )

    # Build legend with inline hover handlers and popups
    legend_parts = []
    seen_labels = set()
    for item in highlighted:
        label = item[1]
        if label and label not in seen_labels:
            seen_labels.add(label)
            color = color_map.get(label, "#808080")
            css_class = label_to_class(label)
            enter_js = hover_in_js.format(cid=container_id, cls=css_class)
            leave_js = hover_out_js.format(cid=container_id)

            # Build popup content for legend item (summary info, no instance-specific values)
            entity_info = label_entity_info.get(label)
            count = label_counts.get(label, 1)
            popup_lines = []
            if entity_info:
                if entity_info.get("kb_title"):
                    popup_lines.append(
                        f"<strong style=&quot;color:#333;&quot;>{escape_js_string(entity_info['kb_title'])}</strong>"
                    )
                if entity_info.get("kb_id"):
                    popup_lines.append(
                        f"<em style=&quot;color:#555;&quot;>ID: {escape_js_string(entity_info['kb_id'])}</em>"
                    )
                if entity_info.get("type"):
                    popup_lines.append(
                        f"<span style=&quot;color:#333;&quot;>Type: {escape_js_string(entity_info['type'])}</span>"
                    )
                # Show occurrence count for legend hover
                popup_lines.append(
                    f"<span style=&quot;color:#333;&quot;>Mentions: {count}</span>"
                )
                if entity_info.get("kb_description"):
                    desc = entity_info["kb_description"]
                    if len(desc) > 150:
                        desc = desc[:150] + "..."
                    popup_lines.append(
                        f"<div style=&quot;margin-top:0.3em;font-size:0.9em;color:#666;&quot;>{escape_js_string(desc)}</div>"
                    )

            popup_content = (
                "<br>".join(popup_lines) if popup_lines else escape_js_string(label)
            )

            show_popup_js = (
                f"var p=document.getElementById('{popup_id}');"
                f"p.innerHTML='{popup_content}';"
                f"var cont=document.getElementById('{container_id}');"
                f"var r=this.getBoundingClientRect();"
                f"var c=cont.getBoundingClientRect();"
                f"var left=r.left-c.left+cont.scrollLeft;"
                f"var top=r.bottom-c.top+cont.scrollTop+5;"
                f"p.style.left=left+'px';p.style.top=top+'px';"
                f"p.style.display='block';"
            )
            hide_popup_js = (
                f"document.getElementById('{popup_id}').style.display='none';"
            )

            # Combine highlight JS with popup JS
            combined_enter = enter_js + show_popup_js
            combined_leave = leave_js + hide_popup_js

            legend_parts.append(
                f'<span class="legend-item" '
                f'style="display: inline-block; margin-right: 1em; cursor: pointer;" '
                f'onmouseenter="{combined_enter}" '
                f'onmouseleave="{combined_leave}">'
                f'<span style="background-color: {color}; padding: 0.1em 0.3em; '
                f'border-radius: 0.2em; font-size: 0.85em;">{html.escape(label)}</span></span>'
            )

    legend_html = ""
    if show_legend:
        legend_html = (
            f'<div class="entity-legend" style="margin-bottom: 0.5em; line-height: 1.8;">{"".join(legend_parts)}</div>'
            if legend_parts
            else ""
        )
    text_html = f'<div class="entity-text" style="line-height: 1.6; white-space: pre-wrap;">{"".join(parts)}</div>'

    # Popup div (hidden by default, absolute positioning relative to container)
    popup_html = (
        f'<div id="{popup_id}" style="'
        f"display: none; "
        f"position: absolute; "
        f"background: white; "
        f"color: #333; "
        f"border: 1px solid #ccc; "
        f"border-radius: 6px; "
        f"padding: 0.5em 0.75em; "
        f"box-shadow: 0 2px 8px rgba(0,0,0,0.15); "
        f"max-width: 350px; "
        f"z-index: 1000; "
        f"font-size: 0.9em; "
        f"line-height: 1.4; "
        f"pointer-events: none;"
        f'"></div>'
    )

    return f'<div id="{container_id}" class="highlighted-container" style="position: relative;">{legend_html}{text_html}{popup_html}</div>'


def format_highlighted_text(
    result: Dict,
) -> Tuple[List[Tuple[str, Optional[str], Optional[Dict]]], Dict[str, str]]:
    """Convert pipeline result to highlighted format.

    Returns (highlighted_data, color_map) for use with highlighted_to_html().
    Each highlighted item is (text, label, entity_info) where entity_info contains details for popup.
    """
    text = result["text"]
    entities = result["entities"]

    if not entities:
        return [(text, None, None)], {}

    # Process entities: build labels in first-seen order
    entity_data = []
    label_order = []
    label_set = set()

    for entity in entities:
        label_type = entity.get("label", "ENT")
        if entity.get("entity_title"):
            label = f"{label_type}: {entity['entity_title']}"
        else:
            label = f"{label_type} [NOT IN KB]"
        entity_data.append((entity, label))
        if label not in label_set:
            label_set.add(label)
            label_order.append(label)

    # Build color_map: sequential palette assignment for maximum color diversity
    color_map = {}
    palette_idx = 0
    for label in label_order:
        if label == "ERROR":
            color_map[label] = ERROR_COLOR
        elif label.endswith("[NOT IN KB]"):
            color_map[label] = UNLINKED_COLOR
        else:
            color_map[label] = ENTITY_COLORS[palette_idx % len(ENTITY_COLORS)]
            palette_idx += 1

    # Sort by position for text reconstruction
    entity_data.sort(key=lambda x: x[0]["start"])

    # Build highlighted text with entity info for popups
    highlighted = []
    last_end = 0

    for entity, label in entity_data:
        if entity["start"] > last_end:
            highlighted.append((text[last_end : entity["start"]], None, None))

        instance_color = color_map.get(label)

        # Build entity info dict for popup
        entity_info = {
            "mention": entity["text"],
            "type": entity.get("label", "ENT"),
            "kb_id": entity.get("entity_id"),
            "kb_title": entity.get("entity_title"),
            "kb_description": entity.get("entity_description"),
            "display_color": instance_color,
        }

        highlighted.append((entity["text"], label, entity_info))
        last_end = entity["end"]

    if last_end < len(text):
        highlighted.append((text[last_end:], None, None))

    return highlighted, color_map


def compute_linking_stats(result: Dict) -> str:
    """Compute statistics about entity linking results."""
    entities = result.get("entities", [])
    if not entities:
        return "No entities found."

    total = len(entities)
    linked = sum(1 for e in entities if e.get("entity_title"))
    unlinked = total - linked

    stats = f"**Entity Linking Statistics**\n\n"
    stats += f"- Total entities: {total}\n"
    stats += f"- Linked to KB: {linked} ({100*linked/total:.1f}%)\n"
    stats += f"- Not in KB: {unlinked} ({100*unlinked/total:.1f}%)\n"

    return stats


def format_error_output(error_title: str, error_message: str) -> Tuple[str, str, Dict]:
    """Format an error for display in the output components.

    Returns (html_output, stats, result) for consistency with run_pipeline.
    """
    import traceback

    # Get full traceback if available
    tb = traceback.format_exc()
    if tb and tb.strip() != "NoneType: None":
        full_error = f"{error_message}\n\n**Traceback:**\n```\n{tb}\n```"
    else:
        full_error = error_message

    # Create HTML error display
    html_output = (
        f'<div style="color: {ERROR_COLOR}; padding: 1em; '
        f"border: 1px solid {ERROR_COLOR}; border-radius: 6px; "
        f'background-color: #FEF2F2;">'
        f"<strong>Error: {error_title}</strong></div>"
    )
    stats = f"**Error**\n\n{full_error}"
    result = {"error": error_title, "details": error_message}

    return gr.update(value=html_output), stats, result


def _run_with_heartbeat(fn, progress_fn, initial_progress, initial_desc):
    """Run fn(report) in a background thread, sending progress heartbeats.

    Blocks until fn completes.  Calls progress_fn every 0.5s to keep
    Gradio's SSE connection alive (without yielding, which would re-render
    output components and reset the UI timer).

    fn receives a report(progress_value, description) callback for updates.
    Returns fn's return value.  Re-raises any exception from fn.
    """
    import queue as _q

    q = _q.Queue()
    result_holder = []
    error_holder = []

    def report(prog, desc):
        q.put((prog, desc))

    def _worker():
        try:
            r = fn(report)
            result_holder.append(r)
        except Exception as e:
            error_holder.append(e)
        finally:
            q.put(None)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    last_progress = initial_progress
    last_desc = initial_desc
    while True:
        try:
            msg = q.get(timeout=0.5)
        except _q.Empty:
            try:
                progress_fn(last_progress, desc=last_desc)
            except Exception:
                pass
            continue
        if msg is None:
            break
        last_progress = msg[0]
        last_desc = msg[1]
        try:
            progress_fn(msg[0], desc=msg[1])
        except Exception:
            pass

    worker.join()

    if error_holder:
        raise error_holder[0]
    return result_holder[0] if result_holder else None


def run_pipeline(
    text_input: str,
    file_input: Optional[gr.File],
    kb_file: Optional[gr.File],
    loader_type: str,
    ner_type: str,
    spacy_model: str,
    gliner_model: str,
    gliner_labels: str,
    gliner_threshold: float,
    labels_from_kb: bool,
    simple_min_len: int,
    cand_type: str,
    cand_embedding_model: str,
    cand_top_k: int,
    cand_use_context: bool,
    cand_api_base_url: str,
    cand_api_key: str,
    reranker_type: str,
    reranker_embedding_model: str,
    reranker_cross_encoder_model: str,
    reranker_api_url: str,
    reranker_api_port: int,
    reranker_top_k: int,
    reranker_gpu_mem_gb: float,
    reranker_max_model_len: int,
    disambig_type: str,
    llm_model: str,
    thinking: bool,
    none_candidate: bool,
    disambig_gpu_mem_gb: float,
    disambig_max_model_len: int,
    disambig_api_base_url: str,
    disambig_api_key: str,
    kb_type: str,
    progress=gr.Progress(),
):
    """Run LELA with selected configuration.

    This is a generator function that yields (html_output, stats, result) tuples.
    Yielding allows Gradio to check for cancellation between steps.
    """
    import sys

    logger = logging.getLogger(__name__)
    logger.info(f"=== run_pipeline ENTERED (run #{_run_counter}) ===")
    sys.stderr.flush()

    # Clear cancellation flag at the start of a new run
    _cancel_event.clear()

    # Note: We intentionally don't clear vLLM instances here - they should be
    # reused across runs to avoid expensive reinitialization and resource leaks.
    # Only general garbage collection is performed.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    no_vis_change = gr.update()
    no_btn_change = gr.update()
    no_mode_change = gr.update()

    if not kb_file:
        from lela.knowledge_bases.yago_downloader import ensure_yago_kb

        kb_path = _run_with_heartbeat(
            lambda report: ensure_yago_kb(),
            progress,
            0.05,
            "Resolving knowledge base...",
        )
    else:
        kb_path = kb_file.name

    if not text_input and not file_input:
        yield (
            *format_error_output(
                "Missing Input", "Please provide either text input or upload a file."
            ),
            no_vis_change, no_btn_change, no_mode_change,
        )
        return

    # Check for cancellation
    if _cancel_event.is_set():
        logger.info("Pipeline cancelled before configuration")
        yield gr.update(value=""), "*Pipeline cancelled.*", {}, no_vis_change, no_btn_change, no_mode_change
        return

    progress(0.1, desc="Building pipeline configuration...")
    yield gr.update(value=""), "*Building configuration...*", {}, no_vis_change, no_btn_change, no_mode_change

    # Build NER params based on type
    ner_params = {}
    if ner_type == "spacy":
        ner_params["model"] = spacy_model
    elif ner_type == "gliner":
        ner_params["model_name"] = gliner_model
        ner_params["threshold"] = gliner_threshold
        ner_params["estimated_vram_gb"] = DEFAULT_GLINER_VRAM_GB
        if gliner_labels:
            ner_params["labels"] = [l.strip() for l in gliner_labels.split(",")]
        if labels_from_kb:
            ner_params["labels_from_kb"] = True
    elif ner_type == "regex":
        ner_params["min_len"] = simple_min_len

    # Build candidate params
    cand_params = {"top_k": cand_top_k}
    if cand_type == "dense":
        cand_params["use_context"] = cand_use_context
        cand_params["model_name"] = cand_embedding_model
        cand_params["estimated_vram_gb"] = get_model_vram_gb(cand_embedding_model)
    elif cand_type == "openai_api_dense":
        cand_params["use_context"] = cand_use_context
        cand_params["model_name"] = cand_embedding_model
        cand_params["base_url"] = cand_api_base_url
        cand_params["api_key"] = cand_api_key

    # Build reranker params
    reranker_params = {"top_k": reranker_top_k}
    if reranker_type in ("embedder_transformers", "embedder_vllm"):
        reranker_params["model_name"] = reranker_embedding_model
    if reranker_type == "cross_encoder_vllm":
        reranker_params["model_name"] = reranker_cross_encoder_model
    if reranker_type == "cross_encoder":
        reranker_params["model_name"] = reranker_cross_encoder_model
        reranker_params["estimated_vram_gb"] = get_model_vram_gb(reranker_cross_encoder_model)
    if reranker_type == "vllm_api_client":
        reranker_params["base_url"] = reranker_api_url
        reranker_params["port"] = reranker_api_port
    if reranker_type in ("embedder_vllm", "cross_encoder_vllm"):
        reranker_params["gpu_memory_gb"] = reranker_gpu_mem_gb
        reranker_params["max_model_len"] = int(reranker_max_model_len)
    if reranker_type == "embedder_transformers":
        reranker_params["estimated_vram_gb"] = get_model_vram_gb(reranker_embedding_model)

    # Build disambiguator params
    disambig_params = {}
    if disambig_type in ("vllm", "transformers"):
        disambig_params["model_name"] = llm_model
        disambig_params["enable_thinking"] = thinking
        disambig_params["add_none_candidate"] = none_candidate
    if disambig_type == "vllm":
        disambig_params["gpu_memory_gb"] = disambig_gpu_mem_gb
        disambig_params["max_model_len"] = int(disambig_max_model_len)
    if disambig_type == "transformers":
        disambig_params["estimated_vram_gb"] = get_model_vram_gb(llm_model)
    if disambig_type == "openai_api":
        disambig_params["base_url"] = disambig_api_base_url
        disambig_params["api_key"] = disambig_api_key or None

    # Override components if candidate_generator is "none" for NER-only pipeline
    if cand_type == "none":
        candidate_generator_config = {"name": "none", "params": {}}
        reranker_config = {"name": "none", "params": {}}
        disambiguator_config = None
    else:
        candidate_generator_config = {"name": cand_type, "params": cand_params}
        reranker_config = (
            {"name": reranker_type, "params": reranker_params}
            if reranker_type != "none"
            else {"name": "none", "params": {}}
        )
        disambiguator_config = (
            {"name": disambig_type, "params": disambig_params}
            if disambig_type != "none"
            else None
        )

    # Resolve labels_from_kb: extract entity types from KB and use as NER labels
    if ner_params.get("labels_from_kb"):
        try:
            from lela.knowledge_bases.jsonl import JSONLKnowledgeBase
            kb = JSONLKnowledgeBase(kb_path)
            kb_types = kb.get_entity_types()
            if kb_types:
                ner_params["labels"] = kb_types
        except Exception as e:
            logger.warning(f"Failed to extract entity types from KB: {e}")
        del ner_params["labels_from_kb"]

    config_dict = {
        "loader": {"name": loader_type, "params": {}},
        "ner": {"name": ner_type, "params": ner_params},
        "candidate_generator": candidate_generator_config,
        "reranker": reranker_config,
        "disambiguator": disambiguator_config,
        "knowledge_base": {"name": kb_type, "params": {"path": kb_path}},
        "cache_dir": ".ner_cache",
        "batch_size": 1,
    }

    # Check for cancellation
    if _cancel_event.is_set():
        logger.info("Pipeline cancelled before initialization")
        yield gr.update(value=""), "*Pipeline cancelled.*", {}, no_vis_change, no_btn_change, no_mode_change
        return

    progress(0.15, desc="Initializing pipeline...")
    yield gr.update(value=""), "*Initializing pipeline...*", {}, no_vis_change, no_btn_change, no_mode_change

    try:
        def _init_pipeline(report):
            def init_progress_callback(local_progress: float, description: str):
                report(0.15 + local_progress * 0.2, description)
                if _cancel_event.is_set():
                    raise InterruptedError("Pipeline cancelled by user")

            return Lela(
                config_dict,
                progress_callback=init_progress_callback,
                cancel_event=_cancel_event,
            )

        pipeline = _run_with_heartbeat(
            _init_pipeline,
            progress,
            0.15,
            "Initializing pipeline...",
        )
    except InterruptedError:
        logger.info("Pipeline cancelled during initialization")
        yield gr.update(value=""), "*Pipeline cancelled.*", {}, no_vis_change, no_btn_change, no_mode_change
        return
    except Exception as e:
        logger.exception("Pipeline initialization failed")
        yield (
            *format_error_output("Pipeline Initialization Failed", str(e)),
            no_vis_change, no_btn_change, no_mode_change,
        )
        return

    # Check for cancellation
    if _cancel_event.is_set():
        logger.info("Pipeline cancelled after initialization")
        yield gr.update(value=""), "*Pipeline cancelled.*", {}, no_vis_change, no_btn_change, no_mode_change
        return

    progress(0.4, desc="Loading document...")
    yield gr.update(value=""), "*Loading document...*", {}, no_vis_change, no_btn_change, no_mode_change

    try:
        if file_input:
            input_path = file_input.name
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                f.write(text_input)
                input_path = f.name

        # Load document (in background thread to keep SSE alive for large files)
        docs = _run_with_heartbeat(
            lambda report: list(pipeline.loader.load(input_path)),
            progress,
            0.4,
            "Loading document...",
        )

        if not file_input:
            os.unlink(input_path)

        if not docs:
            yield (
                *format_error_output(
                    "No Documents Loaded",
                    "The input file was empty or could not be parsed.",
                ),
                no_vis_change, no_btn_change, no_mode_change,
            )
            return

        doc = docs[0]

        # Check for cancellation
        if _cancel_event.is_set():
            logger.info("Pipeline cancelled before processing")
            yield gr.update(value=""), "*Pipeline cancelled.*", {}, no_vis_change, no_btn_change, no_mode_change
            return

        progress(0.45, desc="Processing document...")
        yield gr.update(value=""), "*Processing document...*", {}, no_vis_change, no_btn_change, no_mode_change

        def _run_processing(report):
            def progress_callback(local_progress: float, description: str):
                report(0.45 + local_progress * 0.4, description)
                if _cancel_event.is_set():
                    raise InterruptedError("Pipeline cancelled by user")

            return pipeline.process_document_with_progress(
                doc,
                progress_callback=progress_callback,
                base_progress=0.0,
                progress_range=1.0,
            )

        result = _run_with_heartbeat(
            _run_processing,
            progress,
            0.45,
            "Processing document...",
        )

    except InterruptedError:
        logger.info("Pipeline cancelled during processing")
        yield gr.update(value=""), "*Pipeline cancelled.*", {}, no_vis_change, no_btn_change, no_mode_change
        return
    except Exception as e:
        yield (*format_error_output("Pipeline Execution Failed", str(e)), no_vis_change, no_btn_change, no_mode_change)
        return

    logger.info("Pipeline processing complete, formatting output...")
    sys.stderr.flush()
    progress(0.9, desc="Formatting output...")

    logger.info("Calling format_highlighted_text...")
    sys.stderr.flush()
    highlighted, color_map = format_highlighted_text(result)
    logger.info(f"format_highlighted_text done, got {len(highlighted)} segments")
    sys.stderr.flush()

    # Convert to HTML for the gr.HTML component (no legend for inline preview)
    html_output = highlighted_to_html(highlighted, color_map, show_legend=False)

    logger.info("Calling compute_linking_stats...")
    sys.stderr.flush()
    stats = compute_linking_stats(result)
    logger.info("compute_linking_stats done")
    sys.stderr.flush()

    logger.info("Calling progress(1.0, Done!)...")
    sys.stderr.flush()
    try:
        progress(1.0, desc="Done!")
        logger.info("progress(1.0) returned successfully")
    except Exception as e:
        logger.error(f"progress(1.0) raised exception: {e}")
    sys.stderr.flush()

    # Ensure all GPU operations are complete before returning
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        logger.info("CUDA synchronized")
        sys.stderr.flush()

    logger.info(
        f"=== run_pipeline RETURNING (run #{_run_counter}, {len(result.get('entities', []))} entities) ==="
    )
    sys.stderr.flush()
    # Final yield: show preview with results, hide text_input, show edit_btn, switch to preview mode
    yield gr.update(value=html_output, visible=True), stats, result, gr.update(visible=False), gr.update(visible=True), "preview"


def update_ner_params(ner_choice: str):
    """Show/hide NER-specific parameters based on selection."""
    show_gliner = ner_choice == "gliner"
    vram_text = f"~{DEFAULT_GLINER_VRAM_GB:.1f} GB VRAM" if show_gliner else ""
    return {
        spacy_params: gr.update(visible=(ner_choice == "spacy")),
        gliner_params: gr.update(visible=show_gliner),
        simple_params: gr.update(visible=(ner_choice == "regex")),
        ner_vram_info: gr.update(visible=show_gliner, value=vram_text),
    }


def _format_vram_info(model_id: str) -> str:
    """Format VRAM info text for a model."""
    vram = get_model_vram_gb(model_id)
    return f"~{vram:.1f} GB VRAM"


def update_cand_params(cand_choice: str):
    """Show/hide candidate-specific parameters based on selection."""
    if cand_choice == "none":
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    show_context = cand_choice in ("dense", "openai_api_dense")
    show_embedding_model = cand_choice in ("dense", "openai_api_dense")
    show_openai_api_dense_params = cand_choice == "openai_api_dense"
    show_vram_info = cand_choice == "dense"

    return (
        gr.update(visible=show_embedding_model),
        gr.update(visible=show_context),
        gr.update(visible=show_openai_api_dense_params),
        gr.update(
            visible=show_vram_info,
            value=_format_vram_info("Qwen/Qwen3-Embedding-4B") if show_vram_info else "",
        ),
    )


def update_reranker_params(reranker_choice: str):
    """Show/hide reranker-specific parameters based on selection."""
    show_cross_encoder_model = reranker_choice in (
        "cross_encoder",
        "cross_encoder_vllm",
    )
    show_embedding_model = reranker_choice in (
        "embedder_transformers",
        "embedder_vllm",
    )
    show_vllm_api_client = reranker_choice in ("vllm_api_client", "llama_server")
    # Slider only for vLLM backends
    show_gpu_mem_slider = reranker_choice in (
        "embedder_vllm",
        "cross_encoder_vllm",
    )
    show_context_len_slider = reranker_choice in (
        "embedder_vllm",
        "cross_encoder_vllm",
    )
    # VRAM info for transformers backends
    show_vram_info = reranker_choice in (
        "cross_encoder",
        "embedder_transformers",
    )
    if reranker_choice == "cross_encoder":
        vram_text = _format_vram_info("tomaarsen/Qwen3-Reranker-4B-seq-cls")
    elif reranker_choice == "embedder_transformers":
        vram_text = _format_vram_info("Qwen/Qwen3-Embedding-4B")
    else:
        vram_text = ""
    # Use different model lists for vLLM vs transformers cross-encoder
    if reranker_choice == "cross_encoder_vllm":
        ce_choices = [(m[1], m[0]) for m in VLLM_RERANKER_MODEL_CHOICES]
        ce_default = DEFAULT_VLLM_RERANKER_MODEL
    else:
        ce_choices = [(m[1], m[0]) for m in CROSS_ENCODER_MODEL_CHOICES]
        ce_default = "tomaarsen/Qwen3-Reranker-4B-seq-cls"
    return (
        gr.update(
            visible=show_cross_encoder_model, choices=ce_choices, value=ce_default
        ),
        gr.update(visible=show_embedding_model),
        gr.update(visible=show_vllm_api_client),
        gr.update(visible=show_gpu_mem_slider),
        gr.update(visible=show_context_len_slider),
        gr.update(visible=show_vram_info, value=vram_text),
    )


def update_disambig_params(disambig_choice: str):
    """Show/hide disambiguator-specific parameters based on selection."""
    show_llm = disambig_choice in ("vllm", "transformers")
    show_gpu_mem_slider = disambig_choice == "vllm"
    show_context_len_slider = disambig_choice == "vllm"
    show_vram_info = disambig_choice == "transformers"
    show_openai_api = disambig_choice == "openai_api"
    vram_text = _format_vram_info("Qwen/Qwen3-4B") if show_vram_info else ""
    return (
        gr.update(visible=show_llm),
        gr.update(visible=show_llm),
        gr.update(visible=show_gpu_mem_slider),
        gr.update(visible=show_context_len_slider),
        gr.update(visible=show_openai_api),
        gr.update(visible=show_vram_info, value=vram_text),
    )


def update_loader_from_file(file: Optional[gr.File]):
    """Auto-detect loader type from file extension."""
    if not file:
        return gr.update()

    ext = Path(file.name).suffix.lower()
    loader_map = {
        ".txt": "text",
        ".pdf": "pdf",
        ".docx": "docx",
        ".html": "html",
        ".htm": "html",
    }

    if ext in loader_map:
        return gr.update(value=loader_map[ext])
    return gr.update()


def compute_memory_estimate() -> str:
    """Show GPU info, cached models, and cached KBs."""
    from lela.llm_pool import get_cached_models_info
    from lela.knowledge_bases.jsonl import get_kb_cache_info

    try:
        resources = get_system_resources()

        lines = []
        if resources.gpu_available:
            lines.append(f"**GPU:** {resources.gpu_name}")
            lines.append(
                f"**VRAM:** {resources.gpu_vram_free_gb:.1f}GB free / {resources.gpu_vram_total_gb:.1f}GB total"
            )
        else:
            lines.append("**GPU:** Not available")

        # Show cached models
        cached_info = get_cached_models_info()
        cached_models = []
        for st_info in cached_info.get("sentence_transformers", []):
            model_name = st_info["key"].split(":")[0].split("/")[-1]
            status = "in use" if st_info["in_use"] else "cached"
            cached_models.append(f"{model_name} ({status})")
        for vllm_info in cached_info.get("vllm", []):
            model_name = vllm_info["key"].split(":")[0].split("/")[-1]
            status = "in use" if vllm_info["in_use"] else "cached"
            cached_models.append(f"{model_name} ({status})")
        for gen_info in cached_info.get("generic", []):
            model_name = gen_info["key"].split(":")[-1].split("/")[-1]
            status = "in use" if gen_info["in_use"] else "cached"
            cached_models.append(f"{model_name} ({status})")

        if cached_models:
            lines.append(f"**Cached models:** {', '.join(cached_models)}")

        # Show cached KBs
        kb_cache_info = get_kb_cache_info()
        if kb_cache_info:
            kb_names = [
                f"{Path(kb['path']).name} ({kb['entity_count']:,} entities)"
                for kb in kb_cache_info
            ]
            lines.append(f"**Cached KBs:** {', '.join(kb_names)}")

        return "\n".join(lines)

    except Exception as e:
        return f"*Could not estimate memory: {e}*"


_run_counter = 0


def clear_outputs_for_new_run():
    """Clear outputs and log when a new run starts."""
    global _run_counter
    _run_counter += 1
    # Clear cancellation flag for new run
    _cancel_event.clear()
    logger = logging.getLogger(__name__)
    logger.info(f"=== BUTTON CLICKED - Starting run #{_run_counter} ===")
    import sys

    sys.stderr.flush()
    # Return: preview_html, stats, json, run_btn, cancel_btn,
    #         text_input, edit_btn, view_mode, upload_btn, unload_btn, kb_file
    return (
        gr.update(value="", visible=True),  # preview_html: clear + show (progress bar appears here)
        "*Processing...*",
        None,
        gr.update(visible=False),  # run_btn hidden
        gr.update(visible=True),  # cancel_btn shown
        gr.update(visible=False),  # text_input hidden
        gr.update(value="✎ Edit", visible=False),  # edit_btn hidden until pipeline finishes
        "preview",  # view_mode → preview
        gr.update(interactive=False),  # upload_btn disabled
        gr.update(interactive=False),  # unload_btn disabled
        gr.update(interactive=False),  # kb_file disabled
    )


def restore_buttons_after_run():
    """Restore button visibility and re-enable controls after pipeline completes or is cancelled."""
    return (
        gr.update(visible=True),  # run_btn shown
        gr.update(visible=False, value="Cancel", interactive=True),  # cancel_btn hidden
        gr.update(interactive=True),  # upload_btn re-enabled
        gr.update(interactive=True),  # unload_btn re-enabled
        gr.update(interactive=True),  # kb_file re-enabled
    )


def start_cancellation():
    """Called when cancel button is clicked. Set flag and show cancelling state."""
    logger = logging.getLogger(__name__)
    logger.info("=== CANCEL BUTTON CLICKED ===")
    # Set the cancellation flag - pipeline will check this and stop
    _cancel_event.set()
    # Update button to show "Cancelling..." with loading state (non-interactive)
    return gr.update(value="Cancelling...", interactive=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LELA Gradio UI")
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to run the server on"
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public share link"
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log)
    logger = logging.getLogger(__name__)

    # Silence noisy loggers
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numba.core").setLevel(logging.WARNING)

    # Log vLLM availability status
    vllm_ok = _is_vllm_usable()
    logger.info(f"vLLM disambiguator available: {vllm_ok}")

    # Detect GPU total VRAM for GB sliders
    _resources = get_system_resources()
    gpu_total_gb = round(max(_resources.gpu_vram_total_gb, 1.0), 2)  # fallback minimum 1 GB
    logger.info(f"GPU total VRAM: {gpu_total_gb:.1f} GB")

    components = get_available_components()

    # Custom CSS for cleaner design
    custom_css = """
    #title-bar .html-container {
        padding-left: 0 !important;
    }
    .output-section {
        min-height: 300px;
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid var(--border-color-primary);
        border-radius: var(--radius-lg);
        padding: 1em;
    }
    #main-result-output {
        padding: 0 !important;
        margin: 0 !important;
    }
    #main-result-output > div {
        padding: 0 !important;
        margin: 0 !important;
    }
    .input-header-row {
        align-items: center !important;
        margin-bottom: 0 !important;
        gap: 0.5rem !important;
        flex-wrap: nowrap !important;
        justify-content: flex-start !important;
    }
    .input-header-row > * {
        flex: 0 0 auto !important;
        width: auto !important;
        max-width: fit-content !important;
    }
    .input-title {
        flex: 0 0 auto !important;
        width: auto !important;
    }
    .input-title h3 {
        margin: 0 !important;
    }
    .action-row {
        align-items: flex-start !important;
        gap: 0.5rem !important;
        flex-wrap: nowrap !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.25rem !important;
    }
    #stats-accordion {
        flex: 1 1 auto !important;
        min-width: 0 !important;
    }
    #run-btn {
        max-width: 400px !important;
    }
    #cancel-btn {
        max-width: 400px !important;
    }
    #edit-preview-btn {
        max-width: 100px !important;
    }
    #upload-file-btn {
        max-width: 120px !important;
    }
    .config-row {
        align-items: flex-start !important;
    }
    .pipeline-col-header p {
        font-weight: 600 !important;
        font-size: 1.1em !important;
        margin-bottom: 0.3em !important;
    }
    .kb-row {
        align-items: center !important;
        gap: 0.75rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.25rem !important;
    }
    .kb-filename {
        font-size: 0.85em !important;
        opacity: 0.8;
    }
    .kb-filename p { margin: 0 !important; }
    .gpu-info {
        font-size: 0.85em !important;
        opacity: 0.7;
    }
    .gpu-info p { margin: 0 !important; }
    #config-accordion {
        margin-top: 0.25rem !important;
        position: relative !important;
    }
    #export-btn {
        position: absolute !important;
        margin-top: -32px !important;
        left: 100px !important;
        z-index: 10 !important;
        max-width: 80px !important;
        font-size: 13px !important;
    }
    #unload-btn {
        font-size: 13px !important;
        max-width: 160px !important;
        background: #fef2f2 !important;
        border-color: #fca5a5 !important;
        color: #b91c1c !important;
    }
    #unload-btn:hover {
        background: #fee2e2 !important;
        border-color: #f87171 !important;
    }
    .footer-logos {
        margin-top: 1.5rem !important;
        margin-bottom: 0 !important;
    }
    """

    # JavaScript to cancel pipeline when tab is closed + drag-and-drop support
    custom_head = """
    <script>
    (function() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', init);
        } else {
            init();
        }

        function init() {
            setupPageCloseHandler();
            setupTextAreaDrop();
        }

        function setupPageCloseHandler() {
            console.log("LELA: Page close handler initialized");

            function findCancelButton() {
                const buttons = document.querySelectorAll('button');
                for (const btn of buttons) {
                    const text = btn.textContent.trim();
                    if (text === 'Cancel' && btn.offsetParent !== null) {
                        const style = window.getComputedStyle(btn);
                        if (style.display !== 'none' && style.visibility !== 'hidden') {
                            return btn;
                        }
                    }
                }
                return null;
            }

            window.addEventListener('pagehide', function() {
                const cancelBtn = findCancelButton();
                if (cancelBtn) {
                    cancelBtn.click();
                    console.log("LELA: Triggered cancellation on pagehide");
                }
            }, { capture: true });

            window.addEventListener('beforeunload', function() {
                const cancelBtn = findCancelButton();
                if (cancelBtn) {
                    cancelBtn.click();
                    console.log("LELA: Triggered cancellation on beforeunload");
                }
            }, { capture: true });
        }

        function setupTextAreaDrop() {
            // Use event delegation on document so listeners survive Gradio DOM rebuilds
            document.addEventListener('dragover', function(e) {
                var el = e.target.closest('#main-text-input');
                if (!el) return;
                e.preventDefault();
                e.stopPropagation();
                el.style.outline = '2px dashed #2563EB';
                el.style.outlineOffset = '-2px';
            });
            document.addEventListener('dragleave', function(e) {
                var el = e.target.closest('#main-text-input');
                if (!el) return;
                e.preventDefault();
                el.style.outline = '';
                el.style.outlineOffset = '';
            });
            document.addEventListener('drop', function(e) {
                var el = e.target.closest('#main-text-input');
                if (!el) return;
                e.preventDefault();
                e.stopPropagation();
                el.style.outline = '';
                el.style.outlineOffset = '';
                var file = e.dataTransfer.files[0];
                if (!file) return;

                if (file.name.endsWith('.txt')) {
                    var reader = new FileReader();
                    reader.onload = function() {
                        var ta = el.querySelector('textarea');
                        if (ta) {
                            var nativeSet = Object.getOwnPropertyDescriptor(
                                window.HTMLTextAreaElement.prototype, "value").set;
                            nativeSet.call(ta, reader.result);
                            ta.dispatchEvent(new Event('input', { bubbles: true }));
                        }
                    };
                    reader.readAsText(file);
                } else {
                    var uploadInput = document.querySelector('#upload-file-btn input[type="file"]');
                    if (uploadInput) {
                        var dt = new DataTransfer();
                        dt.items.add(file);
                        uploadInput.files = dt.files;
                        uploadInput.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }
            });
            console.log("LELA: Drag-and-drop initialized on text area");
        }
    })();
    </script>
    """

    with gr.Blocks(title="LELA", fill_height=True, head=custom_head, css=custom_css) as demo:
        gr.HTML(TITLE_HTML, elem_id="title-bar")

        # --- INPUT SECTION ---
        with gr.Row(elem_classes=["input-header-row"]):
            gr.Markdown("### Input Text", elem_classes=["input-title"])
            upload_btn = gr.UploadButton(
                "⇪ Upload File",
                size="sm",
                variant="secondary",
                scale=0,
                min_width=100,
                file_types=[".txt", ".pdf", ".docx", ".html"],
                elem_id="upload-file-btn",
            )
            edit_btn = gr.Button(
                "◉ Preview",
                size="sm",
                variant="secondary",
                scale=0,
                min_width=90,
                elem_id="edit-preview-btn",
                visible=False,
            )
        text_input = gr.Textbox(
            show_label=False,
            placeholder="Enter or paste text, or drag & drop a file...",
            lines=10,
            elem_id="main-text-input",
            value="Albert Einstein was born in Germany. Marie Curie was a pioneering scientist.",
        )
        # Result mode: HTML output (hidden by default)
        preview_html = gr.HTML(
            visible=False,
            elem_id="main-result-output",
            elem_classes=["output-section"],
        )

        # --- ACTION ROW: Run/Cancel + Stats/JSON ---
        with gr.Row(elem_classes=["action-row"]):
            run_btn = gr.Button(
                "Run Pipeline",
                variant="primary",
                scale=0,
                min_width=400,
                elem_id="run-btn",
            )
            cancel_btn = gr.Button(
                "Cancel",
                variant="stop",
                visible=False,
                scale=0,
                min_width=400,
                elem_id="cancel-btn",
            )
            with gr.Accordion("Stats / JSON", open=False, elem_id="stats-accordion"):
                with gr.Tabs():
                    with gr.Tab("Stats"):
                        stats_output = gr.Markdown(
                            "*Run the pipeline to see statistics.*"
                        )
                    with gr.Tab("JSON"):
                        json_output = gr.JSON(label="Pipeline Output")

        # Hidden file component for internal file tracking (used by pipeline)
        file_input = gr.File(visible=False)
        # State: "edit" or "preview" — tracks which mode the text area is in
        view_mode = gr.State("edit")

        # --- CONFIGURATION SECTION (collapsible) ---
        with gr.Accordion("Configuration", open=True, elem_id="config-accordion"):
            export_btn = gr.Button(
                "⎘ Export",
                size="sm",
                variant="secondary",
                scale=0,
                min_width=80,
                elem_id="export-btn",
            )
            export_holder = gr.Textbox(visible=False)
            # Knowledge base upload + Unload button
            with gr.Row(elem_classes=["kb-row"]):
                kb_upload_btn = gr.UploadButton(
                    "▤ Upload Knowledge Base",
                    file_types=[".jsonl"],
                    size="sm",
                    variant="secondary",
                    scale=0,
                    min_width=180,
                    elem_id="kb-upload-btn",
                )
                kb_filename = gr.Markdown(
                    value="*Using default: YAGO 4.5*",
                    elem_classes=["kb-filename"],
                )
                unload_btn = gr.Button(
                    "Unload All Models",
                    variant="secondary",
                    size="sm",
                    scale=0,
                    min_width=140,
                    elem_id="unload-btn",
                )
            kb_file = gr.File(visible=False)

            with gr.Row(equal_height=False, elem_classes=["config-row"]):
                # NER Column
                with gr.Column(scale=1, min_width=180):
                    gr.Markdown("**NER**", elem_classes=["pipeline-col-header"])
                    ner_type = gr.Dropdown(
                        choices=components["ner"],
                        value="regex",
                        label="Model",
                        container=False,

                    )
                    with gr.Group(visible=False) as spacy_params:
                        spacy_model = gr.Textbox(
                            label="SpaCy Model",
                            value="en_core_web_sm",
                            scale=1,
                        )
                    with gr.Group(visible=False) as gliner_params:
                        gliner_model = gr.Textbox(
                            label="GLiNER Model",
                            value=DEFAULT_GLINER_MODEL,
                        )
                        gliner_labels = gr.Textbox(
                            label="Labels (comma-sep)",
                            value="person, organization, location",
                        )
                        gliner_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.5,
                            step=0.05,
                            label="Threshold",
                        )
                        labels_from_kb = gr.Checkbox(
                            label="Use KB entity types as labels",
                            value=False,
                        )
                    with gr.Group(visible=True) as simple_params:
                        simple_min_len = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Min Length",
                        )
                    ner_vram_info = gr.Markdown(
                        visible=False,
                    )

                # Candidates Column
                with gr.Column(scale=1, min_width=180):
                    gr.Markdown("**Candidates**", elem_classes=["pipeline-col-header"])
                    cand_type = gr.Dropdown(
                        choices=components["candidates"],
                        value="fuzzy",
                        label="Generator",
                        container=False,

                    )
                    # Embedding model selection for dense candidates
                    embedding_model_choices = [
                        (m[1], m[0]) for m in EMBEDDING_MODEL_CHOICES
                    ]
                    cand_embedding_model = gr.Dropdown(
                        choices=embedding_model_choices,
                        value="Qwen/Qwen3-Embedding-4B",
                        label="Embedding Model",
                        visible=False,
                    )
                    with gr.Group(
                        visible=False
                    ) as openai_api_dense_cand_params:
                        cand_api_base_url = gr.Textbox(
                            label="Cand. OpenAI API Base URL",
                            value="http://localhost:8000/v1",
                        )
                        cand_api_key = gr.Textbox(
                            label="Cand. OpenAI API Key",
                            value="",
                            type="password",
                        )
                    cand_top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=64,
                        step=1,
                        label="Top K",
                    )
                    cand_use_context = gr.Checkbox(
                        label="Use Context",
                        value=False,
                        visible=False,
                    )
                    cand_vram_info = gr.Markdown(
                        visible=False,
                    )

                # Reranking Column
                with gr.Column(scale=1, min_width=180):
                    gr.Markdown("**Reranking**", elem_classes=["pipeline-col-header"])
                    reranker_type = gr.Dropdown(
                        choices=components["rerankers"],
                        value="none",
                        label="Reranker",
                        container=False,

                    )
                    # Cross-encoder model selection
                    cross_encoder_model_choices = [
                        (m[1], m[0]) for m in CROSS_ENCODER_MODEL_CHOICES
                    ]
                    reranker_cross_encoder_model = gr.Dropdown(
                        choices=cross_encoder_model_choices,
                        value="tomaarsen/Qwen3-Reranker-4B-seq-cls",
                        label="Cross-Encoder Model",
                        visible=False,
                    )
                    # Embedding model selection for embedder reranker
                    reranker_embedding_model = gr.Dropdown(
                        choices=embedding_model_choices,
                        value="Qwen/Qwen3-Embedding-4B",
                        label="Embedding Model",
                        visible=False,
                    )
                    with gr.Group(visible=False) as vllm_api_client_params:
                        reranker_api_url = gr.Textbox(
                            label="Reranker API URL",
                            value="http://localhost",
                        )
                        reranker_api_port = gr.Number(
                            label="Reranker API Port",
                            value=8000,
                        )
                    reranker_top_k = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=10,
                        step=1,
                        label="Top K",
                    )
                    reranker_gpu_mem_gb = gr.Slider(
                        minimum=0.5,
                        maximum=gpu_total_gb,
                        value=min(10.0, gpu_total_gb),
                        step=0.5,
                        label="GPU Memory (GB)",
                        visible=False,
                    )
                    reranker_max_model_len = gr.Slider(
                        minimum=MIN_VLLM_CONTEXT_LEN,
                        maximum=MAX_VLLM_CONTEXT_LEN,
                        value=min(DEFAULT_WEB_VLLM_CONTEXT_LEN, MAX_VLLM_CONTEXT_LEN),
                        step=VLLM_CONTEXT_LEN_STEP,
                        label="Context Length (max_model_len)",
                        visible=False,
                    )
                    reranker_vram_info = gr.Markdown(
                        visible=False,
                    )

                # Disambiguation Column
                with gr.Column(scale=1, min_width=180):
                    gr.Markdown("**Disambiguation**", elem_classes=["pipeline-col-header"])
                    disambig_type = gr.Dropdown(
                        choices=components["disambiguators"],
                        value="first",
                        label="Method",
                        container=False,

                    )
                    # LLM model selection for vLLM-based disambiguators
                    llm_model_choices = [(m[1], m[0]) for m in LLM_MODEL_CHOICES]
                    llm_model = gr.Dropdown(
                        choices=llm_model_choices,
                        value="Qwen/Qwen3-4B",
                        label="LLM Model",
                        visible=False,
                    )
                    with gr.Group(visible=False) as common_disambig_params:
                        thinking = gr.Checkbox(
                            label="Reasoning",
                            value=True,
                        )
                        none_candidate = gr.Checkbox(
                            label="'None' Candidate",
                            value=True,
                        )
                    disambig_gpu_mem_gb = gr.Slider(
                        minimum=0.5,
                        maximum=gpu_total_gb,
                        value=min(10.0, gpu_total_gb),
                        step=0.5,
                        label="GPU Memory (GB)",
                        visible=False,
                    )
                    disambig_max_model_len = gr.Slider(
                        minimum=MIN_VLLM_CONTEXT_LEN,
                        maximum=MAX_VLLM_CONTEXT_LEN,
                        value=min(DEFAULT_WEB_VLLM_CONTEXT_LEN, MAX_VLLM_CONTEXT_LEN),
                        step=VLLM_CONTEXT_LEN_STEP,
                        label="Context Length (max_model_len)",
                        visible=False,
                    )
                    disambig_vram_info = gr.Markdown(
                        visible=False,
                    )
                    with gr.Group(visible=False) as openai_api_disambig_params:
                        disambig_api_base_url = gr.Textbox(
                            label="OpenAI API Base URL",
                            value="http://localhost:8000/v1",
                        )
                        disambig_api_key = gr.Textbox(
                            label="OpenAI API Key",
                            value="",
                            type="password",
                        )

            memory_estimate_display = gr.Markdown(
                value="*Detecting GPU...*",
                elem_classes=["gpu-info"],
                elem_id="gpu-info-display",
            )

        # Hidden loader and KB type (auto-detected)
        loader_type = gr.Dropdown(
            choices=components["loaders"],
            value="text",
            visible=False,
        )
        kb_type = gr.Dropdown(
            choices=components["knowledge_bases"],
            value="jsonl",
            visible=False,
        )

        gr.Markdown(LOGO, elem_classes=["footer-logos"])


        # --- EVENT HANDLERS ---

        # Export config as JSON file
        def handle_export(
            loader_t, ner_t, spacy_m, gliner_m, gliner_l, gliner_th, lbl_from_kb,
            simple_ml, cand_t, cand_emb_m, cand_tk, cand_ctx, cand_api_url,
            cand_api_k, reranker_t, reranker_emb_m, reranker_ce_m, reranker_url,
            reranker_port, reranker_tk, reranker_gpu, reranker_ctx_len,
            disambig_t, llm_m, thinking, none_cand, disambig_gpu, disambig_ctx_len,
            disambig_url, disambig_key, kb_t, kb_f,
        ):
            import json as _json

            # NER params
            ner_params = {}
            if ner_t == "spacy":
                ner_params["model"] = spacy_m
            elif ner_t == "gliner":
                ner_params["model_name"] = gliner_m
                ner_params["threshold"] = gliner_th
                if gliner_l:
                    ner_params["labels"] = [l.strip() for l in gliner_l.split(",")]
                if lbl_from_kb:
                    ner_params["labels_from_kb"] = True
            elif ner_t == "regex":
                ner_params["min_len"] = simple_ml

            # Candidate params
            cand_params = {"top_k": cand_tk}
            if cand_t == "dense":
                cand_params["use_context"] = cand_ctx
                cand_params["model_name"] = cand_emb_m
            elif cand_t == "openai_api_dense":
                cand_params["use_context"] = cand_ctx
                cand_params["model_name"] = cand_emb_m
                cand_params["base_url"] = cand_api_url
                cand_params["api_key"] = cand_api_k

            # Reranker params
            reranker_params = {"top_k": reranker_tk}
            if reranker_t in ("embedder_transformers", "embedder_vllm"):
                reranker_params["model_name"] = reranker_emb_m
            if reranker_t in ("cross_encoder", "cross_encoder_vllm"):
                reranker_params["model_name"] = reranker_ce_m
            if reranker_t in ("vllm_api_client", "llama_server"):
                reranker_params["base_url"] = reranker_url
                reranker_params["port"] = reranker_port
            if reranker_t in ("embedder_vllm", "cross_encoder_vllm"):
                reranker_params["gpu_memory_gb"] = reranker_gpu
                reranker_params["max_model_len"] = int(reranker_ctx_len)

            # Disambiguator params
            disambig_params = {}
            if disambig_t in ("vllm", "transformers"):
                disambig_params["model_name"] = llm_m
                disambig_params["enable_thinking"] = thinking
                disambig_params["add_none_candidate"] = none_cand
            if disambig_t == "vllm":
                disambig_params["gpu_memory_gb"] = disambig_gpu
                disambig_params["max_model_len"] = int(disambig_ctx_len)
            if disambig_t == "openai_api":
                disambig_params["base_url"] = disambig_url
                disambig_params["api_key"] = disambig_key or None

            # Assemble config
            if cand_t == "none":
                cand_config = {"name": "none", "params": {}}
                reranker_config = {"name": "none", "params": {}}
                disambig_config = None
            else:
                cand_config = {"name": cand_t, "params": cand_params}
                reranker_config = (
                    {"name": reranker_t, "params": reranker_params}
                    if reranker_t != "none"
                    else {"name": "none", "params": {}}
                )
                disambig_config = (
                    {"name": disambig_t, "params": disambig_params}
                    if disambig_t != "none"
                    else None
                )

            kb_params = {}
            if kb_f and hasattr(kb_f, "name"):
                kb_params["path"] = Path(kb_f.name).name

            config_dict = {
                "loader": {"name": loader_t, "params": {}},
                "ner": {"name": ner_t, "params": ner_params},
                "candidate_generator": cand_config,
                "reranker": reranker_config,
                "disambiguator": disambig_config,
                "knowledge_base": {"name": kb_t, "params": kb_params},
                "batch_size": 1,
            }

            return _json.dumps(config_dict, indent=2)

        export_btn.click(
            fn=handle_export,
            inputs=[
                loader_type, ner_type, spacy_model, gliner_model, gliner_labels,
                gliner_threshold, labels_from_kb, simple_min_len,
                cand_type, cand_embedding_model, cand_top_k, cand_use_context,
                cand_api_base_url, cand_api_key,
                reranker_type, reranker_embedding_model, reranker_cross_encoder_model,
                reranker_api_url, reranker_api_port, reranker_top_k,
                reranker_gpu_mem_gb, reranker_max_model_len,
                disambig_type, llm_model, thinking, none_candidate,
                disambig_gpu_mem_gb, disambig_max_model_len,
                disambig_api_base_url, disambig_api_key,
                kb_type, kb_file,
            ],
            outputs=[export_holder],
        ).then(
            fn=None,
            inputs=[export_holder],
            js="""async (json_str) => {
                const blob = new Blob([json_str], {type: 'application/json'});
                if (window.showSaveFilePicker) {
                    try {
                        const handle = await window.showSaveFilePicker({
                            suggestedName: 'lela_config.json',
                            types: [{
                                description: 'JSON Config',
                                accept: {'application/json': ['.json']},
                            }],
                        });
                        const writable = await handle.createWritable();
                        await writable.write(blob);
                        await writable.close();
                        return;
                    } catch (e) {
                        if (e.name === 'AbortError') return;
                    }
                }
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'lela_config.json';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }""",
        )

        # KB upload button → populate hidden file + show filename
        def on_kb_upload(file):
            if file is None:
                return None, "*Using default: YAGO 4.5*"
            name = Path(file.name).name if hasattr(file, 'name') else str(file)
            return file, f"**{name}**"

        kb_upload_btn.upload(
            fn=on_kb_upload,
            inputs=[kb_upload_btn],
            outputs=[kb_file, kb_filename],
        )

        # Edit/Preview toggle button
        def toggle_edit_preview(current_mode, current_preview):
            if current_mode == "edit":
                # Switch to preview mode
                if current_preview:
                    html = current_preview
                else:
                    html = "<div style='color:#6B7280;padding:1em;'>Run the pipeline to see results here.</div>"
                return (
                    gr.update(visible=False),  # text_input
                    gr.update(value=html, visible=True),  # preview_html
                    gr.update(value="✎ Edit"),  # edit_btn label
                    "preview",  # view_mode
                )
            else:
                # Switch to edit mode — hide the toggle button
                return (
                    gr.update(visible=True),  # text_input
                    gr.update(visible=False),  # preview_html
                    gr.update(value="◉ Preview", visible=False),  # edit_btn label + hide
                    "edit",  # view_mode
                )

        edit_btn.click(
            fn=toggle_edit_preview,
            inputs=[view_mode, preview_html],
            outputs=[text_input, preview_html, edit_btn, view_mode],
        )

        # Upload button → load file, populate textbox or store file ref
        def handle_file_upload(file):
            if not file:
                return gr.update(), gr.update(), gr.update()
            ext = Path(file.name).suffix.lower()
            loader_map = {
                ".txt": "text",
                ".pdf": "pdf",
                ".docx": "docx",
                ".html": "html",
                ".htm": "html",
            }
            new_loader = loader_map.get(ext, "text")

            # For text files, read content into the textbox
            if ext == ".txt":
                try:
                    content = Path(file.name).read_text(encoding="utf-8")
                    return content, gr.update(value=new_loader), file
                except Exception:
                    pass
            # For non-text files, just store the file reference
            return gr.update(), gr.update(value=new_loader), file

        upload_btn.upload(
            fn=handle_file_upload,
            inputs=[upload_btn],
            outputs=[text_input, loader_type, file_input],
        )

        ner_type.change(
            fn=update_ner_params,
            inputs=[ner_type],
            outputs=[spacy_params, gliner_params, simple_params, ner_vram_info],
        )

        cand_type.change(
            fn=update_cand_params,
            inputs=[cand_type],
            outputs=[
                cand_embedding_model,
                cand_use_context,
                openai_api_dense_cand_params,
                cand_vram_info,
            ],
        )

        reranker_type.change(
            fn=update_reranker_params,
            inputs=[reranker_type],
            outputs=[
                reranker_cross_encoder_model,
                reranker_embedding_model,
                vllm_api_client_params,
                reranker_gpu_mem_gb,
                reranker_max_model_len,
                reranker_vram_info,
            ],
        )

        disambig_type.change(
            fn=update_disambig_params,
            inputs=[disambig_type],
            outputs=[llm_model, common_disambig_params, disambig_gpu_mem_gb, disambig_max_model_len, openai_api_disambig_params, disambig_vram_info],
        )

        # Update VRAM info when model selection changes
        cand_embedding_model.change(
            fn=lambda m: gr.update(value=_format_vram_info(m)),
            inputs=[cand_embedding_model],
            outputs=[cand_vram_info],
        )
        reranker_embedding_model.change(
            fn=lambda m: gr.update(value=_format_vram_info(m)),
            inputs=[reranker_embedding_model],
            outputs=[reranker_vram_info],
        )
        reranker_cross_encoder_model.change(
            fn=lambda m: gr.update(value=_format_vram_info(m)),
            inputs=[reranker_cross_encoder_model],
            outputs=[reranker_vram_info],
        )
        llm_model.change(
            fn=lambda m: gr.update(value=_format_vram_info(m)),
            inputs=[llm_model],
            outputs=[disambig_vram_info],
        )

        # Memory info display
        demo.load(
            fn=compute_memory_estimate,
            inputs=None,
            outputs=[memory_estimate_display],
        )

        # Unload All Models button
        def handle_unload():
            clear_all_models()
            return compute_memory_estimate()

        unload_btn.click(
            fn=handle_unload,
            inputs=None,
            outputs=[memory_estimate_display],
        )

        # Chain: clear outputs → run pipeline → restore buttons
        run_event = (
            run_btn.click(
                fn=clear_outputs_for_new_run,
                inputs=None,
                outputs=[
                    preview_html,
                    stats_output,
                    json_output,
                    run_btn,
                    cancel_btn,
                    text_input,
                    edit_btn,
                    view_mode,
                    upload_btn,
                    unload_btn,
                    kb_file,
                ],
            )
            .then(
                fn=run_pipeline,
                inputs=[
                    text_input,
                    file_input,
                    kb_file,
                    loader_type,
                    ner_type,
                    spacy_model,
                    gliner_model,
                    gliner_labels,
                    gliner_threshold,
                    labels_from_kb,
                    simple_min_len,
                    cand_type,
                    cand_embedding_model,
                    cand_top_k,
                    cand_use_context,
                    cand_api_base_url,
                    cand_api_key,
                    reranker_type,
                    reranker_embedding_model,
                    reranker_cross_encoder_model,
                    reranker_api_url,
                    reranker_api_port,
                    reranker_top_k,
                    reranker_gpu_mem_gb,
                    reranker_max_model_len,
                    disambig_type,
                    llm_model,
                    thinking,
                    none_candidate,
                    disambig_gpu_mem_gb,
                    disambig_max_model_len,
                    disambig_api_base_url,
                    disambig_api_key,
                    kb_type,
                ],
                outputs=[preview_html, stats_output, json_output, text_input, edit_btn, view_mode],
            )
            .then(
                fn=restore_buttons_after_run,
                inputs=None,
                outputs=[run_btn, cancel_btn, upload_btn, unload_btn, kb_file],
            )
            .then(
                fn=compute_memory_estimate,
                inputs=None,
                outputs=[memory_estimate_display],
            )
        )

        # Cancel button: show "Cancelling..." and set flag
        # The pipeline generator checks the flag and exits, then restore_buttons_after_run runs
        cancel_btn.click(
            fn=start_cancellation,
            inputs=None,
            outputs=[cancel_btn],
        )

    logger.info(f"Launching Gradio UI on port {args.port}...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        css=custom_css,
        theme=gr.themes.Default(primary_hue="green"),
    )

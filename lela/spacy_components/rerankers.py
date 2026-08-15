"""
spaCy reranker components for LELA.

Provides factories and components for candidate reranking:
- CrossEncoderRerankerComponent: Cross-encoder reranking (sentence-transformers)
- VLLMAPIClientReranker: Cross-encoder via external vLLM API
- LlamaServerReranker: Reranking via llama.cpp server
- LELAEmbedderRerankerComponent: Embedding-based cosine similarity (SentenceTransformer)
- LELAEmbedderVLLMRerankerComponent: Embedding-based cosine similarity (vLLM)
- LELACrossEncoderVLLMRerankerComponent: Cross-encoder via vLLM score() API (seq-cls)
- NoOpRerankerComponent: Pass-through (no reranking)
"""

import logging
import requests
from typing import List, Optional

import numpy as np
from spacy.language import Language
from spacy.tokens import Doc, Span

from lela.defaults import (
    RERANKER_TOP_K,
    DEFAULT_EMBEDDER_MODEL,
    DEFAULT_VLLM_RERANKER_MODEL,
    RERANKER_TASK,
    SPAN_OPEN,
    SPAN_CLOSE,
    CROSS_ENCODER_PREFIX,
    CROSS_ENCODER_SUFFIX,
    get_model_vram_gb,
)
from lela.llm_pool import (
    get_sentence_transformer_instance,
    release_sentence_transformer,
    get_vllm_instance,
    release_vllm,
    get_generic_instance,
    release_generic,
)
from lela.memory import gb_to_vllm_fraction
from lela.utils import ensure_candidates_extension
from lela.context import build_marked_text
from lela._types import Candidate, ProgressCallback

logger = logging.getLogger(__name__)

# Lazy imports for vLLM
_vllm = None


def _get_vllm():
    """Lazy import of vllm."""
    global _vllm
    if _vllm is None:
        try:
            import vllm

            _vllm = vllm
        except ImportError:
            raise ImportError(
                "vllm package required for vLLM reranker. "
                "Install with: pip install vllm"
            )
    return _vllm


# ============================================================================
# Cross-Encoder Reranker Component
# ============================================================================


@Language.factory(
    "cross_encoder_reranker",
    default_config={
        "model_name": "Qwen/Qwen3-Reranker-4B-seq-cls",
        "top_k": 10,
        "estimated_vram_gb": get_model_vram_gb("Qwen/Qwen3-Reranker-4B-seq-cls"),
        "predict_batch_size": 32,
        "pair_chunk_size": 256,
        "context_window": 0,
    },
)
def create_cross_encoder_reranker_component(
    nlp: Language,
    name: str,
    model_name: str,
    top_k: int,
    estimated_vram_gb: float,
    predict_batch_size: int,
    pair_chunk_size: int,
    context_window: int,
):
    """Factory for cross-encoder reranker component."""
    return CrossEncoderRerankerComponent(
        nlp=nlp,
        model_name=model_name,
        top_k=top_k,
        estimated_vram_gb=estimated_vram_gb,
        predict_batch_size=predict_batch_size,
        pair_chunk_size=pair_chunk_size,
        context_window=context_window,
    )


class CrossEncoderRerankerComponent:
    """
    Cross-encoder reranker component for spaCy.

    Uses sentence-transformers CrossEncoder for pairwise scoring.
    Pairs are batched across all mentions in the document; `predict_batch_size`
    caps the per-GPU-step batch (VRAM), while `pair_chunk_size` controls how
    many pairs are sent to `predict()` per call (Python-side memory + progress
    reporting granularity).
    """

    def __init__(
        self,
        nlp: Language,
        model_name: str = "Qwen/Qwen3-Reranker-4B-seq-cls",
        top_k: int = 10,
        estimated_vram_gb: Optional[float] = None,
        predict_batch_size: int = 32,
        pair_chunk_size: int = 256,
        context_window: int = 0,
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.top_k = top_k
        self.estimated_vram_gb = estimated_vram_gb if estimated_vram_gb is not None else get_model_vram_gb(model_name)
        self.predict_batch_size = predict_batch_size
        self.pair_chunk_size = pair_chunk_size
        self.context_window = context_window
        self.model = None

        ensure_candidates_extension()

        # Optional progress callback for fine-grained progress reporting
        self.progress_callback: Optional[ProgressCallback] = None

        logger.info(f"Cross-encoder reranker initialized: {model_name}")

    def _ensure_model_loaded(self):
        if self.model is not None:
            return
        key = f"cross_encoder:{self.model_name}"

        def loader():
            import torch
            from sentence_transformers import CrossEncoder

            # sentence-transformers wraps pairs in chat messages with
            # "query"/"document" roles when the tokenizer ships a chat template.
            # Qwen3's template ignores those roles and renders an empty string,
            # so the batch tokenizes to zero-length input_ids. We write the chat
            # markup ourselves, so drop the template to keep the text-pair path.
            logger.info(f"Loading CrossEncoder model: {self.model_name}")
            return CrossEncoder(
                self.model_name,
                model_kwargs={"torch_dtype": torch.float16},
                processor_kwargs={"chat_template": None},
                trust_remote_code=True,
            )

        self.model, _ = get_generic_instance(key, loader, self.estimated_vram_gb)

    def _format_query(self, doc: Doc, ent: Span) -> str:
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        marked_text = build_marked_text(doc, ent, self.context_window)
        return f"{prefix}<Instruct>: {RERANKER_TASK}\n<Query>: {marked_text}\n"

    def _format_candidate(self, candidate: Candidate) -> str:
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        document = f"{candidate.entity_id} ({candidate.description if candidate.description else ''})"
        return f"<Document>: {document}{suffix}"

    def __call__(self, doc: Doc) -> Doc:
        """Rerank candidates for all entities in the document."""
        entities = list(doc.ents)

        self._ensure_model_loaded()

        try:
            # Collect every (query, candidate) pair across all mentions,
            # remembering each mention's slice into the flat list.
            all_pairs: List[tuple] = []
            groups = []  # list of (start, end, ent, candidates)
            offset = 0
            for ent in entities:
                candidates = getattr(ent._, "candidates", [])
                if not candidates:
                    continue
                query = self._format_query(doc, ent)
                pairs = [(query, self._format_candidate(c)) for c in candidates]
                all_pairs.extend(pairs)
                groups.append((offset, offset + len(pairs), ent, candidates))
                offset += len(pairs)

            if not all_pairs:
                return doc

            total_pairs = len(all_pairs)
            all_scores = np.empty(total_pairs, dtype=np.float32)

            # Process pairs in chunks for progress reporting + Python-side
            # memory. Within each chunk, predict() mini-batches internally
            # using predict_batch_size, which is the actual VRAM cap.
            for chunk_start in range(0, total_pairs, self.pair_chunk_size):
                chunk_end = min(chunk_start + self.pair_chunk_size, total_pairs)
                chunk_scores = self.model.predict(
                    all_pairs[chunk_start:chunk_end],
                    batch_size=self.predict_batch_size,
                )
                all_scores[chunk_start:chunk_end] = chunk_scores

                if self.progress_callback:
                    self.progress_callback(
                        chunk_end / total_pairs,
                        f"Reranking {chunk_end}/{total_pairs} pairs",
                    )

            # Assign scores back per mention, sort, take top_k.
            for start, end, ent, candidates in groups:
                scores = all_scores[start:end]
                scored = sorted(
                    zip(candidates, scores), key=lambda x: x[1], reverse=True
                )[: self.top_k]
                ent._.candidates = [
                    Candidate(
                        entity_id=c.entity_id,
                        score=float(s),
                        description=c.description,
                    )
                    for c, s in scored
                ]
                ent._.candidate_scores = [float(s) for _, s in scored]
                logger.debug(
                    f"Cross-encoder reranked {end - start} to {len(ent._.candidates)} for '{ent.text}'"
                )
        finally:
            release_generic(f"cross_encoder:{self.model_name}")
            self.model = None

        # Clear progress callback after processing
        self.progress_callback = None

        return doc


# ============================================================================
# vLLM API Client Reranker Component
# ============================================================================


@Language.factory(
    "vllm_api_client_reranker",
    default_config={
        "top_k": 10,
        "base_url": "http://localhost",
        "port": 8000,
        "context_window": 0,
    },
)
def create_lela_vllm_api_client_reranker_component(
    nlp: Language,
    name: str,
    top_k: int,
    base_url: str,
    port: int,
    context_window: int,
):
    """Factory for vLLM API client reranker component."""
    return VLLMAPIClientReranker(
        nlp=nlp,
        top_k=top_k,
        base_url=base_url,
        port=port,
        context_window=context_window,
    )


class VLLMAPIClientReranker:
    PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    QUERY_TEMPLATE = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
    DOCUMENT_TEMPLATE = "<Document>: {doc}{suffix}"

    def __init__(
        self,
        nlp: Language,
        top_k: int = 10,
        base_url: str = "http://localhost",
        port: int = 8000,
        context_window: int = 0,
    ):
        self.nlp = nlp
        self.top_k = top_k
        self.api_url = f"{base_url}:{port}/score"
        self.context_window = context_window
        ensure_candidates_extension()
        logger.info(f"Using vLLM API reranker at {self.api_url}")
        self.progress_callback: Optional[ProgressCallback] = None

    @staticmethod
    def post_http_request(prompt: dict, api_url: str) -> requests.Response:
        headers = {"User-Agent": "Test Client"}
        response = requests.post(api_url, headers=headers, json=prompt)
        response.raise_for_status()
        return response

    def __call__(self, doc: Doc) -> Doc:
        entities = list(doc.ents)
        num_entities = len(entities)

        for i, ent in enumerate(entities):
            if self.progress_callback and num_entities > 0:
                progress = i / num_entities
                ent_text = ent.text[:25] + "..." if len(ent.text) > 25 else ent.text
                self.progress_callback(
                    progress, f"Reranking {i+1}/{num_entities}: {ent_text}"
                )

            candidates = getattr(ent._, "candidates", [])
            if not candidates:
                continue

            marked_text = build_marked_text(doc, ent, self.context_window)
            query = self.QUERY_TEMPLATE.format(
                prefix=self.PREFIX, instruction=RERANKER_TASK, query=marked_text
            )

            documents = [f"{c.entity_id} ({c.description or ''})" for c in candidates]
            documents = [
                self.DOCUMENT_TEMPLATE.format(doc=d, suffix=self.SUFFIX)
                for d in documents
            ]

            try:
                response = self.post_http_request(
                    prompt={
                        "text_1": query,
                        "text_2": documents,
                    },
                    api_url=self.api_url,
                ).json()

                if "data" not in response:
                    logger.error(
                        f"Reranker API response does not contain 'data' field: {response} for query: {query}"
                    )
                    # Keep original candidates if API fails
                    ent._.candidates = candidates[: self.top_k]
                    ent._.candidate_scores = [c.score for c in candidates[: self.top_k]]
                    continue

                scores = [d["score"] for d in response["data"]]
                scored_candidates = list(zip(candidates, scores))
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                top_candidates = scored_candidates[: self.top_k]

                reranked_candidates = [c for c, s in top_candidates]
                reranked_scores = [s for c, s in top_candidates]

                ent._.candidates = reranked_candidates
                ent._.candidate_scores = reranked_scores

            except (requests.exceptions.RequestException, ValueError) as e:
                logger.error(
                    f"Reranker API request failed for entity '{ent.text}': {e}"
                )
                # Keep original candidates on failure
                ent._.candidates = candidates[: self.top_k]
                ent._.candidate_scores = [c.score for c in candidates[: self.top_k]]
        self.progress_callback = None
        return doc


# ============================================================================
# No-Op Reranker Component
# ============================================================================


@Language.factory(
    "noop_reranker",
    default_config={"top_k": 10},
)
def create_noop_reranker_component(
    nlp: Language,
    name: str,
    top_k: int = 10,
):
    """Factory for no-op reranker component."""
    return NoOpRerankerComponent(nlp=nlp, top_k=top_k)


class NoOpRerankerComponent:
    """
    No-op reranker component for spaCy.

    Truncates candidates to top_k. Use when no reranking is needed.
    """

    def __init__(self, nlp: Language, top_k: int = 10):
        self.nlp = nlp
        self.top_k = top_k
        ensure_candidates_extension()

    def __call__(self, doc: Doc) -> Doc:
        """Truncate candidates to top_k without reranking."""
        for ent in doc.ents:
            candidates = getattr(ent._, "candidates", [])
            if candidates and len(candidates) > self.top_k:
                ent._.candidates = candidates[: self.top_k]
        return doc


# ============================================================================
# Llama Server Reranker Component
# ============================================================================


@Language.factory(
    "llama_server_reranker",
    default_config={
        "model_name": "qwen3-reranker",
        "top_k": 10,
        "base_url": "http://localhost",
        "port": 8000,
        "context_window": 0,
    },
)
def create_lela_llama_server_reranker_component(
    nlp: Language,
    name: str,
    model_name: str,
    top_k: int,
    base_url: str,
    port: int,
    context_window: int,
):
    """Factory for Llama Server reranker component."""
    return LlamaServerReranker(
        nlp=nlp,
        model_name=model_name,
        top_k=top_k,
        base_url=base_url,
        port=port,
        context_window=context_window,
    )


class LlamaServerReranker:
    """
    Reranker component that uses a llama.cpp server compatible with the
    OpenAI-style rerank endpoint.
    """

    def __init__(
        self,
        nlp: Language,
        model_name: str,
        top_k: int = 10,
        base_url: str = "http://localhost",
        port: int = 8000,
        context_window: int = 0,
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.top_k = top_k
        self.api_url = f"{base_url}:{port}/v1/rerank"
        self.context_window = context_window
        ensure_candidates_extension()
        logger.info(
            f"Using Llama Server reranker for model '{self.model_name}' at {self.api_url}"
        )
        self.progress_callback: Optional[ProgressCallback] = None

    @staticmethod
    def post_http_request(payload: dict, api_url: str) -> requests.Response:
        headers = {"User-Agent": "LELA Client", "Content-Type": "application/json"}
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        return response

    def __call__(self, doc: Doc) -> Doc:
        entities = list(doc.ents)
        num_entities = len(entities)

        for i, ent in enumerate(entities):
            if self.progress_callback and num_entities > 0:
                progress = i / num_entities
                ent_text = ent.text[:25] + "..." if len(ent.text) > 25 else ent.text
                self.progress_callback(
                    progress, f"Reranking {i+1}/{num_entities}: {ent_text}"
                )

            candidates = getattr(ent._, "candidates", [])
            if not candidates:
                continue

            query_text = build_marked_text(doc, ent, self.context_window)
            document_texts = [
                f"{c.entity_id} ({c.description or ''})" for c in candidates
            ]

            try:
                response = self.post_http_request(
                    payload={
                        "model": self.model_name,
                        "query": query_text,
                        "documents": document_texts,
                        "top_n": self.top_k,
                    },
                    api_url=self.api_url,
                ).json()

                if "results" not in response:
                    logger.error(
                        f"Reranker API response does not contain 'results' field: {response} for query: {query_text}"
                    )
                    # Keep original candidates if API fails
                    ent._.candidates = candidates[: self.top_k]
                    ent._.candidate_scores = [c.score for c in candidates[: self.top_k]]
                    continue

                results = response["results"]

                reranked_candidates = []
                reranked_scores = []

                for result in results:
                    original_index = result.get("index")
                    score = result.get("relevance_score")

                    if original_index is None or score is None:
                        continue

                    if 0 <= original_index < len(candidates):
                        candidate = candidates[original_index]
                        reranked_candidates.append(
                            Candidate(
                                entity_id=candidate.entity_id,
                                score=float(score),
                                description=candidate.description,
                            )
                        )
                        reranked_scores.append(float(score))

                ent._.candidates = reranked_candidates
                ent._.candidate_scores = reranked_scores

            except (requests.exceptions.RequestException, ValueError) as e:
                logger.error(
                    f"Llama Server Reranker API request failed for entity '{ent.text}': {e}"
                )
                # Keep original candidates on failure
                ent._.candidates = candidates[: self.top_k]
                ent._.candidate_scores = [c.score for c in candidates[: self.top_k]]
        self.progress_callback = None
        return doc


# ============================================================================
# LELA Embedder Reranker Component (SentenceTransformer)
# ============================================================================


@Language.factory(
    "embedder_transformers_reranker",
    default_config={
        "model_name": DEFAULT_EMBEDDER_MODEL,
        "top_k": RERANKER_TOP_K,
        "device": None,
        "estimated_vram_gb": get_model_vram_gb(DEFAULT_EMBEDDER_MODEL),
        "context_window": 0,
    },
)
def create_lela_embedder_transformers_reranker_component(
    nlp: Language,
    name: str,
    model_name: str,
    top_k: int,
    device: Optional[str],
    estimated_vram_gb: float,
    context_window: int,
):
    """Factory for LELA embedder reranker component."""
    return LELAEmbedderRerankerComponent(
        nlp=nlp,
        model_name=model_name,
        top_k=top_k,
        device=device,
        estimated_vram_gb=estimated_vram_gb,
        context_window=context_window,
    )


class LELAEmbedderRerankerComponent:
    """
    Embedding-based reranker component for spaCy.

    Uses SentenceTransformers to rerank candidates by cosine similarity.
    The mention is marked in the document text with brackets for context.

    Memory management: Model is loaded on-demand and released after use,
    allowing it to be evicted if memory is needed for later stages.
    """

    def __init__(
        self,
        nlp: Language,
        model_name: str = DEFAULT_EMBEDDER_MODEL,
        top_k: int = RERANKER_TOP_K,
        device: Optional[str] = None,
        estimated_vram_gb: Optional[float] = None,
        context_window: int = 0,
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.top_k = top_k
        self.device = device
        self.estimated_vram_gb = estimated_vram_gb if estimated_vram_gb is not None else get_model_vram_gb(model_name)
        self.context_window = context_window

        ensure_candidates_extension()

        self.progress_callback: Optional[ProgressCallback] = None

        logger.info(f"LELA embedder reranker initialized: {model_name}")

    def _embed_texts(self, texts: List[str], model) -> np.ndarray:
        """Embed texts using the SentenceTransformer model."""
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def _format_query(self, doc: Doc, ent: Span) -> str:
        """Format query with marked mention in text."""
        marked_text = build_marked_text(doc, ent, self.context_window)
        return f"Instruct: {RERANKER_TASK}\nQuery: {marked_text}"

    def _format_candidate(self, candidate: Candidate) -> str:
        """Format candidate for embedding."""
        if candidate.description:
            return f"{candidate.entity_id}: {candidate.description}"
        return candidate.entity_id

    def __call__(self, doc: Doc) -> Doc:
        """Rerank candidates for all entities in the document."""
        entities = list(doc.ents)
        num_entities = len(entities)

        needs_reranking = any(
            len(getattr(ent._, "candidates", [])) > self.top_k for ent in entities
        )

        if not needs_reranking:
            return doc

        if self.progress_callback:
            self.progress_callback(
                0.0, f"Loading reranker model ({self.model_name.split('/')[-1]})..."
            )

        model, was_cached = get_sentence_transformer_instance(
            self.model_name, self.device, estimated_vram_gb=self.estimated_vram_gb
        )

        if self.progress_callback:
            status = "Using cached model" if was_cached else "Model loaded"
            self.progress_callback(0.1, f"{status}, reranking candidates...")

        processing_start = 0.1
        processing_range = 0.9

        try:
            for i, ent in enumerate(entities):
                if self.progress_callback and num_entities > 0:
                    progress = processing_start + (i / num_entities) * processing_range
                    ent_text = ent.text[:25] + "..." if len(ent.text) > 25 else ent.text
                    self.progress_callback(
                        progress, f"Reranking {i+1}/{num_entities}: {ent_text}"
                    )

                candidates = getattr(ent._, "candidates", [])
                if not candidates or len(candidates) <= self.top_k:
                    continue

                query_text = self._format_query(doc, ent)
                candidate_texts = [self._format_candidate(c) for c in candidates]

                all_texts = [query_text] + candidate_texts
                embeddings = self._embed_texts(all_texts, model)

                query_embedding = embeddings[0:1]
                candidate_embeddings = embeddings[1:]
                similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()

                scored_candidates = list(zip(candidates, similarities))
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                top_candidates = scored_candidates[: self.top_k]

                reranked = []
                reranked_scores = []
                for candidate, score in top_candidates:
                    reranked.append(
                        Candidate(
                            entity_id=candidate.entity_id,
                            score=float(score),
                            description=candidate.description,
                        )
                    )
                    reranked_scores.append(float(score))

                ent._.candidates = reranked
                ent._.candidate_scores = reranked_scores

                logger.debug(
                    f"Reranked {len(candidates)} to {len(ent._.candidates)} for '{ent.text}'"
                )
        finally:
            release_sentence_transformer(self.model_name, self.device)

        self.progress_callback = None
        return doc


# ============================================================================
# LELA Cross-Encoder vLLM Reranker Component
# ============================================================================


@Language.factory(
    "cross_encoder_vllm_reranker",
    default_config={
        "model_name": DEFAULT_VLLM_RERANKER_MODEL,
        "top_k": RERANKER_TOP_K,
        "gpu_memory_gb": None,
        "max_model_len": None,
        "context_window": 0,
    },
)
def create_lela_cross_encoder_vllm_reranker_component(
    nlp: Language,
    name: str,
    model_name: str,
    top_k: int,
    gpu_memory_gb: Optional[float],
    max_model_len: Optional[int],
    context_window: int,
):
    """Factory for LELA cross-encoder vLLM reranker component."""
    return LELACrossEncoderVLLMRerankerComponent(
        nlp=nlp,
        model_name=model_name,
        top_k=top_k,
        gpu_memory_gb=gpu_memory_gb,
        max_model_len=max_model_len,
        context_window=context_window,
    )


class LELACrossEncoderVLLMRerankerComponent:
    """
    Cross-encoder reranker using vLLM's score() API with seq-cls models.

    Uses the Qwen3-Reranker-seq-cls model variant which has a classification
    head, enabling direct use of vLLM's score() API for relevance scoring.
    Model is loaded on-demand and released after use.
    """

    QUERY_TEMPLATE = "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n"
    DOCUMENT_TEMPLATE = "<Document>: {doc}{suffix}"

    def __init__(
        self,
        nlp: Language,
        model_name: str = DEFAULT_VLLM_RERANKER_MODEL,
        top_k: int = RERANKER_TOP_K,
        gpu_memory_gb: Optional[float] = None,
        max_model_len: Optional[int] = None,
        context_window: int = 0,
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.top_k = top_k
        self.gpu_memory_gb = gpu_memory_gb
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gb_to_vllm_fraction(gpu_memory_gb) if gpu_memory_gb is not None else None
        self.context_window = context_window

        ensure_candidates_extension()

        self.model = None
        self.progress_callback: Optional[ProgressCallback] = None

        logger.info(f"LELA cross-encoder vLLM reranker initialized: {model_name}")

    def _format_query(self, doc: Doc, ent: Span) -> str:
        """Format query with marked mention in text."""
        marked_text = build_marked_text(doc, ent, self.context_window)
        return self.QUERY_TEMPLATE.format(
            prefix=CROSS_ENCODER_PREFIX,
            instruction=RERANKER_TASK,
            query=marked_text,
        )

    def _format_document(self, candidate: Candidate) -> str:
        """Format a candidate as a document string for scoring."""
        doc_text = f"{candidate.entity_id} ({candidate.description or ''})"
        return self.DOCUMENT_TEMPLATE.format(doc=doc_text, suffix=CROSS_ENCODER_SUFFIX)

    def _ensure_model_loaded(self, progress_callback=None):
        """Load model on-demand if not already loaded."""
        if self.model is None:
            _get_vllm()

            if progress_callback:
                progress_callback(
                    0.0, f"Loading reranker model ({self.model_name.split('/')[-1]})..."
                )

            self.model, was_cached = get_vllm_instance(
                model_name=self.model_name,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                estimated_vram_gb=self.gpu_memory_gb or 10.0,
                hf_overrides={
                    "architectures": ["Qwen3ForSequenceClassification"],
                    "classifier_from_token": ["no", "yes"],
                    "is_original_qwen3_reranker": True,
                },
            )

            if progress_callback:
                status = "Using cached model" if was_cached else "Model loaded"
                progress_callback(0.1, f"{status}, reranking candidates...")

    def __call__(self, doc: Doc) -> Doc:
        """Rerank candidates for all entities in the document."""
        entities = list(doc.ents)
        num_entities = len(entities)

        needs_reranking = any(
            len(getattr(ent._, "candidates", [])) > self.top_k for ent in entities
        )

        if not needs_reranking:
            return doc

        self._ensure_model_loaded(self.progress_callback)

        try:
            # Collect all (query, document) pairs across entities for batched scoring
            all_queries = []
            all_documents = []
            # Track which entities need reranking and how many pairs each has
            work_items = []  # (entity_index, candidates, num_pairs)

            for i, ent in enumerate(entities):
                candidates = getattr(ent._, "candidates", [])
                if not candidates or len(candidates) <= self.top_k:
                    continue

                query = self._format_query(doc, ent)
                documents = [self._format_document(c) for c in candidates]

                work_items.append((i, candidates, len(documents)))
                all_queries.extend([query] * len(documents))
                all_documents.extend(documents)

            if all_queries:
                if self.progress_callback:
                    self.progress_callback(
                        0.2,
                        f"Scoring {len(all_queries)} pairs across {len(work_items)} entities...",
                    )

                # Single batched score call using vLLM's N -> N pattern
                outputs = self.model.score(all_queries, all_documents)
                all_scores = [out.outputs.score for out in outputs]

                # Split scores back per entity and apply top_k selection
                offset = 0
                for ent_idx, candidates, num_pairs in work_items:
                    ent = entities[ent_idx]
                    scores = all_scores[offset : offset + num_pairs]
                    offset += num_pairs

                    scored_candidates = list(zip(candidates, scores))
                    scored_candidates.sort(key=lambda x: x[1], reverse=True)
                    top_candidates = scored_candidates[: self.top_k]

                    reranked = []
                    reranked_scores = []
                    for candidate, score in top_candidates:
                        reranked.append(
                            Candidate(
                                entity_id=candidate.entity_id,
                                score=float(score),
                                description=candidate.description,
                            )
                        )
                        reranked_scores.append(float(score))

                    ent._.candidates = reranked
                    ent._.candidate_scores = reranked_scores

                    logger.debug(
                        f"Cross-encoder (vLLM) reranked {len(candidates)} to {len(ent._.candidates)} for '{ent.text}'"
                    )
        finally:
            release_vllm(self.model_name, gpu_memory_utilization=self.gpu_memory_utilization)
            self.model = None  # Drop reference so pool eviction can free GPU memory

        self.progress_callback = None
        return doc


# ============================================================================
# LELA Embedder vLLM Reranker Component
# ============================================================================


@Language.factory(
    "embedder_vllm_reranker",
    default_config={
        "model_name": DEFAULT_EMBEDDER_MODEL,
        "top_k": RERANKER_TOP_K,
        "gpu_memory_gb": None,
        "max_model_len": None,
        "context_window": 0,
    },
)
def create_lela_embedder_vllm_reranker_component(
    nlp: Language,
    name: str,
    model_name: str,
    top_k: int,
    gpu_memory_gb: Optional[float],
    max_model_len: Optional[int],
    context_window: int,
):
    """Factory for LELA embedder vLLM reranker component."""
    return LELAEmbedderVLLMRerankerComponent(
        nlp=nlp,
        model_name=model_name,
        top_k=top_k,
        gpu_memory_gb=gpu_memory_gb,
        max_model_len=max_model_len,
        context_window=context_window,
    )


class LELAEmbedderVLLMRerankerComponent:
    """
    Embedding-based reranker using vLLM's .encode() API.

    Same functionality as LELAEmbedderRerankerComponent but using vLLM
    instead of SentenceTransformers for faster inference.
    Model is loaded on-demand and released after use.
    """

    def __init__(
        self,
        nlp: Language,
        model_name: str = DEFAULT_EMBEDDER_MODEL,
        top_k: int = RERANKER_TOP_K,
        gpu_memory_gb: Optional[float] = None,
        max_model_len: Optional[int] = None,
        context_window: int = 0,
    ):
        self.nlp = nlp
        self.model_name = model_name
        self.top_k = top_k
        self.gpu_memory_gb = gpu_memory_gb
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gb_to_vllm_fraction(gpu_memory_gb) if gpu_memory_gb is not None else None
        self.context_window = context_window

        ensure_candidates_extension()

        self.model = None
        self.progress_callback: Optional[ProgressCallback] = None

        logger.info(f"LELA embedder vLLM reranker initialized: {model_name}")

    def _format_query(self, doc: Doc, ent: Span) -> str:
        """Format query with marked mention in text."""
        marked_text = build_marked_text(doc, ent, self.context_window)
        return f"Instruct: {RERANKER_TASK}\nQuery: {marked_text}"

    def _format_candidate(self, candidate: Candidate) -> str:
        """Format candidate for embedding."""
        if candidate.description:
            return f"{candidate.entity_id}: {candidate.description}"
        return candidate.entity_id

    def _ensure_model_loaded(self, progress_callback=None):
        """Load model on-demand if not already loaded."""
        if self.model is None:
            _get_vllm()

            if progress_callback:
                progress_callback(
                    0.0, f"Loading reranker model ({self.model_name.split('/')[-1]})..."
                )

            self.model, was_cached = get_vllm_instance(
                model_name=self.model_name,
                convert="embed",
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                estimated_vram_gb=self.gpu_memory_gb or 10.0,
            )

            if progress_callback:
                status = "Using cached model" if was_cached else "Model loaded"
                progress_callback(0.1, f"{status}, reranking candidates...")

    def __call__(self, doc: Doc) -> Doc:
        """Rerank candidates for all entities in the document."""
        entities = list(doc.ents)
        num_entities = len(entities)

        needs_reranking = any(
            len(getattr(ent._, "candidates", [])) > self.top_k for ent in entities
        )

        if not needs_reranking:
            return doc

        self._ensure_model_loaded(self.progress_callback)

        processing_start = 0.1
        processing_range = 0.9

        try:
            for i, ent in enumerate(entities):
                if self.progress_callback and num_entities > 0:
                    progress = processing_start + (i / num_entities) * processing_range
                    ent_text = ent.text[:25] + "..." if len(ent.text) > 25 else ent.text
                    self.progress_callback(
                        progress, f"Reranking {i+1}/{num_entities}: {ent_text}"
                    )

                candidates = getattr(ent._, "candidates", [])
                if not candidates or len(candidates) <= self.top_k:
                    continue

                query_text = self._format_query(doc, ent)
                candidate_texts = [self._format_candidate(c) for c in candidates]

                all_texts = [query_text] + candidate_texts
                outputs = self.model.encode(all_texts)

                embeddings = np.array([out.outputs.embedding for out in outputs])
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)
                embeddings = embeddings / norms

                query_embedding = embeddings[0:1]
                candidate_embeddings = embeddings[1:]
                similarities = np.dot(candidate_embeddings, query_embedding.T).flatten()

                scored_candidates = list(zip(candidates, similarities))
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                top_candidates = scored_candidates[: self.top_k]

                reranked = []
                reranked_scores = []
                for candidate, score in top_candidates:
                    reranked.append(
                        Candidate(
                            entity_id=candidate.entity_id,
                            score=float(score),
                            description=candidate.description,
                        )
                    )
                    reranked_scores.append(float(score))

                ent._.candidates = reranked
                ent._.candidate_scores = reranked_scores

                logger.debug(
                    f"Embedder (vLLM) reranked {len(candidates)} to {len(ent._.candidates)} for '{ent.text}'"
                )
        finally:
            release_vllm(self.model_name, convert="embed", gpu_memory_utilization=self.gpu_memory_utilization)
            self.model = None  # Drop reference so pool eviction can free GPU memory

        self.progress_callback = None
        return doc

"""BERT Span Text Encoder - Extracts span-level embeddings from BERT.

This module provides a replacement for StructuredTextEncoder that uses
BERT-derived span embeddings instead of random embeddings.

Key difference from v1:
- v1: StructuredTextEncoder uses random embeddings (TextHashEncoder)
- v2: SpanTextEncoder uses BERT token embeddings for parsed spans

The span embeddings are semantically grounded because:
1. They come from the same BERT model that processes the utterance
2. Target/anchor/relation embeddings share the same semantic space
3. They maintain learned relationships (chair ~ table, etc.)

Usage:
    encoder = SpanTextEncoder(
        bert_model_name="distilbert-base-uncased",
        output_dim=256,
    )

    # Encode batch
    q_t, q_a, q_r, masks = encoder.encode_batch(
        utterances=["the chair next to the table"],
        alignments=[span_alignment_result],
    )
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class SpanTextEncoder(nn.Module):
    """Extract span-level embeddings from BERT token embeddings.

    This encoder:
    1. Tokenizes utterances with BERT
    2. Gets token-level embeddings from BERT
    3. Extracts span embeddings for target/anchor/relation
    4. Projects to output dimension
    5. Returns masks indicating whether spans were found
    """

    def __init__(
        self,
        bert_model_name: str = "distilbert-base-uncased",
        output_dim: int = 256,
        max_length: int = 128,
        use_cls_fallback: bool = True,
        freeze_bert: bool = True,
    ):
        """Initialize SpanTextEncoder.

        Args:
            bert_model_name: HuggingFace model name
            output_dim: Output embedding dimension (projected from BERT hidden dim)
            max_length: Maximum sequence length for BERT
            use_cls_fallback: Use CLS token embedding when span not found
            freeze_bert: Whether to freeze BERT parameters
        """
        super().__init__()

        self.bert_model_name = bert_model_name
        self.output_dim = output_dim
        self.max_length = max_length
        self.use_cls_fallback = use_cls_fallback

        # Load BERT model and tokenizer
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers")

        log.info(f"Loading BERT model: {bert_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name)

        # Get hidden dimension
        self.bert_hidden_dim = self.bert.config.hidden_size

        # Freeze BERT if requested
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            log.info("BERT parameters frozen")

        # Projection layers
        self.proj_target = nn.Linear(self.bert_hidden_dim, output_dim)
        self.proj_anchor = nn.Linear(self.bert_hidden_dim, output_dim)
        self.proj_relation = nn.Linear(self.bert_hidden_dim, output_dim)

        # Output activation
        self.tanh = nn.Tanh()

    def encode_utterance_tokens(
        self,
        utterance: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """Encode a single utterance with BERT, returning token embeddings.

        Args:
            utterance: Text string

        Returns:
            token_embeddings: [seq_len, hidden_dim]
            attention_mask: [seq_len]
            encoding_info: dict with tokenization details
        """
        # Tokenize
        encoded = self.tokenizer(
            utterance,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        # Forward pass through BERT
        # Always use no_grad if BERT parameters are frozen (don't require gradients)
        use_no_grad = not any(p.requires_grad for p in self.bert.parameters())
        with torch.no_grad() if use_no_grad else torch.enable_grad():
            outputs = self.bert(
                input_ids=encoded["input_ids"].to(self.bert.device),
                attention_mask=encoded["attention_mask"].to(self.bert.device),
            )

        # Get token embeddings (exclude batch dimension since single utterance)
        token_embeddings = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]
        attention_mask = encoded["attention_mask"].squeeze(0).to(token_embeddings.device)  # [seq_len]

        # Store offset mapping for span alignment
        encoding_info = {
            "input_ids": encoded["input_ids"].squeeze(0).to(token_embeddings.device),
            "offset_mapping": encoded.get("offset_mapping", torch.zeros(self.max_length, 2, device=token_embeddings.device)),
            "tokens": self.tokenizer.convert_ids_to_tokens(encoded["input_ids"].squeeze(0)),
        }

        return token_embeddings, attention_mask, encoding_info

    def extract_span_embedding(
        self,
        token_embeddings: torch.Tensor,
        token_start: Optional[int],
        token_end: Optional[int],
        attention_mask: torch.Tensor,
        fallback_to_cls: bool = True,
    ) -> Tuple[torch.Tensor, bool]:
        """Extract embedding for a span of tokens.

        Args:
            token_embeddings: [seq_len, hidden_dim]
            token_start: Start token index (None if not found)
            token_end: End token index (exclusive, None if not found)
            attention_mask: [seq_len]
            fallback_to_cls: Use CLS token if span not found

        Returns:
            span_embedding: [hidden_dim]
            found: Whether span was successfully found
        """
        hidden_dim = token_embeddings.shape[-1]

        if token_start is not None and token_end is not None:
            # Valid span indices
            # Ensure indices are within bounds
            seq_len = token_embeddings.shape[0]
            token_start = min(token_start, seq_len - 1)
            token_end = min(token_end, seq_len)

            if token_start < token_end:
                # Extract span tokens
                span_embeddings = token_embeddings[token_start:token_end]  # [span_len, hidden_dim]

                # Mean pooling over span
                span_embedding = span_embeddings.mean(dim=0)  # [hidden_dim]
                return span_embedding, True

        # Span not found, use fallback
        if fallback_to_cls:
            # CLS token is at index 0
            cls_embedding = token_embeddings[0]  # [hidden_dim]
            return cls_embedding, False
        else:
            # Zero embedding
            zero_embedding = torch.zeros(hidden_dim, device=token_embeddings.device)
            return zero_embedding, False

    def encode_single(
        self,
        utterance: str,
        alignment: "UtteranceSpanAlignment",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a single utterance with span alignment.

        Args:
            utterance: Text string
            alignment: UtteranceSpanAlignment from span_alignment.py

        Returns:
            q_t: Target span embedding [output_dim]
            q_a: Anchor span embedding [output_dim]
            q_r: Relation span embedding [output_dim]
            masks: [3] boolean tensor (target_found, anchor_found, relation_found)
        """
        # Get BERT token embeddings
        token_embeddings, attention_mask, encoding_info = self.encode_utterance_tokens(utterance)

        # Convert character spans to token spans if needed
        # (alignment may have token indices if tokenizer was used during alignment)
        # If not, we compute token spans here using offset mapping
        offsets = encoding_info["offset_mapping"]

        def char_to_token(char_start: Optional[int], char_end: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
            """Convert character indices to token indices using offset mapping."""
            if char_start is None or char_end is None:
                return None, None

            token_start = None
            token_end = None

            for i, (start, end) in enumerate(offsets.tolist()):
                if start == end:  # Special token (CLS, SEP, PAD)
                    continue
                if start <= char_start < end:
                    token_start = i
                if start < char_end <= end:
                    token_end = i + 1
                if token_start is not None and token_end is not None:
                    break

            return token_start, token_end

        # Get token spans for each component
        target_token_start = alignment.target.token_start
        target_token_end = alignment.target.token_end

        # If token indices not available, convert from character indices
        if target_token_start is None and alignment.target.char_start is not None:
            target_token_start, target_token_end = char_to_token(
                alignment.target.char_start, alignment.target.char_end
            )

        anchor_token_start = alignment.anchor.token_start
        anchor_token_end = alignment.anchor.token_end
        if anchor_token_start is None and alignment.anchor.char_start is not None:
            anchor_token_start, anchor_token_end = char_to_token(
                alignment.anchor.char_start, alignment.anchor.char_end
            )

        relation_token_start = alignment.relation.token_start
        relation_token_end = alignment.relation.token_end
        if relation_token_start is None and alignment.relation.char_start is not None:
            relation_token_start, relation_token_end = char_to_token(
                alignment.relation.char_start, alignment.relation.char_end
            )

        # Extract span embeddings
        target_emb, target_found = self.extract_span_embedding(
            token_embeddings, target_token_start, target_token_end, attention_mask, self.use_cls_fallback
        )
        anchor_emb, anchor_found = self.extract_span_embedding(
            token_embeddings, anchor_token_start, anchor_token_end, attention_mask, self.use_cls_fallback
        )
        relation_emb, relation_found = self.extract_span_embedding(
            token_embeddings, relation_token_start, relation_token_end, attention_mask, self.use_cls_fallback
        )

        # Project to output dimension
        q_t = self.tanh(self.proj_target(target_emb))
        q_a = self.tanh(self.proj_anchor(anchor_emb))
        q_r = self.tanh(self.proj_relation(relation_emb))

        # Create masks on same device as token_embeddings
        masks = torch.tensor([target_found, anchor_found, relation_found], dtype=torch.bool, device=token_embeddings.device)

        return q_t, q_a, q_r, masks

    def encode_batch(
        self,
        utterances: List[str],
        alignments: List["UtteranceSpanAlignment"],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a batch of utterances with span alignments.

        Args:
            utterances: List of text strings
            alignments: List of UtteranceSpanAlignment results

        Returns:
            q_t: Target span embeddings [B, output_dim]
            q_a: Anchor span embeddings [B, output_dim]
            q_r: Relation span embeddings [B, output_dim]
            masks: [B, 3] boolean tensor (found status for each component)
        """
        batch_size = len(utterances)

        # Encode each utterance
        q_t_list = []
        q_a_list = []
        q_r_list = []
        masks_list = []

        for i in range(batch_size):
            utterance = utterances[i]
            alignment = alignments[i] if i < len(alignments) else None

            if alignment is None:
                # Create empty alignment
                from rag3d.parsers.span_alignment import UtteranceSpanAlignment, SpanAlignment
                alignment = UtteranceSpanAlignment(
                    utterance=utterance,
                    target=SpanAlignment(text="", found=False, fallback_used=True),
                    anchor=SpanAlignment(text="", found=False, fallback_used=True),
                    relation=SpanAlignment(text="", found=False, fallback_used=True),
                )

            q_t, q_a, q_r, masks = self.encode_single(utterance, alignment)
            q_t_list.append(q_t)
            q_a_list.append(q_a)
            q_r_list.append(q_r)
            masks_list.append(masks)

        # Stack into batch tensors
        q_t = torch.stack(q_t_list, dim=0)  # [B, output_dim]
        q_a = torch.stack(q_a_list, dim=0)  # [B, output_dim]
        q_r = torch.stack(q_r_list, dim=0)  # [B, output_dim]
        masks = torch.stack(masks_list, dim=0)  # [B, 3]

        return q_t, q_a, q_r, masks

    def forward(
        self,
        utterances: List[str],
        parsed_list: List,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode batch from parsed outputs.

        This is the main entry point for model integration.
        It aligns parsed components and extracts span embeddings.

        Args:
            utterances: List of raw utterance strings
            parsed_list: List of parsed outputs (ParsedUtterance or dict)

        Returns:
            q_t: Target embeddings [B, output_dim]
            q_a: Anchor embeddings [B, output_dim]
            q_r: Relation embeddings [B, output_dim]
            masks: [B, 3] boolean masks for found status
        """
        from rag3d.parsers.span_alignment import align_batch_utterances

        # Align parsed components to spans
        alignments = align_batch_utterances(utterances, parsed_list, self.tokenizer)

        # Extract span embeddings
        return self.encode_batch(utterances, alignments)


# Convenience function for building encoder
def build_span_text_encoder(config: dict) -> SpanTextEncoder:
    """Build SpanTextEncoder from config.

    Args:
        config: Configuration dict

    Returns:
        SpanTextEncoder instance
    """
    text_config = config.get("text_encoder", {})
    return SpanTextEncoder(
        bert_model_name=text_config.get("bert_model_name", "distilbert-base-uncased"),
        output_dim=text_config.get("output_dim", 256),
        max_length=text_config.get("max_length", 128),
        use_cls_fallback=text_config.get("use_cls_fallback", True),
        freeze_bert=text_config.get("freeze_bert", True),
    )


# Test function
def test_span_text_encoder():
    """Test SpanTextEncoder on sample utterances."""
    from rag3d.parsers.span_alignment import align_parsed_utterance

    encoder = SpanTextEncoder(output_dim=256)

    utterances = [
        "the chair next to the table",
        "the lamp above the desk",
        "the red sofa",
    ]

    test_parsed = [
        {"target_head": "chair", "anchor_head": "table", "relation_types": ["next-to"]},
        {"target_head": "lamp", "anchor_head": "desk", "relation_types": ["above"]},
        {"target_head": "sofa", "anchor_head": None, "relation_types": ["none"]},
    ]

    # Convert to ParsedUtterance-like objects
    from dataclasses import dataclass

    @dataclass
    class SimpleParsed:
        target_head: str
        anchor_head: str
        relation_types: list

    parsed_list = [
        SimpleParsed(**p) for p in test_parsed
    ]

    # Encode
    q_t, q_a, q_r, masks = encoder.forward(utterances, parsed_list)

    print(f"Target embeddings shape: {q_t.shape}")
    print(f"Anchor embeddings shape: {q_a.shape}")
    print(f"Relation embeddings shape: {q_r.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Masks:\n{masks}")

    # Check semantic relationships
    print("\nSemantic grounding check:")
    print(f"  'chair' vs 'table' similarity: {torch.cosine_similarity(q_t[0:1], q_a[0:1]).item():.3f}")
    print(f"  'chair' vs 'lamp' similarity: {torch.cosine_similarity(q_t[0:1], q_t[1:2]).item():.3f}")
    # These should have meaningful similarities based on BERT semantics


if __name__ == "__main__":
    test_span_text_encoder()
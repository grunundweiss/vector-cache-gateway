"""Semantic cache gateway over a local vector store."""

from gateway.cache import CacheHit, SemanticCache
from gateway.conversation import (
    ContextWindowResolver,
    ResolvedQuery,
    RewriteResolver,
    Turn,
    VerbatimResolver,
)
from gateway.engine import CacheResult, MissOutcome, SemanticCacheEngine
from gateway.generator import (
    Generator,
    InferenceEngineGenerator,
    RetrievalOnlyGenerator,
)
from gateway.guard import (
    AcronymGuard,
    CacheGuard,
    CompositeGuard,
    NullGuard,
    NumericGuard,
    PolarityGuard,
    RerankGuard,
    TokenOverlapGuard,
    default_guard,
)
from gateway.index import available_backends, build_index
from gateway.metrics import CacheMetrics
from gateway.middleware import ChatCacheMiddleware, is_cacheable
from gateway.partition import PartitionedCache
from gateway.persistence import SnapshotMismatch, load_snapshot, save_snapshot
from gateway.policy import GlobalThreshold, IntentThresholds, classify
from gateway.semantic_cache import SemanticCacheGateway
from gateway.vector_store import LocalVectorStore

__all__ = [
    "AcronymGuard",
    "CacheGuard",
    "CacheHit",
    "CacheMetrics",
    "CacheResult",
    "ChatCacheMiddleware",
    "CompositeGuard",
    "ContextWindowResolver",
    "Generator",
    "GlobalThreshold",
    "InferenceEngineGenerator",
    "IntentThresholds",
    "LocalVectorStore",
    "MissOutcome",
    "NullGuard",
    "NumericGuard",
    "PartitionedCache",
    "PolarityGuard",
    "RerankGuard",
    "ResolvedQuery",
    "RetrievalOnlyGenerator",
    "RewriteResolver",
    "SemanticCache",
    "SemanticCacheEngine",
    "SemanticCacheGateway",
    "SnapshotMismatch",
    "TokenOverlapGuard",
    "Turn",
    "VerbatimResolver",
    "available_backends",
    "build_index",
    "classify",
    "default_guard",
    "is_cacheable",
    "load_snapshot",
    "save_snapshot",
]

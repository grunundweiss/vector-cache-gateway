"""A semantic cache as an HTTP proxy in front of a chat endpoint.

    pip install -e ".[serve]"
    uvicorn examples.fastapi_proxy:app --port 8000

    curl -s localhost:8000/v1/chat/completions \
      -H 'x-user-id: alice' -H 'content-type: application/json' \
      -d '{"messages":[{"role":"user","content":"How long must KYC records be kept?"}]}'

This is not part of the package and nothing imports it; FastAPI is an optional
extra. It exists because "use it as middleware" is a sentence that hides three
decisions, and they are easier to read as code:

**Where the namespace comes from.** Here, an authenticated user header. Getting
this wrong is not a tuning mistake -- it serves one user's answer to another.
The default of a single shared namespace is correct only for answers that
depend on nothing user-specific.

**What the client is told.** The cache status, the similarity and the entry id
come back in headers, so a client can attach feedback to the answer it actually
received and an operator can see which responses were served from cache without
reading logs.

**What happens on a tool call.** Nothing: `is_cacheable` rejects the request and
it goes straight upstream. That is the middleware's job, not the caller's.

Replace `call_model` with a real client. Everything else works as written.
"""

from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from gateway.engine import SemanticCacheEngine
from gateway.middleware import ChatCacheMiddleware
from gateway.vector_store import LocalVectorStore


def call_model(messages, **kwargs) -> str:
    """Stand-in for the upstream LLM. Replace with a real client."""
    question = messages[-1]["content"]
    return f"(placeholder answer to {question!r})"


def stream_model(messages, **kwargs):
    """Stand-in for a streaming upstream call."""
    for word in call_model(messages, **kwargs).split(" "):
        yield word + " "


engine = SemanticCacheEngine(
    # The encoder, not the corpus: a chat proxy caches answers, it does not
    # retrieve documents. LocalVectorStore is used here only for its encode().
    LocalVectorStore(),
    threshold=0.85,
    max_entries=10_000,
    resolver="context",
    # Uncomment to raise the bar on numeric and polarity questions. Unmeasured
    # defaults -- see gateway/policy.py.
    # threshold_policy="intent",
    ttl=24 * 3600,
)

middleware = ChatCacheMiddleware(
    engine,
    call_model,
    stream_fn=stream_model,
    # The data boundary. An unauthenticated header is a placeholder for whatever
    # your auth layer already established.
    namespace_fn=lambda messages, kwargs: kwargs.get("namespace") or "anonymous",
)

app = FastAPI(title="vector-cache-gateway proxy")


@app.post("/v1/chat/completions")
async def chat(request: Request, x_user_id: str = Header(default="anonymous")):
    body = await request.json()
    messages = body.pop("messages", [])
    stream = body.pop("stream", False)

    if stream:
        # A cache hit is re-chunked into a synthetic stream by the middleware,
        # so both paths produce the same shape of response.
        chunks = middleware.stream(messages, namespace=x_user_id, **body)
        return StreamingResponse(
            (f"data: {chunk}\n\n" for chunk in chunks), media_type="text/event-stream"
        )

    result = middleware.complete(messages, namespace=x_user_id, **body)
    return JSONResponse(
        {"choices": [{"message": {"role": "assistant", "content": result.answer}}]},
        headers={
            "x-cache-status": result.status,
            "x-cache-similarity": "" if result.similarity is None else f"{result.similarity:.4f}",
            "x-cache-entry": result.entry_id or "",
        },
    )


@app.post("/v1/feedback")
async def feedback(request: Request):
    """Retract an answer a user rejected.

    The client sends back the entry id it got in ``x-cache-entry``. Anything
    else -- the query, the conversation id -- would mean searching again to
    decide what to delete, which can delete the wrong entry.
    """
    body = await request.json()
    evicted = middleware.feedback(body["entry_id"], helpful=body.get("helpful", False))
    return {"evicted": evicted}


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus exposition. Hit rate and mean hit similarity are the two to
    alert on: the first says whether the cache is saving anything, the second
    says whether it is saving it by cheating."""
    return PlainTextResponse(middleware.metrics.prometheus())


@app.get("/cache/stats")
async def stats() -> dict:
    return middleware.stats()

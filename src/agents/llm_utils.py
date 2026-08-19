import json
import logging
import os
import time
from typing import Any, Optional, Sequence


# ---------------------------------------------------------------------
# Suppress noisy gRPC / Google logs such as:
# I0504 ... ev_poll_posix.cc:593] FD from fork parent still in poll list
# These must be set before Google / gRPC clients are initialized.
# ---------------------------------------------------------------------
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_LOGGING_MIN_LOG_LEVEL", "2")

logging.getLogger("grpc").setLevel(logging.ERROR)
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)


# Optional google-genai imports for Gemini 3 thinking control
try:
    from google import genai
    from google.genai import types as genai_types
except Exception:
    genai = None
    genai_types = None


_CONFIG_CACHE: Optional[dict[str, Any]] = None


def _load_config() -> dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    config_path = os.getenv("EXEOS_CONFIG_PATH", "config.json")
    if not os.path.exists(config_path):
        _CONFIG_CACHE = {}
        return _CONFIG_CACHE

    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG_CACHE = json.load(f)
    return _CONFIG_CACHE


def _cfg(key: str, default: Optional[str] = None) -> Optional[str]:
    cfg = _load_config()
    v = cfg.get(key)
    return v if (v is not None and str(v) != "") else default


def _vertex_model_id_for_alias(model_id: str) -> str:
    """
    Map friendly aliases to actual Vertex model ids.
    """
    if model_id == "gemini-3-pro":
        return "gemini-3-pro-preview"
    return model_id


def _vertex_location_for_model(model_id: str) -> str:
    """
    Decide which Vertex location to use for a given model.
    """
    base_loc = _cfg("gcp_location", "us-central1")

    global_models = {
        "gemini-3-pro-preview",
        "gemini-3-pro",
        "gemini-3-pro-image-preview",
    }

    if model_id in global_models:
        return "global"

    return str(base_loc)


def _is_gemini3(model_id: str) -> bool:
    return model_id.startswith("gemini-3")


def _gemini3_thinking_level() -> str:
    raw = _cfg("gemini3_thinking_level", "LOW")
    if raw is None:
        return "LOW"

    s = str(raw).strip().upper()
    return "HIGH" if s == "HIGH" else "LOW"


def _require_import(name: str, exc: Exception) -> None:
    raise ImportError(
        f"Missing optional dependency '{name}'. Install/update it first."
    ) from exc


def _messages_to_text(messages: Sequence[Any]) -> str:
    parts: list[str] = []
    for m in messages:
        content = getattr(m, "content", "")
        if content:
            parts.append(str(content))
    return "\n".join(parts)


def get_llm(model: str):
    """
    Return a LangChain chat model.

    Supported model names:
      - gpt-4o, gpt-4o-mini, o4-mini, etc.
      - vertex-gemini-2.5-pro
      - vertex-gemini-3-pro
      - vertex-gemini-3-pro-preview
    """

    if model.startswith("gpt") or model in {"o4-mini", "o3", "o3-mini"}:
        try:
            from langchain_openai import ChatOpenAI
        except Exception as e:
            _require_import("langchain-openai", e)

        return ChatOpenAI(
            model=model,
            api_key=_cfg("openai_api_key"),
            organization=_cfg("openai_org_id"),
            temperature=0,
        )

    if model.startswith("vertex-"):
        try:
            import google.auth
            from langchain_google_genai import ChatGoogleGenerativeAI
        except Exception as e:
            _require_import("langchain-google-genai/google-auth", e)

        raw_id = model.replace("vertex-", "", 1).strip()
        model_id = _vertex_model_id_for_alias(raw_id)
        location = _vertex_location_for_model(model_id)

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        # Tell the new Google GenAI LangChain integration to use Vertex AI.
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
        if _cfg("gcp_project_id"):
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", str(_cfg("gcp_project_id")))
        if location:
            os.environ.setdefault("GOOGLE_CLOUD_LOCATION", str(location))

        return ChatGoogleGenerativeAI(
            model=model_id,
            vertexai=True,
            project=_cfg("gcp_project_id"),
            location=location,
            credentials=credentials,
            temperature=0,
        )

    raise ValueError(f"Unsupported model: {model}")


def _gemini3_call_via_genai(
    model_name: str,
    messages: Sequence[Any],
    use_logprobs: bool,
    log_dir: Optional[str],
) -> Optional[str]:
    """
    Handle Gemini 3 calls through the google-genai client.
    This keeps explicit thinking-level control for Gemini 3.
    """
    if genai is None or genai_types is None:
        print(
            "[DEBUG] google-genai SDK not available; "
            "falling back to default model invocation without explicit thinking control."
        )
        return None

    project_id = _cfg("gcp_project_id")
    location = _vertex_location_for_model(model_name)

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )

    text_prompt = _messages_to_text(messages)

    lvl = _gemini3_thinking_level()
    level_enum = (
        genai_types.ThinkingLevel.HIGH
        if lvl == "HIGH"
        else genai_types.ThinkingLevel.LOW
    )

    config = genai_types.GenerateContentConfig(
        thinking_config=genai_types.ThinkingConfig(
            thinking_level=level_enum
        )
    )

    response = client.models.generate_content(
        model=model_name,
        contents=text_prompt,
        config=config,
    )

    content = response.text or ""

    if use_logprobs and log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            usage = getattr(response, "usage_metadata", None)

            meta: dict[str, Any] = {
                "full_text": content,
                "model": model_name,
                "thinking_level": lvl,
            }

            if usage is not None:
                meta.update(
                    {
                        "prompt_token_count": getattr(usage, "prompt_token_count", None),
                        "candidates_token_count": getattr(
                            usage, "candidates_token_count", None
                        ),
                        "thoughts_token_count": getattr(
                            usage, "thoughts_token_count", None
                        ),
                        "total_token_count": getattr(usage, "total_token_count", None),
                    }
                )

            out_path = os.path.join(log_dir, f"gemini3_usage_{ts}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            print(f"[DEBUG] Gemini 3 usage metadata written to {out_path}")
        except Exception as e:
            print("[DEBUG] Failed to write Gemini 3 usage metadata:", e)

    return content


def llm_call(
    llm: Any,
    messages: Sequence[Any],
    use_logprobs: bool = False,
    log_dir: Optional[str] = None,
    top_logprobs: int = 5,
) -> str:
    """
    Wrapper for configured chat model providers.
    When use_logprobs=True and supported, saves token-level probabilities.
    """
    model_name = (
        getattr(llm, "model_name", "")
        or getattr(llm, "model", "")
        or getattr(llm, "model_id", "")
    )

    module_name = getattr(type(llm), "__module__", "").lower()
    class_name = getattr(type(llm), "__name__", "").lower()

    is_google_genai = (
        "google_genai" in module_name
        or "chatgooglegenerativeai" in class_name
    )
    is_openai = "openai" in module_name or "chatopenai" in class_name

    if is_google_genai and _is_gemini3(str(model_name)):
        content = _gemini3_call_via_genai(
            model_name=str(model_name),
            messages=messages,
            use_logprobs=use_logprobs,
            log_dir=log_dir,
        )
        if content is not None:
            return content

    if use_logprobs and is_openai:
        try:
            bound = llm.bind(logprobs=True, top_logprobs=top_logprobs)
            response = bound.invoke(messages)
            content = getattr(response, "content", "") or ""

            if log_dir:
                try:
                    os.makedirs(log_dir, exist_ok=True)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    out_path = os.path.join(log_dir, f"openai_logprobs_{ts}.json")

                    meta = {
                        "full_text": content,
                        "response_metadata": getattr(response, "response_metadata", {}),
                    }

                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2)

                    print(f"[DEBUG] OpenAI logprobs written to {out_path}")
                except Exception as e:
                    print("[DEBUG] Failed to write OpenAI logprobs:", e)

            return content
        except Exception as e:
            print("[DEBUG] OpenAI logprobs failed; falling back to invoke:", e)

    if use_logprobs and is_google_genai:
        print(
            "[DEBUG] Token logprobs are not handled through ChatGoogleGenerativeAI "
            "in this wrapper; falling back to normal invoke."
        )

    response = llm.invoke(messages)
    return getattr(response, "content", "") or str(response)

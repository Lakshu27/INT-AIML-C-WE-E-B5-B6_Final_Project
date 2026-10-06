"""Quick check that the LLM API call works (never prints your key)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coverwise import config  # noqa: E402
from coverwise.llm import LLMClient, LLMError  # noqa: E402

key = config.GROQ_API_KEY if config.LLM_PROVIDER == "groq" else config.GEMINI_API_KEY
env_file = config.ROOT_DIR / ".env"
print(f".env found       : {env_file.exists()} ({env_file})")
print(f"Provider         : {config.LLM_PROVIDER!r}")
print(f"Key loaded       : {'yes (' + key[:4] + '...' + key[-4:] + ')' if key else 'NO - check .env'}")
llm = LLMClient()
print(f"Client           : {llm.label}")
if not llm.available:
    sys.exit("LLM not available -> fix .env (file name, provider, key)")
try:
    t = time.time()
    reply = llm.chat("You are a test assistant.", "Reply with exactly: COVERWISE OK", max_tokens=20, retries=1)
    print(f"Plain call       : {reply.strip()!r} ({time.time() - t:.1f}s)")
    out = llm.chat_json("Return JSON only.", 'Return {"status": "ok", "copayment_percentage": 5}', max_tokens=60)
    print(f"JSON call        : {out}")
    print("\nAPI call works - the app will use the LLM.")
except LLMError as exc:
    msg = str(exc)
    print(f"\nAPI call FAILED: {msg[:300]}")
    if "401" in msg or "invalid_api_key" in msg.lower():
        print("-> Key invalid or revoked: create a new key and update GROQ_API_KEY in .env")
    elif "model" in msg.lower():
        print("-> Model problem: set another current model in GROQ_MODEL")
    elif "429" in msg:
        print("-> Rate limit: wait a minute and retry")
    else:
        print("-> Check internet / proxy / firewall")
    sys.exit(1)
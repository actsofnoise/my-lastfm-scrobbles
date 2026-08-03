"""
Test: Web Search Grounding Check

Verifies EMPIRICALLY whether each AI backend actually has web search
grounding enabled, before we build any pipeline logic that assumes it does.

Tests:
1. A time-sensitive question with a knowable, checkable answer (cherry
   blossom season dates) — if grounding works, the answer should look
   current and (for Gemini/Gemma) come with citation sources.
2. A real tracklist question (King Crimson - "Red") — the actual use case
   we care about for the album-matching pipeline.

For Gemini and Gemma: uses the google_search tool and checks whether
grounding_metadata.grounding_chunks actually comes back with real sources.
An empty/missing grounding_metadata means the model answered from its own
training data, NOT from a live search, even if the tool was requested.

For DeepSeek (via NVIDIA): there is no equivalent "google_search" tool in
the OpenAI-compatible chat completions spec that NVIDIA serves. This test
just asks the same questions and prints the raw answer so you can judge by
eye whether it looks current / correct, or looks like a memorized guess.
NVIDIA's endpoint does not expose an official way to confirm real browsing
happened (no citations returned), so treat this backend's result with a
lot more skepticism than Gemini/Gemma's citation-backed answers.
"""

import os
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

GEMINI_MODEL = "gemini-3.6-flash"
GEMMA_MODEL = "gemma-4-26b-a4b-it"  # per your snippet — double check this matches what you have access to
NVIDIA_MODEL = "deepseek-ai/deepseek-v4-flash"

TEST_PROMPTS = [
    "What are the dates for cherry blossom season in Tokyo this year?",
    "What is the complete tracklist of the album 'Red' by King Crimson, in track order? List just the song titles, numbered.",
]


def test_gemini_family(model_name: str, label: str):
    print(f"\n{'=' * 60}")
    print(f"TESTING: {label} ({model_name}) — Gemini API with google_search tool")
    print(f"{'=' * 60}")

    if not GEMINI_API_KEY:
        print("⚠️ GEMINI_API_KEY not set — skipping.")
        return

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("⚠️ google-genai not installed. Run: pip install google-genai")
        return

    client = genai.Client()

    for prompt in TEST_PROMPTS:
        print(f"\n--- Prompt: {prompt}")
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}]
                ),
            )
        except Exception as e:
            print(f"❌ Request failed: {e}")
            print("   (This likely means this model does NOT support the google_search tool at all.)")
            continue

        print(f"Answer: {response.text}")

        # Check grounding metadata — this is the real proof search happened
        try:
            candidate = response.candidates[0]
            grounding = getattr(candidate, 'grounding_metadata', None)
            chunks = getattr(grounding, 'grounding_chunks', None) if grounding else None

            if chunks:
                print(f"✅ GROUNDED — {len(chunks)} real source(s) cited:")
                for chunk in chunks:
                    web = getattr(chunk, 'web', None)
                    if web:
                        print(f"    - {web.title} — {web.uri}")
            else:
                print("⚠️ NOT GROUNDED — no citations returned. This answer came from the")
                print("   model's own training data, not a live search, even though the")
                print("   google_search tool was requested.")
        except Exception as e:
            print(f"⚠️ Could not read grounding metadata: {e}")


def test_nvidia_deepseek():
    print(f"\n{'=' * 60}")
    print(f"TESTING: DeepSeek V4 Flash (via NVIDIA) — no search tool available in this API")
    print(f"{'=' * 60}")

    if not NVIDIA_API_KEY:
        print("⚠️ NVIDIA_API_KEY not set — skipping.")
        return

    try:
        from openai import OpenAI
    except ImportError:
        print("⚠️ openai not installed. Run: pip install openai")
        return

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
        timeout=120.0
    )

    for prompt in TEST_PROMPTS:
        print(f"\n--- Prompt: {prompt}")
        try:
            completion = client.chat.completions.create(
                model=NVIDIA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                top_p=0.95,
                max_tokens=2048,
                extra_body={"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}},
                stream=False
            )
            answer = completion.choices[0].message.content
            print(f"Answer: {answer}")
            print("ℹ️  No citations/grounding metadata exist in this API to verify a live")
            print("   search actually happened — judge by eye whether this looks current")
            print("   or like a memorized/guessed answer.")
        except Exception as e:
            print(f"❌ Request failed: {e}")


if __name__ == "__main__":
    print("WEB SEARCH GROUNDING TEST")
    print("Checking whether each backend can genuinely verify facts via live search,")
    print("or whether it's still answering from memory regardless of what we ask for.")

    test_gemini_family(GEMINI_MODEL, "Gemini Flash")
    test_gemini_family(GEMMA_MODEL, "Gemma")
    test_nvidia_deepseek()

    print(f"\n{'=' * 60}")
    print("DONE. Read the ✅/⚠️ markers above for each backend before building")
    print("any pipeline logic that assumes web search is actually happening.")
    print(f"{'=' * 60}")

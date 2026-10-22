
def extract_structured_note(
    note_text: str,
    api_key: Optional[str] = None,
    force_demo: bool = False,
    model_name: Optional[str] = None,
) -> ExtractionResult:
    """
    Extract structured information from synthetic clinical notes.
    Uses Gemini Developer API when a key is provided.
    Falls back to the offline demo engine when requested.
    """

    # 1. Validate input
    cleaned_text = (note_text or "").strip()

    if not cleaned_text:
        return ExtractionResult(
            success=False,
            error_message="Input clinical note is empty.",
            is_demo_mode=True,
        )

    if len(cleaned_text) < 25:
        return ExtractionResult(
            success=False,
            error_message="Please provide a more detailed note (minimum 25 characters).",
            is_demo_mode=True,
        )

    # 2. Resolve API key and model
    effective_key = (api_key or "").strip() or get_gemini_api_key()
    effective_model = model_name or DEFAULT_GEMINI_MODEL

    # 3. Offline demo mode
    if force_demo or not effective_key:
        structured_data = _offline_demo_extractor(cleaned_text)

        return ExtractionResult(
            success=True,
            data=structured_data,
            is_demo_mode=True,
            model_used="offline-synthetic-heuristic-engine",
        )

    # 4. Gemini Developer API
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=effective_key)

        # Convert the Pydantic model schema to JSON text.
        # It is included in the prompt, NOT passed as response_schema.
        output_json_schema = json.dumps(
            StructuredClinicalNote.model_json_schema(),
            indent=2,
        )

        prompt = f"""
You are a clinical information structuring assistant.
Process ONLY the synthetic clinical note provided below.

Return one valid JSON object matching the requested structure.
Do not include Markdown fences or explanatory text outside the JSON.
Do not invent patient facts that are absent from the note.
Use null for unknown optional values and empty lists when appropriate.
This output is an experimental research demonstration, not a diagnosis.

Expected JSON structure:
{output_json_schema}

Synthetic clinical note:
{cleaned_text}
"""

        response = client.models.generate_content(
            model=effective_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        raw_text = (response.text or "").strip()

        if not raw_text:
            raise ValueError("Gemini returned an empty response.")

        # Handle a response wrapped in Markdown fences, if any.
        if raw_text.startswith("```"):
            raw_text = raw_text.removeprefix("```json").removeprefix("```")
            raw_text = raw_text.removesuffix("```").strip()

        parsed_dict = json.loads(raw_text)
        structured_data = StructuredClinicalNote.model_validate(parsed_dict)

        return ExtractionResult(
            success=True,
            data=structured_data,
            is_demo_mode=False,
            model_used=effective_model,
            raw_response=raw_text,
        )

    except Exception as e:
        return ExtractionResult(
            success=False,
            error_message=(
                f"Gemini API Error: {e}. "
                "You can enable Demo Mode in the sidebar to test without an API key."
            ),
            is_demo_mode=False,
        )

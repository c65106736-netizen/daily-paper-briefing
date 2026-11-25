models_to_try = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash"]
    try:
        for m in client.models.list():
            m_name = getattr(m, "name", "").replace("models/", "")
            if "flash" in m_name and "lite" not in m_name and "image" not in m_name and "audio" not in m_name:
                if m_name not in models_to_try:
                    models_to_try.append(m_name)
    except Exception:
        pass

    last_error = None
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.2
                )
            )
            print(f"성공적으로 사용된 모델: {model_name}")
            return json.loads(response.text).get("selected_papers", [])
        except Exception as e:
            last_error = e
            print(f"모델 {model_name} 호출 실패, 다음 모델 시도 중: {e}")
            continue

    raise RuntimeError(f"모든 Gemini 모델 호출 실패: {last_error}")

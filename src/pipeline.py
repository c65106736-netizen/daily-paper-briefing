import os
import json
import time
import re
from datetime import datetime, timezone, timedelta
import requests
from google import genai
from google.genai import types

KST = timezone(timedelta(hours=9))
NOW_DT = datetime.now(KST)
NOW_KST = NOW_DT.strftime("%Y-%m-%d %H:%M KST")
COOLDOWN_DAYS = 30  # 한 달(30일) 뒤 재추천 허용

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def fetch_semantic_scholar(keyword, limit=15):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": keyword,
        "limit": limit,
        "fields": "paperId,title,abstract,url,year,venue,citationCount,authors,externalIds,openAccessPdf"
    }
    for attempt in range(3):
        try:
            res = requests.get(url, params=params, timeout=15)
            if res.status_code == 200:
                return res.json().get("data", [])
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return []

def fetch_openalex(keyword, limit=15):
    url = "https://api.openalex.org/works"
    params = {
        "search": keyword,
        "per-page": limit,
        "filter": "has_abstract:true",
        "sort": "cited_by_count:desc"
    }
    try:
        res = requests.get(url, params=params, timeout=15)
        if res.status_code != 200:
            return []
        results = []
        for item in res.json().get("results", []):
            inv_idx = item.get("abstract_inverted_index") or {}
            if not inv_idx:
                continue
            words = {}
            for word, positions in inv_idx.items():
                for pos in positions:
                    words[pos] = word
            abstract = " ".join(words[i] for i in sorted(words.keys()))
            loc = item.get("primary_location") or {}
            source = loc.get("source") or {}
            authors = [a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])[:5]]
            doi = (item.get("doi") or "").replace("https://doi.org/", "")
            results.append({
                "paperId": item.get("id", "").split("/")[-1],
                "title": item.get("title", ""),
                "abstract": abstract,
                "url": item.get("doi") or loc.get("landing_page_url") or "",
                "year": item.get("publication_year"),
                "venue": source.get("display_name") or "OpenAlex Indexed Journal",
                "citationCount": item.get("cited_by_count", 0),
                "authors": [{"name": name} for name in authors if name],
                "externalIds": {"DOI": doi} if doi else {},
                "openAccessPdf": {"url": loc.get("pdf_url")} if loc.get("pdf_url") else None
            })
        return results
    except Exception:
        return []

def analyze_with_gemini(client, keyword, candidates, target_count=5):
    prompt = f"""
당신은 글로벌 전략컨설팅 펌 및 학술 리서치 기관의 수석 애널리스트입니다.
사용자 관심 키워드: "{keyword}"

아래 후보 논문 목록에서 학술적 전문성, 저널 권위도, 방법론의 엄밀성, 기업 실무 적용 가치를 종합 평가하여 최상위 {target_count}편을 엄선하고 심층 분석 리포트를 작성하세요.

[후보 논문 데이터]
{json.dumps(candidates, ensure_ascii=False)}

[작성 규칙]
1. 정확히 {target_count}편을 선정하세요 (후보가 적으면 있는 만큼 선정).
2. total_score는 0~100점 정수로 매기고, journal_authority(저널 권위), methodology_rigor(방법론 엄밀성), practical_impact(실무 파급력) 세부 점수도 각각 0~100점으로 평가하세요.
3. 모든 분석 항목(korean_title, core_hypothesis, methodology, key_findings, strategic_insight, limitations, evaluation_comment)은 전문가 수준의 깊이 있는 한국어로 작성하세요.
4. original_abstract는 후보 데이터의 영문 abstract 원문을 수정 없이 그대로 포함하세요.
5. apa_citation은 정확한 APA 7판 형식의 영문 인용 문구로 작성하세요.
"""
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "selected_papers": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "paper_id": {"type": "STRING"},
                        "original_title": {"type": "STRING"},
                        "korean_title": {"type": "STRING"},
                        "authors": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "venue": {"type": "STRING"},
                        "year": {"type": "INTEGER"},
                        "citation_count": {"type": "INTEGER"},
                        "doi": {"type": "STRING"},
                        "url": {"type": "STRING"},
                        "pdf_url": {"type": "STRING"},
                        "total_score": {"type": "INTEGER"},
                        "journal_authority": {"type": "INTEGER"},
                        "methodology_rigor": {"type": "INTEGER"},
                        "practical_impact": {"type": "INTEGER"},
                        "evaluation_comment": {"type": "STRING"},
                        "core_hypothesis": {"type": "STRING"},
                        "methodology": {"type": "STRING"},
                        "key_findings": {"type": "STRING"},
                        "strategic_insight": {"type": "STRING"},
                        "limitations": {"type": "STRING"},
                        "tags": {"type": "ARRAY", "items": {"type": "STRING"}},
                        "original_abstract": {"type": "STRING"},
                        "apa_citation": {"type": "STRING"}
                    },
                    "required": [
                        "paper_id", "original_title", "korean_title", "authors", "venue",
                        "year", "citation_count", "url", "total_score", "journal_authority",
                        "methodology_rigor", "practical_impact", "evaluation_comment",
                        "core_hypothesis", "methodology", "key_findings", "strategic_insight",
                        "limitations", "tags", "original_abstract", "apa_citation"
                    ]
                }
            }
        },
        "required": ["selected_papers"]
    }

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

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

    config = load_json("keywords.json", {"keyword": "corporate strategy", "papers_per_day": 5})
    keyword = config.get("keyword", "corporate strategy")
    target_count = int(config.get("papers_per_day", 5))

    # seen_records 형식: { "paper_id": "2026-11-25T14:30:00+09:00", ... }
    history_data = load_json("data/history.json", {"seen_records": {}})
    seen_records = history_data.get("seen_records", {})
    if isinstance(history_data.get("seen_ids"), list) and not seen_records:
        for old_id in history_data["seen_ids"]:
            seen_records[old_id] = NOW_DT.isoformat()

    # 30일이 지난 기록은 자동 만료시켜 한 달 뒤 재추천 가능하도록 정리
    active_seen_records = {}
    for pid, ts_str in seen_records.items():
        try:
            seen_dt = datetime.fromisoformat(ts_str)
            if (NOW_DT - seen_dt).days < COOLDOWN_DAYS:
                active_seen_records[pid] = ts_str
        except Exception:
            active_seen_records[pid] = NOW_DT.isoformat()

    raw_papers = fetch_semantic_scholar(keyword, limit=15) + fetch_openalex(keyword, limit=15)

    unique_candidates = []
    batch_titles = set()
    for p in raw_papers:
        pid = str(p.get("paperId") or "")
        title = (p.get("title") or "").strip()
        abstract = (p.get("abstract") or "").strip()
        norm_title = re.sub(r"\W+", "", title.lower())

        if not pid or not title or len(abstract) < 100:
            continue
        if pid in active_seen_records or norm_title in batch_titles:
            continue

        batch_titles.add(norm_title)
        authors = [a.get("name", "") for a in (p.get("authors") or [])[:5] if a.get("name")]
        doi = (p.get("externalIds") or {}).get("DOI", "")
        pdf_url = (p.get("openAccessPdf") or {}).get("url") or ""
        url = p.get("url") or (f"https://doi.org/{doi}" if doi else "")

        unique_candidates.append({
            "paper_id": pid,
            "title": title,
            "abstract": abstract[:2500],
            "authors": authors,
            "venue": p.get("venue") or "Academic Journal / Working Paper",
            "year": p.get("year") or 2025,
            "citation_count": int(p.get("citationCount") or 0),
            "doi": doi,
            "url": url,
            "pdf_url": pdf_url
        })

    unique_candidates.sort(key=lambda x: (x["citation_count"], x["year"]), reverse=True)
    top_candidates = unique_candidates[:15]

    if not top_candidates:
        print("새로운 후보 논문이 없습니다.")
        return

    client = genai.Client(api_key=api_key)
    selected = analyze_with_gemini(client, keyword, top_candidates, target_count=target_count)

    for item in selected:
        active_seen_records[item["paper_id"]] = NOW_DT.isoformat()

    latest_payload = {
        "updated_at": NOW_KST,
        "keyword": keyword,
        "count": len(selected),
        "papers": selected
    }
    save_json("data/latest.json", latest_payload)

    archive = load_json("data/archive.json", [])
    archive.insert(0, latest_payload)
    save_json("data/archive.json", archive[:60])
    save_json("data/history.json", {"seen_records": active_seen_records})
    print(f"완료: '{keyword}' 키워드로 {len(selected)}편 분석 및 저장 완료 (30일 쿨다운 적용)")

if __name__ == "__main__":
    main()

import re
from typing import List
import httpx
from models import HFResourceRecord
from services.query_utils import extract_core_scientific_terms

HF_MODELS_API = "https://huggingface.co/api/models"
HF_DATASETS_API = "https://huggingface.co/api/datasets"


def _extract_hf_keywords(query: str) -> List[str]:
    words = extract_core_scientific_terms(query, max_terms=5)
    if not words:
        return ["protein"]

    queries = []
    # 1. Primary 2-keyword domain phrase (e.g. "antimicrobial peptide")
    if len(words) >= 2:
        queries.append(" ".join(words[:2]))
    # 2. Primary 3-keyword phrase if available
    if len(words) >= 3:
        queries.append(" ".join(words[:3]))
    # 3. Individual high-value domain tokens (length >= 3)
    for w in words[:4]:
        if len(w) >= 3 and w not in queries:
            queries.append(w)
    return queries[:4]


async def search_huggingface_hub(
    query: str,
    arxiv_ids: List[str] = None,
    hf_api_key: str = "",
    max_models: int = 10,
    max_datasets: int = 5,
    hf_facet_queries: List[str] = None,
) -> List[HFResourceRecord]:
    headers = {"User-Agent": "CiteLume/1.0"}
    if hf_api_key and hf_api_key.strip():
        headers["Authorization"] = f"Bearer {hf_api_key.strip()}"

    seen_ids = set()
    resources: List[HFResourceRecord] = []
    kw_queries = []
    if hf_facet_queries:
        for fq in hf_facet_queries:
            clean_fq = fq.strip()
            if clean_fq and clean_fq not in kw_queries:
                kw_queries.append(clean_fq)
    for eq in _extract_hf_keywords(query):
        if eq not in kw_queries:
            kw_queries.append(eq)
    kw_queries = kw_queries[:5]

    async with httpx.AsyncClient(timeout=15.0, headers=headers, follow_redirects=True) as client:
        # 1. First check models directly linked to retrieved arXiv papers
        if arxiv_ids:
            for aid in arxiv_ids[:6]:
                clean_aid = re.sub(r"v\d+$", "", aid.strip())
                try:
                    resp = await client.get(
                        HF_MODELS_API,
                        params={"filter": f"arxiv:{clean_aid}", "sort": "downloads", "direction": "-1", "limit": "3"},
                    )
                    if resp.status_code == 200:
                        for item in resp.json():
                            mid = item.get("id") or item.get("modelId")
                            if not mid or mid in seen_ids:
                                continue
                            seen_ids.add(mid)
                            tags = item.get("tags") or []
                            linked_arxiv = [t.split("arxiv:")[-1] for t in tags if t.startswith("arxiv:")]
                            resources.append(
                                HFResourceRecord(
                                    id=mid,
                                    resource_type="model",
                                    pipeline_tag=item.get("pipeline_tag") or "custom-model",
                                    downloads=int(item.get("downloads") or 0),
                                    likes=int(item.get("likes") or 0),
                                    author=mid.split("/")[0] if "/" in mid else "huggingface",
                                    description=f"Directly linked to paper arXiv:{clean_aid}. Pipeline: {item.get('pipeline_tag') or 'scientific/ML model'}.",
                                    tags=tags[:10],
                                    linked_arxiv_ids=linked_arxiv,
                                    url=f"https://huggingface.co/{mid}",
                                    last_modified=str(item.get("lastModified") or "")[:10],
                                )
                            )
                except Exception:
                    pass

        # 2. Search HF models by topic keywords
        for kw in kw_queries:
            if len([r for r in resources if r.resource_type == "model"]) >= max_models:
                break
            try:
                resp = await client.get(
                    HF_MODELS_API,
                    params={"search": kw, "sort": "downloads", "direction": "-1", "limit": str(max_models)},
                )
                if resp.status_code == 200:
                    for item in resp.json():
                        mid = item.get("id") or item.get("modelId")
                        if not mid or mid in seen_ids:
                            continue
                        seen_ids.add(mid)
                        tags = item.get("tags") or []
                        linked_arxiv = [t.split("arxiv:")[-1] for t in tags if t.startswith("arxiv:")]
                        pipeline = item.get("pipeline_tag") or (item.get("library_name") or "model")
                        resources.append(
                            HFResourceRecord(
                                id=mid,
                                resource_type="model",
                                pipeline_tag=pipeline,
                                downloads=int(item.get("downloads") or 0),
                                likes=int(item.get("likes") or 0),
                                author=mid.split("/")[0] if "/" in mid else "community",
                                description=f"Hugging Face {pipeline} model matched for '{kw}'.",
                                tags=[t for t in tags if not t.startswith("arxiv:")][:8],
                                linked_arxiv_ids=linked_arxiv,
                                url=f"https://huggingface.co/{mid}",
                                last_modified=str(item.get("lastModified") or "")[:10],
                            )
                        )
                        if len([r for r in resources if r.resource_type == "model"]) >= max_models:
                            break
            except Exception as exc:
                print(f"[HF Models Scraper Warning] {exc}")

        # 3. Search HF datasets by topic keywords
        for kw in kw_queries[:2]:
            if len([r for r in resources if r.resource_type == "dataset"]) >= max_datasets:
                break
            try:
                resp = await client.get(
                    HF_DATASETS_API,
                    params={"search": kw, "sort": "downloads", "direction": "-1", "limit": str(max_datasets)},
                )
                if resp.status_code == 200:
                    for item in resp.json():
                        did = item.get("id")
                        if not did or did in seen_ids:
                            continue
                        seen_ids.add(did)
                        tags = item.get("tags") or []
                        linked_arxiv = [t.split("arxiv:")[-1] for t in tags if t.startswith("arxiv:")]
                        desc = (item.get("description") or f"Hugging Face dataset matched for '{kw}'.").strip()
                        desc = re.sub(r"\s+", " ", desc)[:200]
                        resources.append(
                            HFResourceRecord(
                                id=did,
                                resource_type="dataset",
                                pipeline_tag="dataset",
                                downloads=int(item.get("downloads") or 0),
                                likes=int(item.get("likes") or 0),
                                author=did.split("/")[0] if "/" in did else "community",
                                description=desc,
                                tags=[t for t in tags if not t.startswith("arxiv:")][:8],
                                linked_arxiv_ids=linked_arxiv,
                                url=f"https://huggingface.co/datasets/{did}",
                                last_modified=str(item.get("lastModified") or "")[:10],
                            )
                        )
                        if len([r for r in resources if r.resource_type == "dataset"]) >= max_datasets:
                            break
            except Exception as exc:
                print(f"[HF Datasets Scraper Warning] {exc}")

    return resources

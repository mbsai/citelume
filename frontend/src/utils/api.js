const API_BASE = '';

export async function runLiteratureSearch(payload) {
  const response = await fetch(`${API_BASE}/api/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const errText = await response.text();
    let detail = errText;
    try {
      detail = JSON.parse(errText).detail || errText;
    } catch (_) {}
    // Demo limits and busy messages are written for people, so show them as they are
    if ([400, 429, 503].includes(response.status) && typeof detail === 'string') throw new Error(detail);
    throw new Error(`Search failed (${response.status}): ${typeof detail === 'string' ? detail : errText}`);
  }
  return response.json();
}

export async function fetchDemoStatus() {
  try {
    const response = await fetch(`${API_BASE}/api/demo-status`);
    if (!response.ok) return { demo: false };
    return response.json();
  } catch (_) {
    return { demo: false };
  }
}

export async function fetchModelScale(provider, model) {
  const params = new URLSearchParams({ provider: provider || '', model: model || '' });
  const response = await fetch(`${API_BASE}/api/model-scale?${params.toString()}`);
  if (!response.ok) return { tier: 'unknown', reason: '' };
  return response.json();
}

export async function fetchSearchProgress(progressId) {
  const response = await fetch(`${API_BASE}/api/search-progress/${encodeURIComponent(progressId)}`);
  if (!response.ok) return null;
  return response.json();
}

export async function expandPaperNode(payload) {
  const response = await fetch(`${API_BASE}/api/expand-node`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`Node expansion failed (${response.status})`);
  }
  return response.json();
}

export async function fetchAvailableHfModels(hfApiKey = '', search = '') {
  const params = new URLSearchParams();
  if (hfApiKey) params.set('hf_api_key', hfApiKey);
  if (search) params.set('search', search);
  const response = await fetch(`${API_BASE}/api/hf-models?${params.toString()}`);
  if (!response.ok) return { models: [] };
  return response.json();
}

export async function fetchProviderModels(payload) {
  const response = await fetch(`${API_BASE}/api/provider-models`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) return { models: [], live_fetched: false };
  return response.json();
}

export async function testLlmApiKey(payload) {
  const response = await fetch(`${API_BASE}/api/test-llm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    return { success: false, message: `Server error (${response.status})` };
  }
  return response.json();
}

export async function testSourceApiKey(source, apiKey = '') {
  const response = await fetch(`${API_BASE}/api/test-source`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source, api_key: apiKey }),
  });
  if (!response.ok) {
    return { success: false, message: `Server error (${response.status})` };
  }
  return response.json();
}

export async function downloadReportFromServer(format, researchData, exportOptions = {}) {
  const payload = {
    format,
    query: researchData.query,
    mode: researchData.mode,
    executive_answer: researchData.executive_answer,
    sections: researchData.sections || [],
    consensus: researchData.consensus || {},
    papers: researchData.papers || [],
    bibliography: researchData.bibliography || [],
    gap_matrix: researchData.gap_matrix || null,
    graph: researchData.graph || null,
    hf_resources: researchData.hf_resources || [],
    review_protocol: researchData.review_protocol || null,
    include_evidence_table: exportOptions.include_evidence_table !== false,
    include_gap_matrix: exportOptions.include_gap_matrix !== false,
    include_network_summary: exportOptions.include_network_summary !== false,
    include_hf_models: exportOptions.include_hf_models !== false,
  };

  const response = await fetch(`${API_BASE}/api/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`Export failed (${response.status})`);
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const safeSlug = (researchData.query || 'synthesis')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .slice(0, 40);
  a.href = url;
  a.download = `literature_report_${researchData.mode}_${safeSlug}.${format}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

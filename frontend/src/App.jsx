import React, { useState, useEffect } from 'react';
import ApiConfigModal from './components/ApiConfigModal';
import SynthesisPanel from './components/SynthesisPanel';
import EvidenceTable from './components/EvidenceTable';
import LiteratureGraph from './components/LiteratureGraph';
import KnowledgeGapMatrix from './components/KnowledgeGapMatrix';
import HuggingFacePanel from './components/HuggingFacePanel';
import { runLiteratureSearch, fetchSearchProgress, fetchModelScale, fetchDemoStatus } from './utils/api';

// Default LLM: NVIDIA Nemotron 3 Ultra (free) on OpenRouter, the model the public demo runs without a key.
// Bump CONFIG_VERSION when the default LLM changes, so saved settings that were never set by hand move to it.
const CONFIG_VERSION = 2;
const DEFAULT_LLM = {
  llm_provider: 'openrouter',
  llm_model: 'nvidia/nemotron-3-ultra-550b-a55b:free',
  llm_base_url: '',
  llm_fallback_models: ['nvidia/nemotron-3.5-super-vl:free', 'nvidia/nemotron-3.5-lightning:free'],
};
const DEFAULT_CONFIG = {
  ...DEFAULT_LLM,
  config_version: CONFIG_VERSION,
  llm_api_key: '',
  provider_keys: {},
  provider_models: {},
  provider_base_urls: {},
  pubmed_api_key: '',
  s2_api_key: '',
  hf_api_key: '',
  arxiv_api_key: '',
  biorxiv_api_key: '',
  enable_pubmed: true,
  enable_arxiv: true,
  enable_biorxiv: true,
  enable_s2: true,
  enable_hf: true,
  enable_fulltext: true,
};

// Field selector: '' lets the backend auto-detect the domain from the query
const DOMAIN_OPTIONS = [
  ['', 'Auto-detect from question'],
  ['biomedical_clinical', 'Biomedical & Clinical'],
  ['chemistry', 'Chemistry'],
  ['physics', 'Physics'],
  ['mathematics', 'Mathematics'],
  ['computer_science_ai', 'Computer Science & AI'],
  ['engineering_materials', 'Engineering & Materials'],
  ['social_sciences', 'Social Sciences'],
  ['humanities_interdisciplinary', 'Humanities'],
  ['interdisciplinary_biomedical_ai', 'Interdisciplinary Biomedical AI'],
];

// Deep Research review frameworks (how the review is reported) and body organisations (how findings are arranged)
const REVIEW_FRAMEWORK_OPTIONS = [
  ['narrative', 'Narrative review (SANRA)'],
  ['systematic_prisma', 'Systematic-style (PRISMA 2020)'],
  ['scoping', 'Scoping review (PRISMA-ScR)'],
];
const ORGANIZATION_OPTIONS = [
  ['thematic', 'Thematic'],
  ['chronological', 'Chronological'],
  ['methodological', 'Methodological'],
  ['theoretical', 'Theoretical'],
];

// Shown until the backend reports its first stage via /api/search-progress
const INITIAL_PROGRESS = { stage: 0, total: 6, label: 'Starting search...', detail: '' };

const formatElapsed = (secs) => {
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return m > 0 ? `${m}m ${String(s).padStart(2, '0')}s` : `${s}s`;
};

// Strips "Attempt N: ..." wrappers so the banner shows the provider's actual reason
const failureReason = (task) => {
  const last = (task.validation_errors || []).slice(-1)[0] || 'Unknown error';
  const idx = last.indexOf('Reason: ');
  return idx >= 0 ? last.slice(idx + 8) : last.replace(/^Attempt \d+[^:]*:\s*/, '');
};

// Tells the user when AI steps fell back to offline extraction or were answered by a backup model
function LlmStatusBanner({ auditTrail }) {
  const llmTasks = auditTrail.filter((t) => t.model_used !== 'deterministic_offline_engine');
  if (auditTrail.length > 0 && llmTasks.length === 0) {
    return (
      <div className="bg-slate-50 border border-slate-300 text-slate-700 rounded-xl p-3 text-xs">
        No LLM is configured, so this report was built with offline (rule-based) extraction only. Add an API key
        in ⚙ API Keys & Sources Setup for AI-written synthesis.
      </div>
    );
  }
  const degraded = llmTasks.filter((t) => t.degraded_to_deterministic);
  const fallbackNotes = [
    ...new Set(
      llmTasks.flatMap((t) => (t.validation_errors || []).filter((e) => e.includes('answered by fallback model')))
    ),
  ];
  if (degraded.length === 0 && fallbackNotes.length === 0) return null;
  const reasons = [...new Set(degraded.map(failureReason))];
  return (
    <div
      className={`rounded-xl p-3 text-xs border ${
        degraded.length > 0 ? 'bg-amber-50 border-amber-300 text-amber-900' : 'bg-sky-50 border-sky-300 text-sky-900'
      }`}
    >
      {degraded.length > 0 && (
        <>
          <p className="font-semibold">
            ⚠ {degraded.length} of {llmTasks.length} AI steps fell back to offline (rule-based) extraction, so parts of
            this report are less detailed.
          </p>
          <ul className="mt-1 list-disc pl-5 space-y-0.5">
            {reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
          <p className="mt-1 text-amber-800">
            Try again shortly, pick another model, or add backup models in ⚙ API Keys & Sources Setup.
          </p>
        </>
      )}
      {fallbackNotes.map((n) => (
        <p key={n} className={degraded.length > 0 ? 'mt-1' : ''}>
          ↪ {n}
        </p>
      ))}
    </div>
  );
}

// Which models suit Deep Research; small models are held back until the user explicitly opts in
function DeepResearchModelNote({ model, hasLlm, scale, override, onOverride }) {
  return (
    <div
      className={`w-full rounded-lg border px-3 py-2 text-xs ${
        scale.tier === 'small' && hasLlm ? 'bg-amber-50 border-amber-300 text-amber-900' : 'bg-purple-50 border-purple-200 text-purple-900'
      }`}
    >
      <p>
        <span className="font-semibold">Deep Research writes a long academic review (2,500+ words) from up to 60 studies.</span>{' '}
        Use a large model with a long context window: Claude Sonnet/Opus, GPT-4o/4.1, Gemini 2.5 Pro, DeepSeek V3/R1,
        Llama 3.3 70B, Qwen 2.5 72B, or the free NVIDIA Nemotron Super/Ultra on OpenRouter.
      </p>
      {!hasLlm && <p className="mt-1">No LLM is configured, so the review will be built with rule-based extraction only.</p>}
      {hasLlm && scale.tier === 'small' && (
        <div className="mt-1.5 space-y-1">
          <p className="font-semibold">
            Your current model ({model}, {scale.reason}) is small for Deep Research: expect a shorter, less detailed review.
          </p>
          <label className="inline-flex items-center gap-1.5 cursor-pointer">
            <input type="checkbox" checked={override} onChange={(e) => onOverride(e.target.checked)} />
            Run anyway with this model
          </label>
        </div>
      )}
    </div>
  );
}

// Relevance screening and full-text coverage, so users can see which studies informed the synthesis
function ScreeningNotes({ warnings, papers }) {
  const relevance = warnings.filter(
    (w) => w.startsWith('Query interpreted') || w.startsWith('Relevance screening') || w.startsWith('Source unavailable') ||
      w.startsWith('Model guidance')
  );
  const fullText = papers.filter((p) => p.full_text_source).length;
  if (!relevance.length && !papers.length) return null;
  return (
    <div className="bg-sky-50 border border-sky-200 text-sky-900 rounded-xl p-3 text-xs space-y-1">
      <p>
        Open-access full text was used for {fullText} of {papers.length} studies; the rest were read from abstracts
        (paywalled articles are not retrieved).
      </p>
      {relevance.map((w) => (
        <p key={w}>{w}</p>
      ))}
    </div>
  );
}

function DemoBanner({ demo, hasOwnKey, onOpenConfig }) {
  if (!demo || !demo.demo) return null;
  const r = demo.remaining || {};
  const l = demo.limits || {};
  return (
    <div className="rounded-xl border border-indigo-200 bg-indigo-50 px-4 py-3 text-xs text-indigo-900 space-y-1">
      <div className="font-semibold">Free public demo</div>
      {hasOwnKey ? (
        <div>You are using your own LLM key, so the daily demo limits do not apply. Your key is sent with each search and is not stored on the server.</div>
      ) : (
        <div>
          Runs on NVIDIA's free models via OpenRouter. Left today from this address:{' '}
          <strong>{r.standard ?? 0} of {l.standard ?? 3} Standard</strong> and{' '}
          <strong>{r.deep_research ?? 0} of {l.deep_research ?? 1} Deep Research</strong> (resets {demo.resets || '00:00 UTC'}).{' '}
          <button type="button" onClick={onOpenConfig} className="underline font-medium">Add your own LLM key</button> for unlimited use.
        </div>
      )}
      {!demo.server_key_configured && !hasOwnKey && (
        <div className="text-amber-800">The shared demo key is not set up yet, so a search needs your own LLM key.</div>
      )}
    </div>
  );
}

export default function App() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [configOpen, setConfigOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState('standard'); // 'standard' | 'deep_research'
  const [domain, setDomain] = useState('');
  const [reviewFramework, setReviewFramework] = useState('narrative');
  const [organization, setOrganization] = useState('thematic');
  // Deep Research model check: scale of the configured model, and the user's explicit override for small models
  const [modelScale, setModelScale] = useState({ tier: 'unknown', reason: '' });
  const [deepOverride, setDeepOverride] = useState(false);
  const [activeTab, setActiveTab] = useState('synthesis');
  const [loading, setLoading] = useState(false);
  const [progressId, setProgressId] = useState(null);
  const [progress, setProgress] = useState(INITIAL_PROGRESS);
  const [elapsedSecs, setElapsedSecs] = useState(0);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  // Public demo (Hugging Face Space): shared NVIDIA free models with daily per-address limits
  const [demo, setDemo] = useState({ demo: false });
  const refreshDemo = () => fetchDemoStatus().then(setDemo);
  useEffect(() => {
    refreshDemo();
  }, []);

  // Query History state with saved results
  const [history, setHistory] = useState([]);
  const [activeHistoryId, setActiveHistoryId] = useState(null);

  useEffect(() => {
    try {
      // One-time move of settings and history saved under the project's former name
      [['litmap_studio_config', 'citelume_config'], ['litmap_studio_history', 'citelume_history']].forEach(([oldKey, newKey]) => {
        const oldVal = localStorage.getItem(oldKey);
        if (oldVal !== null && localStorage.getItem(newKey) === null) localStorage.setItem(newKey, oldVal);
      });
      const saved = localStorage.getItem('citelume_config');
      if (saved) {
        let parsed = JSON.parse(saved);
        // Settings saved before the default changed, with no LLM key entered, were never chosen by hand:
        // move them to the current default model
        if ((parsed.config_version || 1) < CONFIG_VERSION) {
          if (!parsed.llm_api_key) parsed = { ...parsed, ...DEFAULT_LLM };
          parsed.config_version = CONFIG_VERSION;
          localStorage.setItem('citelume_config', JSON.stringify(parsed));
        }
        setConfig((prev) => ({ ...prev, ...parsed }));
      }
      const savedHist = localStorage.getItem('citelume_history');
      if (savedHist) {
        const parsed = JSON.parse(savedHist);
        if (Array.isArray(parsed)) {
          setHistory(parsed);
        }
      }
    } catch (e) {
      console.error(e);
    }
  }, []);

  useEffect(() => {
    if (mode !== 'deep_research') return;
    setDeepOverride(false);
    const onDemoModel = demo.demo && !config.llm_api_key;
    fetchModelScale(onDemoModel ? 'openrouter' : config.llm_provider, onDemoModel ? demo.default_model : config.llm_model)
      .then(setModelScale)
      .catch(() => setModelScale({ tier: 'unknown', reason: '' }));
  }, [mode, config.llm_provider, config.llm_model, config.llm_api_key, demo.demo, demo.default_model]);

  const hasLlm = !!(
    config.llm_api_key ||
    ['ollama', 'custom'].includes(config.llm_provider) ||
    (demo.demo && demo.server_key_configured)
  );
  const deepBlocked = mode === 'deep_research' && hasLlm && modelScale.tier === 'small' && !deepOverride;

  // Poll the backend for the real pipeline stage while the search runs, and tick an elapsed timer
  useEffect(() => {
    if (!loading || !progressId) return;
    const startedAt = Date.now();
    setProgress(INITIAL_PROGRESS);
    setElapsedSecs(0);
    const poll = setInterval(async () => {
      try {
        const p = await fetchSearchProgress(progressId);
        // Stage numbers only move forward, so a stale poll response can never move the indicator back
        if (p && p.stage > 0) setProgress((prev) => (p.stage >= prev.stage ? p : prev));
      } catch (_) {}
    }, 1000);
    const tick = setInterval(() => setElapsedSecs(Math.floor((Date.now() - startedAt) / 1000)), 1000);
    return () => {
      clearInterval(poll);
      clearInterval(tick);
    };
  }, [loading, progressId]);

  const saveHistoryToStorage = (nextHistory) => {
    setHistory(nextHistory);
    try {
      localStorage.setItem('citelume_history', JSON.stringify(nextHistory.slice(0, 12)));
    } catch (e) {
      // If localStorage quota is tight, keep the most recent 5 full results
      try {
        localStorage.setItem('citelume_history', JSON.stringify(nextHistory.slice(0, 5)));
      } catch (_) {}
    }
  };

  const handleSearch = async (e) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;
    // A topic needs at least two letters; the server gives fuller advice for vaguer queries
    if (!/\p{L}{2,}/u.test(query) && !/\d{4,}/.test(query)) {
      setError('Please type a research topic, for example "statins after myocardial infarction", or paste a DOI or PubMed link.');
      return;
    }
    const pid = `search_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
    setProgressId(pid);
    setLoading(true);
    setError('');
    try {
      const data = await runLiteratureSearch({
        query: query.trim(),
        mode,
        max_per_source: mode === 'deep_research' ? 18 : 10,
        config,
        progress_id: pid,
        domain: domain || null,
        // Review types apply to Deep Research; Standard keeps the default narrative, thematic synthesis
        review_framework: mode === 'deep_research' ? reviewFramework : 'narrative',
        organization: mode === 'deep_research' ? organization : 'thematic',
      });
      setResult(data);

      const entryId = `hist_${Date.now()}`;
      const newEntry = {
        id: entryId,
        query: query.trim(),
        mode,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', month: 'short', day: 'numeric' }),
        paperCount: data.papers?.length || 0,
        hfCount: data.hf_resources?.length || 0,
        verdict: data.consensus?.consensus_verdict || 'Synthesized',
        resultData: data,
      };
      setActiveHistoryId(entryId);
      const filteredPrev = history.filter(
        (h) => !(h.query.toLowerCase() === newEntry.query.toLowerCase() && h.mode === newEntry.mode)
      );
      saveHistoryToStorage([newEntry, ...filteredPrev].slice(0, 15));
    } catch (err) {
      setError(err.message || 'Failed to run literature search');
    } finally {
      setLoading(false);
      if (demo.demo) refreshDemo();
    }
  };

  const handleSelectHistoryItem = (item) => {
    if (!item || !item.resultData) return;
    setQuery(item.query);
    setMode(item.mode || 'standard');
    setResult(item.resultData);
    setActiveHistoryId(item.id);
    setError('');
  };

  const handleDeleteHistoryItem = (id, e) => {
    if (e) e.stopPropagation();
    const next = history.filter((h) => h.id !== id);
    saveHistoryToStorage(next);
    if (activeHistoryId === id) {
      setActiveHistoryId(null);
    }
  };

  const handleClearHistory = () => {
    saveHistoryToStorage([]);
    setActiveHistoryId(null);
  };

  const handleGraphExpand = (parentNode, neighbors) => {
    if (!result || !neighbors || neighbors.length === 0) return;
    setResult((prev) => {
      const existingIds = new Set(prev.graph.nodes.map((n) => n.id));
      const newNodes = [...prev.graph.nodes];
      const newEdges = [...prev.graph.edges];

      neighbors.forEach((nb, idx) => {
        if (!existingIds.has(nb.id)) {
          existingIds.add(nb.id);
          const angle = (idx / neighbors.length) * Math.PI * 2;
          newNodes.push({
            id: nb.id,
            label: `${nb.authors?.[0]?.split(' ').pop() || 'Paper'} (${nb.year || 'n.d.'})`,
            full_title: nb.title,
            node_type: 'paper',
            citation_index: null,
            year: nb.year || 2024,
            citations: nb.citation_count || 0,
            source: nb.source || 'Semantic Scholar',
            cluster_id: parentNode.cluster_id,
            cluster_label: parentNode.cluster_label,
            stance: 'supporting',
            url: nb.url,
            x_timeline: Math.min(0.92, Math.max(0.08, parentNode.x_timeline + Math.cos(angle) * 0.08)),
            y_timeline: Math.min(0.90, Math.max(0.10, parentNode.y_timeline + Math.sin(angle) * 0.12)),
            x_force: Math.min(0.92, Math.max(0.08, parentNode.x_force + Math.cos(angle) * 0.09)),
            y_force: Math.min(0.90, Math.max(0.10, parentNode.y_force + Math.sin(angle) * 0.12)),
            pagerank: 0.05,
          });
        }
        newEdges.push({
          id: `exp:${parentNode.id}->${nb.id}`,
          source: parentNode.id,
          target: nb.id,
          edge_type: 'citation',
          weight: 0.9,
          label: 'cites',
        });
      });

      return {
        ...prev,
        graph: {
          ...prev.graph,
          nodes: newNodes,
          edges: newEdges,
        },
      };
    });
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800">
      {/* Top Navbar */}
      <header className="border-b border-slate-200 bg-white/95 backdrop-blur sticky top-0 z-30 shadow-sm">
        <div className="max-w-[1440px] mx-auto px-4 py-3.5 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center font-black text-white shadow-sm">
              CL
            </div>
            <div>
              <h1 className="text-base font-bold text-slate-900 tracking-tight">
                CiteLume — Synthesis, Evidence Matrix, Gap Discovery & Network Mapping
              </h1>
              <p className="text-[11px] text-slate-500">
                Multi-Repository Literature Synthesis • Structured Evidence Table • Knowledge Gap Matrix • Interactive Network Map • Hugging Face Hub
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <button
              onClick={() => setConfigOpen(true)}
              className="px-3.5 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 border border-slate-300 text-xs font-semibold text-slate-800 flex items-center gap-2 transition"
            >
              <span>⚙ API Keys & Sources Setup</span>
              <span className="px-1.5 py-0.5 rounded bg-indigo-100 text-indigo-700 border border-indigo-200 text-[10px] font-bold">
                {config.llm_provider.toUpperCase()}
              </span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Layout: Left Query History Side Box + Right Workspace */}
      <div className="max-w-[1440px] mx-auto px-4 py-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Sidebar: Query History with Saved Results */}
        <aside className="lg:col-span-3 bg-white border border-slate-200 rounded-2xl p-4 shadow-sm lg:sticky lg:top-20 max-h-[85vh] flex flex-col">
          <div className="flex items-center justify-between border-b border-slate-200 pb-3 mb-3">
            <div>
              <h2 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
                Query History & Saved Results
              </h2>
              <p className="text-[11px] text-slate-500 mt-0.5">
                Click any past query to reload its synthesis & tables
              </p>
            </div>
            {history.length > 0 && (
              <button
                type="button"
                onClick={handleClearHistory}
                className="text-[11px] text-rose-600 hover:text-rose-700 font-semibold px-2 py-0.5 rounded hover:bg-rose-50 transition"
              >
                Clear All
              </button>
            )}
          </div>

          {history.length === 0 ? (
            <div className="text-xs text-slate-500 bg-slate-50 border border-dashed border-slate-200 rounded-xl p-4 text-center my-2">
              No queries run yet. Your search queries and full synthesized results will appear here automatically.
            </div>
          ) : (
            <div className="space-y-2.5 overflow-y-auto pr-1 flex-1">
              {history.map((item) => {
                const isActive = activeHistoryId === item.id;
                return (
                  <div
                    key={item.id}
                    onClick={() => handleSelectHistoryItem(item)}
                    className={`p-3 rounded-xl border cursor-pointer transition group ${
                      isActive
                        ? 'bg-indigo-50/90 border-indigo-300 shadow-sm'
                        : 'bg-slate-50/70 hover:bg-slate-100/80 border-slate-200'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <p className="text-xs font-semibold text-slate-800 line-clamp-2 leading-snug">
                        {item.query}
                      </p>
                      <button
                        type="button"
                        onClick={(e) => handleDeleteHistoryItem(item.id, e)}
                        className="text-slate-400 hover:text-rose-600 text-xs px-1 opacity-70 group-hover:opacity-100"
                        title="Remove from history"
                      >
                        ✕
                      </button>
                    </div>
                    <div className="flex flex-wrap items-center gap-1.5 mt-2 text-[10px]">
                      <span
                        className={`px-1.5 py-0.5 rounded font-semibold ${
                          item.mode === 'deep_research'
                            ? 'bg-purple-100 text-purple-800 border border-purple-200'
                            : 'bg-indigo-100 text-indigo-800 border border-indigo-200'
                        }`}
                      >
                        {item.mode === 'deep_research' ? 'Deep (≤60)' : 'Standard (≤20)'}
                      </span>
                      <span className="px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 font-medium">
                        {item.paperCount} papers
                      </span>
                      {item.hfCount > 0 && (
                        <span className="px-1.5 py-0.5 rounded bg-amber-50 text-amber-800 border border-amber-200 font-medium">
                          {item.hfCount} HF
                        </span>
                      )}
                      <span className="text-slate-400 ml-auto">{item.timestamp}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </aside>

        {/* Main Workspace Column */}
        <main className="lg:col-span-9 space-y-6">
          {/* Query Box & Mode Selector */}
          <form
            onSubmit={handleSearch}
            className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-4"
          >
            <DemoBanner demo={demo} hasOwnKey={!!config.llm_api_key} onOpenConfig={() => setConfigOpen(true)} />
            <div className="flex flex-col md:flex-row gap-3">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a research question, list keywords, or paste a DOI / PubMed / arXiv link..."
                className="flex-1 bg-slate-50 border border-slate-300 focus:border-indigo-500 focus:bg-white rounded-xl px-4 py-3 text-sm text-slate-900 outline-none transition"
              />
              <button
                type="submit"
                disabled={loading || deepBlocked}
                title={deepBlocked ? 'Deep Research needs a larger model - see the note below, or tick "Run anyway"' : undefined}
                className="px-6 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:opacity-60 text-white font-semibold text-sm shadow-sm transition shrink-0"
              >
                {loading
                  ? mode === 'deep_research'
                    ? 'Running Deep Review (≤60 Papers)...'
                    : 'Synthesizing (≤20 Papers)...'
                  : mode === 'deep_research'
                  ? 'Run Deep Research Review (Up to 60 Papers)'
                  : 'Synthesize & Map Literature (Up to 20 Papers)'}
              </button>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
              {/* Mode Toggle: Standard (max 20) vs Deep Research (max 60) */}
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs text-slate-500 font-medium">Synthesis Depth:</span>
                <button
                  type="button"
                  onClick={() => setMode('standard')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition border ${
                    mode === 'standard'
                      ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                      : 'bg-slate-50 text-slate-600 border-slate-300 hover:bg-slate-100'
                  }`}
                >
                  ⚡ Standard Synthesis (Max 20 Papers)
                </button>
                <button
                  type="button"
                  onClick={() => setMode('deep_research')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition border ${
                    mode === 'deep_research'
                      ? 'bg-purple-600 text-white border-purple-600 shadow-sm'
                      : 'bg-slate-50 text-slate-600 border-slate-300 hover:bg-slate-100'
                  }`}
                >
                  🔬 Deep Research Literature Review (Up to 60 Papers)
                </button>
              </div>

              <div className="flex flex-wrap items-center gap-2 text-xs">
                <label className="inline-flex items-center gap-1.5">
                  <span className="text-slate-500 font-medium">Field:</span>
                  <select
                    value={domain}
                    onChange={(e) => setDomain(e.target.value)}
                    className="bg-white border border-slate-300 rounded-lg px-2 py-1.5 text-xs font-medium text-slate-800"
                  >
                    {DOMAIN_OPTIONS.map(([val, label]) => (
                      <option key={val || 'auto'} value={val}>{label}</option>
                    ))}
                  </select>
                </label>
                {mode === 'deep_research' && (
                  <>
                    <label className="inline-flex items-center gap-1.5">
                      <span className="text-slate-500 font-medium">Review type:</span>
                      <select
                        value={reviewFramework}
                        onChange={(e) => setReviewFramework(e.target.value)}
                        className="bg-white border border-purple-300 rounded-lg px-2 py-1.5 text-xs font-medium text-slate-800"
                      >
                        {REVIEW_FRAMEWORK_OPTIONS.map(([val, label]) => (
                          <option key={val} value={val}>{label}</option>
                        ))}
                      </select>
                    </label>
                    <label className="inline-flex items-center gap-1.5">
                      <span className="text-slate-500 font-medium">Organise findings:</span>
                      <select
                        value={organization}
                        onChange={(e) => setOrganization(e.target.value)}
                        className="bg-white border border-purple-300 rounded-lg px-2 py-1.5 text-xs font-medium text-slate-800"
                      >
                        {ORGANIZATION_OPTIONS.map(([val, label]) => (
                          <option key={val} value={val}>{label}</option>
                        ))}
                      </select>
                    </label>
                  </>
                )}
              </div>

              {mode === 'deep_research' && (
                <DeepResearchModelNote
                  model={demo.demo && !config.llm_api_key ? demo.default_model : config.llm_model}
                  hasLlm={hasLlm}
                  scale={modelScale}
                  override={deepOverride}
                  onOverride={setDeepOverride}
                />
              )}

              {/* Interactive Checkboxes to Tick / Untick Repositories Before Search */}
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="text-slate-500 font-medium mr-1">Include Repositories:</span>
                {[
                  ['enable_pubmed', 'PubMed'],
                  ['enable_arxiv', 'arXiv (Public)'],
                  ['enable_biorxiv', 'bioRxiv / medRxiv (Public)'],
                  ['enable_s2', 'Semantic Scholar'],
                  ['enable_hf', 'Hugging Face Hub'],
                  ['enable_fulltext', 'Open-access full text'],
                ].map(([fieldKey, label]) => {
                  const enabled = !!config[fieldKey];
                  return (
                    <label
                      key={fieldKey}
                      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border cursor-pointer select-none transition ${
                        enabled
                          ? 'bg-emerald-50 border-emerald-300 text-emerald-800 font-semibold'
                          : 'bg-slate-50 border-slate-200 text-slate-400 hover:text-slate-600'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={enabled}
                        onChange={(e) => {
                          const checked = e.target.checked;
                          setConfig((prev) => {
                            const next = { ...prev, [fieldKey]: checked };
                            localStorage.setItem('citelume_config', JSON.stringify(next));
                            return next;
                          });
                        }}
                        className="accent-emerald-600 rounded cursor-pointer"
                      />
                      <span>{label}</span>
                    </label>
                  );
                })}
              </div>
            </div>
          </form>

          {/* Live Progress Indicator driven by the backend's reported pipeline stage */}
          {loading && (
            <div className="bg-indigo-50/90 border border-indigo-200 rounded-2xl p-5 shadow-sm space-y-3">
              <div className="flex items-center gap-3.5">
                <div className="w-7 h-7 rounded-full border-3 border-indigo-200 border-t-indigo-600 animate-spin shrink-0" />
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold uppercase tracking-wider text-indigo-700">
                      {mode === 'deep_research'
                        ? 'Deep Research Literature Review Running (Up to 60 Papers)...'
                        : 'Standard Literature Synthesis Running (Up to 20 Papers)...'}
                    </span>
                    <span className="text-[11px] font-mono text-indigo-600 font-semibold">
                      {progress.stage > 0 ? `Stage ${progress.stage} of ${progress.total}` : 'Starting'} •{' '}
                      {formatElapsed(elapsedSecs)} elapsed
                    </span>
                  </div>
                  <p className="text-sm font-medium text-slate-800 mt-1 transition-all duration-300">
                    {progress.label}
                  </p>
                  {progress.detail && (
                    <p className="text-[11px] text-slate-500 mt-0.5">{progress.detail}</p>
                  )}
                </div>
              </div>
              <div className="w-full h-2 bg-indigo-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-indigo-600 rounded-full transition-all duration-500 animate-pulse"
                  style={{ width: `${Math.min(95, (Math.max(progress.stage, 0.3) / progress.total) * 100)}%` }}
                />
              </div>
            </div>
          )}

          {error && (
            <div className="bg-rose-50 border border-rose-300 text-rose-800 rounded-xl p-4 text-xs font-medium">
              {error}
            </div>
          )}

          {!loading && result && <LlmStatusBanner auditTrail={result.evidence_base?.task_audit_trail || []} />}
          {!loading && result && <ScreeningNotes warnings={result.warnings || []} papers={result.papers || []} />}

          {result && (
            <>
              {/* Navigation Tabs in Exact User-Requested Order:
                  1. Cited Synthesis & Bibliography
                  2. Structured Evidence Table
                  3. Knowledge Gap Matrix
                  4. Literature Network Map
                  5. Hugging Face Models & Datasets
              */}
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-3">
                <div className="flex flex-wrap gap-2">
                  {[
                    ['synthesis', `1. Cited Synthesis & Bibliography (${result.bibliography?.length || 0})`],
                    ['evidence', `2. Structured Evidence Table (${result.papers?.length || 0})`],
                    ['gap_matrix', '3. Knowledge Gap Matrix'],
                    ['network', `4. Literature Network Map (${result.graph?.nodes?.length || 0})`],
                    ['huggingface', `5. Hugging Face Models & Datasets (${result.hf_resources?.length || 0})`],
                  ].map(([tabId, label]) => (
                    <button
                      key={tabId}
                      onClick={() => setActiveTab(tabId)}
                      className={`px-3.5 py-2 rounded-xl text-xs font-semibold transition border ${
                        activeTab === tabId
                          ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                          : 'bg-white text-slate-700 hover:bg-slate-100 border-slate-200'
                      }`}
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Active Tab View */}
              {activeTab === 'synthesis' && <SynthesisPanel data={result} />}
              {activeTab === 'evidence' && (
                <EvidenceTable
                  papers={result.papers}
                  evidenceMatrix={result.evidence_matrix}
                  query={result.query}
                />
              )}
              {activeTab === 'gap_matrix' && (
                <KnowledgeGapMatrix gapMatrix={result.gap_matrix} papers={result.papers} />
              )}
              {activeTab === 'network' && (
                <LiteratureGraph
                  graph={result.graph}
                  papers={result.papers}
                  config={config}
                  onGraphExpand={handleGraphExpand}
                />
              )}
              {activeTab === 'huggingface' && <HuggingFacePanel resources={result.hf_resources} />}
            </>
          )}
        </main>
      </div>

      <ApiConfigModal
        isOpen={configOpen}
        onClose={() => setConfigOpen(false)}
        config={config}
        setConfig={setConfig}
      />
    </div>
  );
}

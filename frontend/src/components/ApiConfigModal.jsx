import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  fetchAvailableHfModels,
  fetchProviderModels,
  testLlmApiKey,
  testSourceApiKey,
} from '../utils/api';

export const PROVIDER_PRESETS = {
  gemini: ['gemini-2.0-flash', 'gemini-2.5-flash', 'gemini-2.5-pro'],
  openai: ['gpt-4o', 'gpt-4o-mini', 'o3-mini'],
  anthropic: ['claude-3-5-haiku-latest', 'claude-3-5-sonnet-latest', 'claude-3-7-sonnet-latest'],
  openrouter: [
    'nvidia/nemotron-3-ultra-550b-a55b:free',
    'nvidia/nemotron-3.5-super-vl:free',
    'google/gemma-4-31b-it:free',
    'google/gemini-2.5-flash',
  ],
  huggingface: [
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'meta-llama/Llama-3.3-70B-Instruct',
    'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'Qwen/Qwen2.5-72B-Instruct',
  ],
  ollama: ['deepseek-r1:latest', 'llama3.2:latest', 'mistral:latest', 'qwen2.5:latest'],
  custom: ['deepseek-chat', 'gpt-4o-mini', 'llama-3.3-70b-versatile'],
};

// Free OpenRouter models that passed full-pipeline tests; used as one-click fallback suggestions
const RECOMMENDED_OPENROUTER_FALLBACKS = [
  'nvidia/nemotron-3.5-super-vl:free',
  'nvidia/nemotron-3-ultra-550b-a55b:free',
  'nvidia/nemotron-3.5-lightning:free',
  'google/gemma-4-26b-a4b-it:free',
];

const parseModelList = (text) =>
  text
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .slice(0, 3);

export const CUSTOM_AGGREGATOR_PRESETS = [
  { label: 'LM Studio (Local)', url: 'http://localhost:1234/v1' },
  { label: 'vLLM (Local)', url: 'http://localhost:8000/v1' },
  { label: 'Groq Cloud', url: 'https://api.groq.com/openai/v1' },
  { label: 'Together AI', url: 'https://api.together.xyz/v1' },
  { label: 'DeepSeek API', url: 'https://api.deepseek.com/v1' },
  { label: 'Fireworks AI', url: 'https://api.fireworks.ai/inference/v1' },
];

export default function ApiConfigModal({ isOpen, onClose, config, setConfig }) {
  const [providerModels, setProviderModels] = useState([]);
  const [loadingProviderModels, setLoadingProviderModels] = useState(false);
  const [providerStatusMsg, setProviderStatusMsg] = useState('');
  const [modelFilter, setModelFilter] = useState('');
  const [customModelInput, setCustomModelInput] = useState(config.llm_model || '');
  const [fallbackInput, setFallbackInput] = useState((config.llm_fallback_models || []).join(', '));

  // Config loads from localStorage after first render; sync the text box when the stored list arrives
  useEffect(() => {
    setFallbackInput((config.llm_fallback_models || []).join(', '));
  }, [isOpen]); // eslint-disable-line react-hooks/exhaustive-deps

  // Test & Save states
  const [testingLlm, setTestingLlm] = useState(false);
  const [llmTestResult, setLlmTestResult] = useState(null);
  const [sourceTestStatus, setSourceTestStatus] = useState({});

  // HF scraper states
  const [hfModels, setHfModels] = useState([]);
  const [scrapingHf, setScrapingHf] = useState(false);
  const [hfSearchTerm, setHfSearchTerm] = useState('');

  // Keep customModelInput in sync when config.llm_model changes
  useEffect(() => {
    setCustomModelInput(config.llm_model || '');
  }, [config.llm_model]);

  const saveConfigToStorage = (nextCfg) => {
    localStorage.setItem('citelume_config', JSON.stringify(nextCfg));
  };

  const configRef = useRef(config);
  configRef.current = config;

  const refreshModelsForProvider = useCallback(
    async (overrideCfg = null) => {
      const activeCfg = overrideCfg || configRef.current;
      setLoadingProviderModels(true);
      try {
        const res = await fetchProviderModels({
          llm_provider: activeCfg.llm_provider,
          llm_api_key: activeCfg.llm_api_key || '',
          llm_base_url: activeCfg.llm_base_url || '',
          hf_api_key: activeCfg.hf_api_key || '',
        });
        const fetched = res.models || [];
        setProviderModels(fetched);

        // If no model is configured yet, default to first available
        if (!activeCfg.llm_model && fetched.length > 0) {
          setConfig((prev) => {
            const next = { ...prev, llm_model: fetched[0].id };
            saveConfigToStorage(next);
            return next;
          });
        }

        const freeCount = fetched.filter((m) => m.is_free).length;
        if (res.live_fetched) {
          setProviderStatusMsg(
            `✓ Loaded ${fetched.length} models from ${res.source_label} (${freeCount} Free models at top, sorted A–Z)`
          );
        } else {
          setProviderStatusMsg(
            `Showing ${fetched.length} preset models for ${activeCfg.llm_provider.toUpperCase()} (sorted Free first, then A–Z)`
          );
        }
      } catch (err) {
        console.error(err);
        setProviderStatusMsg('Could not reach provider models endpoint; using sorted presets.');
      } finally {
        setLoadingProviderModels(false);
      }
    },
    [setConfig]
  );

  // Trigger model refresh ONLY when modal opens, or provider/endpoint changes
  useEffect(() => {
    if (!isOpen) return;
    const timer = setTimeout(() => {
      refreshModelsForProvider();
    }, 200);
    return () => clearTimeout(timer);
  }, [
    isOpen,
    config.llm_provider,
    refreshModelsForProvider,
  ]);

  if (!isOpen) return null;

  const updateField = (key, val) => {
    setConfig((prev) => {
      const next = { ...prev };

      if (key === 'llm_provider') {
        const oldProv = prev.llm_provider;
        const newProv = val;

        // 1. Save current provider's state
        const updatedKeys = { ...(prev.provider_keys || {}), [oldProv]: prev.llm_api_key || '' };
        const updatedModels = { ...(prev.provider_models || {}), [oldProv]: prev.llm_model || '' };
        const updatedBaseUrls = { ...(prev.provider_base_urls || {}), [oldProv]: prev.llm_base_url || '' };

        // 2. Retrieve new provider's stored state or sensible presets
        let newKey = updatedKeys[newProv];
        if (newKey === undefined) {
          // If switching to openrouter and user already had an openrouter key, keep it, otherwise empty
          newKey = newProv === 'openrouter' && (prev.llm_api_key || '').startsWith('sk-or-v1-')
            ? prev.llm_api_key
            : '';
        }

        let newModel = updatedModels[newProv];
        if (!newModel) {
          newModel = PROVIDER_PRESETS[newProv]?.[0] || 'gemini-2.5-flash';
        }

        let newBaseUrl = updatedBaseUrls[newProv];
        if (newBaseUrl === undefined) {
          if (newProv === 'ollama') newBaseUrl = 'http://localhost:11434';
          else if (newProv === 'openrouter') newBaseUrl = 'https://openrouter.ai/api/v1';
          else if (newProv === 'custom') newBaseUrl = 'http://localhost:1234/v1';
          else newBaseUrl = '';
        }

        next.llm_provider = newProv;
        next.llm_api_key = newKey;
        next.llm_model = newModel;
        next.llm_base_url = newBaseUrl;
        next.provider_keys = updatedKeys;
        next.provider_models = updatedModels;
        next.provider_base_urls = updatedBaseUrls;

        saveConfigToStorage(next);
        setTimeout(() => {
          refreshModelsForProvider(next);
        }, 50);
        return next;
      }

      if (key === 'llm_model') {
        next.llm_model = val;
        next.provider_models = { ...(prev.provider_models || {}), [prev.llm_provider]: val };
      } else if (key === 'llm_api_key') {
        next.llm_api_key = val;
        next.provider_keys = { ...(prev.provider_keys || {}), [prev.llm_provider]: val };
      } else if (key === 'llm_base_url') {
        next.llm_base_url = val;
        next.provider_base_urls = { ...(prev.provider_base_urls || {}), [prev.llm_provider]: val };
      } else {
        next[key] = val;
      }

      saveConfigToStorage(next);
      return next;
    });
  };

  const handleTestAndSaveLlm = async () => {
    setTestingLlm(true);
    setLlmTestResult(null);
    saveConfigToStorage(config);
    try {
      const res = await testLlmApiKey({
        llm_provider: config.llm_provider,
        llm_api_key: config.llm_api_key || '',
        llm_model: config.llm_model || '',
        llm_base_url: config.llm_base_url || '',
        hf_api_key: config.hf_api_key || '',
      });
      setLlmTestResult(res);
    } catch (err) {
      setLlmTestResult({ success: false, message: err.message });
    } finally {
      setTestingLlm(false);
    }
  };

  const handleTestAndSaveSource = async (sourceKey, apiKeyVal) => {
    saveConfigToStorage(config);
    setSourceTestStatus((prev) => ({
      ...prev,
      [sourceKey]: { loading: true, success: null, message: 'Testing connection & API key...' },
    }));
    try {
      const res = await testSourceApiKey(sourceKey, apiKeyVal);
      setSourceTestStatus((prev) => ({
        ...prev,
        [sourceKey]: { loading: false, success: res.success, message: res.message },
      }));
    } catch (err) {
      setSourceTestStatus((prev) => ({
        ...prev,
        [sourceKey]: { loading: false, success: false, message: err.message },
      }));
    }
  };

  const handleScrapeHf = async () => {
    setScrapingHf(true);
    try {
      const data = await fetchAvailableHfModels(config.hf_api_key, hfSearchTerm);
      setHfModels(data.models || []);
    } catch (e) {
      console.error(e);
    } finally {
      setScrapingHf(false);
    }
  };

  const rawOptions =
    providerModels.length > 0
      ? providerModels
      : (PROVIDER_PRESETS[config.llm_provider] || []).map((id) => ({
          id,
          name: id.includes(':free') ? `★ [FREE] ${id}` : id,
          clean_name: id,
          is_free: id.includes(':free') || config.llm_provider === 'ollama',
        }));

  const filteredModels = rawOptions.filter((m) => {
    if (!modelFilter.trim()) return true;
    const q = modelFilter.toLowerCase();
    return (
      (m.id || '').toLowerCase().includes(q) ||
      (m.name || '').toLowerCase().includes(q)
    );
  });

  // Sort Free models A-Z and Paid/Other models A-Z
  const freeModels = filteredModels
    .filter((m) => m.is_free)
    .sort((a, b) => (a.clean_name || a.name || a.id).localeCompare(b.clean_name || b.name || b.id));
  const otherModels = filteredModels
    .filter((m) => !m.is_free)
    .sort((a, b) => (a.clean_name || a.name || a.id).localeCompare(b.clean_name || b.name || b.id));

  const isCurrentModelInList = rawOptions.some((m) => m.id === config.llm_model);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="bg-white border border-slate-200 rounded-2xl max-w-3xl w-full p-6 shadow-2xl text-slate-800 max-h-[92vh] overflow-y-auto">
        <div className="flex items-center justify-between border-b border-slate-200 pb-4 mb-5">
          <div>
            <h2 className="text-xl font-bold text-slate-900">API Keys, Local/Custom LLMs & Source Verification</h2>
            <p className="text-xs text-slate-500 mt-1">
              Select your preferred LLM provider, customize models, and test connections individually with instant diagnostics.
            </p>
          </div>
          <button
            onClick={() => {
              saveConfigToStorage(config);
              onClose();
            }}
            className="px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-sm"
          >
            Save All & Close
          </button>
        </div>

        {/* 1. LLM Provider Setup */}
        <div className="mb-6 bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h3 className="text-xs font-bold text-indigo-700 uppercase tracking-wider">
                1. LLM Synthesis Engine (Cloud, OpenRouter, Local Ollama, or Custom Provider)
              </h3>
              <p className="text-[11px] text-slate-500 mt-0.5">
                Active: <span className="font-semibold text-slate-800">{config.llm_provider.toUpperCase()}</span> • Model: <code className="text-indigo-700 font-semibold">{config.llm_model || 'default'}</code>
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => refreshModelsForProvider()}
                disabled={loadingProviderModels}
                className="px-2.5 py-1 rounded-lg bg-white hover:bg-slate-100 border border-slate-300 text-slate-700 text-xs font-semibold transition"
              >
                {loadingProviderModels ? '↻ Loading Models...' : '↻ Refresh Models'}
              </button>
              <button
                type="button"
                onClick={handleTestAndSaveLlm}
                disabled={testingLlm}
                className="px-3 py-1 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold shadow-sm transition"
              >
                {testingLlm ? 'Testing...' : `✓ Test Connection (${config.llm_provider.toUpperCase()})`}
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="block text-xs text-slate-600 font-medium mb-1">LLM Provider / Aggregator</label>
              <select
                value={config.llm_provider}
                onChange={(e) => updateField('llm_provider', e.target.value)}
                className="w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-xs font-medium text-slate-800"
              >
                <option value="gemini">Google Gemini</option>
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic Claude</option>
                <option value="openrouter">OpenRouter (Aggregator)</option>
                <option value="ollama">Ollama (Local LLM)</option>
                <option value="custom">Custom Provider / Aggregator (OpenAI-Compatible)</option>
                <option value="huggingface">Hugging Face Inference API</option>
              </select>
            </div>

            <div>
              <label className="block text-xs text-slate-600 font-medium mb-1">
                {config.llm_provider === 'ollama'
                  ? 'API Key (Optional for Local Ollama)'
                  : `${config.llm_provider.toUpperCase()} API Key`}
              </label>
              <input
                type="password"
                value={config.llm_api_key || ''}
                onChange={(e) => updateField('llm_api_key', e.target.value)}
                placeholder={
                  config.llm_provider === 'ollama'
                    ? 'Leave blank for local Ollama'
                    : config.llm_provider === 'openrouter'
                    ? 'sk-or-v1-...'
                    : config.llm_provider === 'gemini'
                    ? 'AIza...'
                    : config.llm_provider === 'anthropic'
                    ? 'sk-ant-...'
                    : 'Paste API Key...'
                }
                className="w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-xs text-slate-800"
              />
            </div>
          </div>

          {(config.llm_provider === 'ollama' ||
            config.llm_provider === 'custom' ||
            config.llm_provider === 'openrouter' ||
            config.llm_base_url) && (
            <div className="pt-1 space-y-2">
              <label className="block text-xs text-slate-600 font-medium">Provider Base URL / Endpoint</label>
              <input
                type="text"
                value={
                  config.llm_base_url ||
                  (config.llm_provider === 'ollama'
                    ? 'http://localhost:11434'
                    : config.llm_provider === 'openrouter'
                    ? 'https://openrouter.ai/api/v1'
                    : 'http://localhost:1234/v1')
                }
                onChange={(e) => updateField('llm_base_url', e.target.value)}
                placeholder="e.g. http://localhost:11434 or https://api.groq.com/openai/v1"
                className="w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-xs font-mono text-slate-800"
              />
              {config.llm_provider === 'custom' && (
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[11px] text-slate-500 mr-1">Quick Aggregator Endpoints:</span>
                  {CUSTOM_AGGREGATOR_PRESETS.map((preset) => (
                    <button
                      key={preset.label}
                      type="button"
                      onClick={() => updateField('llm_base_url', preset.url)}
                      className="px-2 py-0.5 rounded bg-indigo-50 hover:bg-indigo-100 text-[11px] text-indigo-700 border border-indigo-200"
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Model Selection from Catalog */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-1">
            <div className="md:col-span-2">
              <label className="block text-xs text-slate-600 font-medium mb-1">
                Select Model from Catalog (★ {freeModels.length} Free at Top A–Z • {otherModels.length} Standard A–Z)
              </label>
              <select
                value={config.llm_model}
                onChange={(e) => updateField('llm_model', e.target.value)}
                className="w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-xs font-medium text-slate-800"
              >
                {!isCurrentModelInList && config.llm_model && (
                  <option value={config.llm_model}>
                    ★ [Active Selected Model] {config.llm_model}
                  </option>
                )}
                {freeModels.length > 0 && (
                  <optgroup label={`★ Free Models (A–Z) — ${freeModels.length} available`}>
                    {freeModels.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name || m.id}
                      </option>
                    ))}
                  </optgroup>
                )}
                {otherModels.length > 0 && (
                  <optgroup label={`Standard / Paid Models (A–Z) — ${otherModels.length} available`}>
                    {otherModels.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name || m.id}
                      </option>
                    ))}
                  </optgroup>
                )}
              </select>
            </div>
            <div>
              <label className="block text-xs text-slate-600 font-medium mb-1">Filter Model List</label>
              <input
                type="text"
                value={modelFilter}
                onChange={(e) => setModelFilter(e.target.value)}
                placeholder="Search catalog (e.g. free, gemini, llama)..."
                className="w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-xs text-slate-800"
              />
            </div>
          </div>

          {/* Dedicated Custom Model ID Input */}
          <div className="pt-2 bg-white border border-slate-200 rounded-lg p-3 space-y-2">
            <div className="flex flex-wrap items-center justify-between gap-1">
              <label className="block text-xs font-bold text-slate-700">
                Custom / Unlisted Model ID
              </label>
              <span className="text-[11px] text-slate-500">
                Type any model ID and click "Apply Model" to switch instantly
              </span>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={customModelInput}
                onChange={(e) => setCustomModelInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && customModelInput.trim()) {
                    e.preventDefault();
                    updateField('llm_model', customModelInput.trim());
                  }
                }}
                placeholder="e.g. nvidia/nemotron-3-ultra-550b-a55b:free, google/gemma-4-31b-it:free"
                className="flex-1 bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-xs font-mono text-slate-800 focus:bg-white"
              />
              <button
                type="button"
                onClick={() => {
                  if (customModelInput.trim()) {
                    updateField('llm_model', customModelInput.trim());
                  }
                }}
                className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-sm transition whitespace-nowrap"
              >
                Apply Model
              </button>
            </div>
          </div>

          {config.llm_provider === 'openrouter' && (
            <div className="bg-white border border-slate-200 rounded-lg p-3 space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-1">
                <label className="block text-xs font-bold text-slate-700">
                  Backup Models (automatic fallback, up to 3)
                </label>
                <button
                  type="button"
                  onClick={() => {
                    const picks = RECOMMENDED_OPENROUTER_FALLBACKS.filter((m) => m !== config.llm_model).slice(0, 3);
                    setFallbackInput(picks.join(', '));
                    updateField('llm_fallback_models', picks);
                  }}
                  className="px-2 py-0.5 rounded bg-indigo-50 hover:bg-indigo-100 text-[11px] text-indigo-700 border border-indigo-200"
                >
                  Use recommended free backups
                </button>
              </div>
              <input
                type="text"
                value={fallbackInput}
                onChange={(e) => {
                  setFallbackInput(e.target.value);
                  updateField('llm_fallback_models', parseModelList(e.target.value));
                }}
                placeholder="Comma-separated model IDs, tried in order if the main model is rate-limited or down"
                className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-xs font-mono text-slate-800 focus:bg-white"
              />
              <p className="text-[11px] text-slate-500">
                OpenRouter switches to the next model automatically; the report notes which model answered.
              </p>
            </div>
          )}

          {providerStatusMsg && (
            <div className="text-[11px] text-slate-600 bg-white border border-slate-200 rounded-lg px-3 py-1.5">
              {providerStatusMsg}
            </div>
          )}

          {llmTestResult && (
            <div
              className={`text-xs rounded-lg px-3 py-2 border font-medium ${
                llmTestResult.success
                  ? 'bg-emerald-50 border-emerald-300 text-emerald-800'
                  : 'bg-rose-50 border-rose-300 text-rose-800'
              }`}
            >
              {llmTestResult.success ? '✓ Verified & Saved: ' : '✗ Test Failed: '}
              {llmTestResult.message}
            </div>
          )}
        </div>

        {/* 2. Literature Databases Setup with Individual Test & Save */}
        <div className="mb-6 bg-slate-50 border border-slate-200 rounded-xl p-4">
          <h3 className="text-xs font-bold text-emerald-700 uppercase tracking-wider mb-3">
            2. Literature Sources — Individual API Key Test & Save
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {/* PubMed */}
            <div className="p-3 bg-white rounded-lg border border-slate-200 space-y-2 shadow-sm">
              <div className="flex items-center justify-between">
                <label className="flex items-center gap-2 text-xs font-semibold text-slate-800">
                  <input
                    type="checkbox"
                    checked={config.enable_pubmed}
                    onChange={(e) => updateField('enable_pubmed', e.target.checked)}
                  />
                  PubMed (NCBI E-utilities)
                </label>
                <button
                  type="button"
                  onClick={() => handleTestAndSaveSource('pubmed', config.pubmed_api_key)}
                  disabled={sourceTestStatus.pubmed?.loading}
                  className="px-2.5 py-1 rounded bg-slate-100 hover:bg-emerald-600 text-[11px] font-semibold text-slate-700 hover:text-white border border-slate-300 transition"
                >
                  {sourceTestStatus.pubmed?.loading ? 'Testing...' : '✓ Test & Save'}
                </button>
              </div>
              <input
                type="password"
                value={config.pubmed_api_key || ''}
                onChange={(e) => updateField('pubmed_api_key', e.target.value)}
                placeholder="Optional NCBI API Key (boosts to 10 req/s)"
                className="w-full bg-slate-50 border border-slate-300 rounded px-2.5 py-1.5 text-xs text-slate-800"
              />
              {sourceTestStatus.pubmed && (
                <div
                  className={`text-[11px] font-medium ${
                    sourceTestStatus.pubmed.success ? 'text-emerald-700' : 'text-rose-700'
                  }`}
                >
                  {sourceTestStatus.pubmed.message}
                </div>
              )}
            </div>

            {/* Semantic Scholar */}
            <div className="p-3 bg-white rounded-lg border border-slate-200 space-y-2 shadow-sm">
              <div className="flex items-center justify-between">
                <label className="flex items-center gap-2 text-xs font-semibold text-slate-800">
                  <input
                    type="checkbox"
                    checked={config.enable_s2}
                    onChange={(e) => updateField('enable_s2', e.target.checked)}
                  />
                  Semantic Scholar Graph API
                </label>
                <button
                  type="button"
                  onClick={() => handleTestAndSaveSource('s2', config.s2_api_key)}
                  disabled={sourceTestStatus.s2?.loading}
                  className="px-2.5 py-1 rounded bg-slate-100 hover:bg-emerald-600 text-[11px] font-semibold text-slate-700 hover:text-white border border-slate-300 transition"
                >
                  {sourceTestStatus.s2?.loading ? 'Testing...' : '✓ Test & Save'}
                </button>
              </div>
              <input
                type="password"
                value={config.s2_api_key || ''}
                onChange={(e) => updateField('s2_api_key', e.target.value)}
                placeholder="Optional S2 API Key (OpenAlex fallback active)"
                className="w-full bg-slate-50 border border-slate-300 rounded px-2.5 py-1.5 text-xs text-slate-800"
              />
              <input
                type="password"
                value={config.openalex_api_key || ''}
                onChange={(e) => updateField('openalex_api_key', e.target.value)}
                placeholder="OpenAlex API key for the fallback (free at openalex.org/settings/api)"
                className="w-full bg-slate-50 border border-slate-300 rounded px-2.5 py-1.5 text-xs text-slate-800"
              />
              <p className="text-[11px] text-slate-500">
                OpenAlex now requires a free key for regular use; without one its fallback is quickly rate-limited.
              </p>
              {sourceTestStatus.s2 && (
                <div
                  className={`text-[11px] font-medium ${
                    sourceTestStatus.s2.success ? 'text-emerald-700' : 'text-rose-700'
                  }`}
                >
                  {sourceTestStatus.s2.message}
                </div>
              )}
            </div>

            {/* arXiv */}
            <div className="p-3 bg-white rounded-lg border border-slate-200 space-y-2 shadow-sm">
              <div className="flex items-center justify-between">
                <label className="flex items-center gap-2 text-xs font-semibold text-slate-800">
                  <input
                    type="checkbox"
                    checked={config.enable_arxiv}
                    onChange={(e) => updateField('enable_arxiv', e.target.checked)}
                  />
                  arXiv.org API
                </label>
                <button
                  type="button"
                  onClick={() => handleTestAndSaveSource('arxiv', '')}
                  disabled={sourceTestStatus.arxiv?.loading}
                  className="px-2.5 py-1 rounded bg-slate-100 hover:bg-emerald-600 text-[11px] font-semibold text-slate-700 hover:text-white border border-slate-300 transition"
                >
                  {sourceTestStatus.arxiv?.loading ? 'Testing...' : '✓ Test Endpoint'}
                </button>
              </div>
              <p className="text-[11px] text-slate-500">Public academic open access (no API key required).</p>
              {sourceTestStatus.arxiv && (
                <div
                  className={`text-[11px] font-medium ${
                    sourceTestStatus.arxiv.success ? 'text-emerald-700' : 'text-rose-700'
                  }`}
                >
                  {sourceTestStatus.arxiv.message}
                </div>
              )}
            </div>

            {/* bioRxiv */}
            <div className="p-3 bg-white rounded-lg border border-slate-200 space-y-2 shadow-sm">
              <div className="flex items-center justify-between">
                <label className="flex items-center gap-2 text-xs font-semibold text-slate-800">
                  <input
                    type="checkbox"
                    checked={config.enable_biorxiv}
                    onChange={(e) => updateField('enable_biorxiv', e.target.checked)}
                  />
                  bioRxiv & medRxiv API
                </label>
                <button
                  type="button"
                  onClick={() => handleTestAndSaveSource('biorxiv', '')}
                  disabled={sourceTestStatus.biorxiv?.loading}
                  className="px-2.5 py-1 rounded bg-slate-100 hover:bg-emerald-600 text-[11px] font-semibold text-slate-700 hover:text-white border border-slate-300 transition"
                >
                  {sourceTestStatus.biorxiv?.loading ? 'Testing...' : '✓ Test Endpoint'}
                </button>
              </div>
              <p className="text-[11px] text-slate-500">Public biological preprint server (no API key required).</p>
              {sourceTestStatus.biorxiv && (
                <div
                  className={`text-[11px] font-medium ${
                    sourceTestStatus.biorxiv.success ? 'text-emerald-700' : 'text-rose-700'
                  }`}
                >
                  {sourceTestStatus.biorxiv.message}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* 3. Hugging Face Hub Scraper */}
        <div className="mb-2 bg-slate-50 border border-slate-200 rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-bold text-amber-700 uppercase tracking-wider">
              3. Hugging Face Hub Integration
            </h3>
            <button
              type="button"
              onClick={() => handleTestAndSaveSource('hf', config.hf_api_key)}
              disabled={sourceTestStatus.hf?.loading}
              className="px-2.5 py-1 rounded bg-slate-100 hover:bg-emerald-600 text-[11px] font-semibold text-slate-700 hover:text-white border border-slate-300 transition"
            >
              {sourceTestStatus.hf?.loading ? 'Testing...' : '✓ Test & Save HF Token'}
            </button>
          </div>

          <div className="space-y-3">
            <input
              type="password"
              value={config.hf_api_key || ''}
              onChange={(e) => updateField('hf_api_key', e.target.value)}
              placeholder="Hugging Face User Access Token (hf_...)"
              className="w-full bg-white border border-slate-300 rounded-lg px-3 py-2 text-xs text-slate-800"
            />
            {sourceTestStatus.hf && (
              <div
                className={`text-[11px] font-medium ${
                  sourceTestStatus.hf.success ? 'text-emerald-700' : 'text-rose-700'
                }`}
              >
                {sourceTestStatus.hf.message}
              </div>
            )}

            <div className="flex items-center gap-2">
              <input
                type="text"
                value={hfSearchTerm}
                onChange={(e) => setHfSearchTerm(e.target.value)}
                placeholder="Search models on Hub (e.g. esm2, protein, chem)..."
                className="flex-1 bg-white border border-slate-300 rounded-lg px-3 py-1.5 text-xs text-slate-800"
              />
              <button
                type="button"
                onClick={handleScrapeHf}
                disabled={scrapingHf}
                className="px-3 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white text-xs font-semibold"
              >
                {scrapingHf ? 'Scraping Hub...' : 'Search Models'}
              </button>
            </div>
          </div>

          {hfModels.length > 0 && (
            <div className="mt-3 max-h-40 overflow-y-auto space-y-1.5 border-t border-slate-200 pt-3">
              {hfModels.map((m) => (
                <div
                  key={m.id}
                  className="flex items-center justify-between text-xs px-2 py-1 rounded bg-slate-50 border border-slate-200"
                >
                  <span className="font-mono text-amber-800 font-medium">
                    {m.id} <span className="text-slate-400">({m.downloads?.toLocaleString()} dl)</span>
                  </span>
                  <button
                    onClick={() => {
                      updateField('llm_provider', 'huggingface');
                      updateField('llm_model', m.id);
                    }}
                    className="text-indigo-600 hover:underline text-xs font-semibold"
                  >
                    Use as LLM
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

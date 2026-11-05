import React from 'react';

export default function HuggingFacePanel({ resources = [] }) {
  const models = resources.filter((r) => r.resource_type === 'model');
  const datasets = resources.filter((r) => r.resource_type === 'dataset');

  return (
    <div className="space-y-6">
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
        <h2 className="text-base font-bold text-slate-900 mb-1">
          Scraped Hugging Face Hub Models ({models.length}) & Datasets ({datasets.length})
        </h2>
        <p className="text-xs text-slate-500 mb-4">
          Domain-specific models and datasets discovered on Hugging Face Hub via decomposed domain queries and direct paper arXiv ID tags (`arxiv:&lt;id&gt;`).
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {resources.map((res) => (
            <div
              key={res.id}
              className="bg-slate-50/70 border border-slate-200 hover:border-amber-400 rounded-xl p-4 flex flex-col justify-between transition shadow-sm"
            >
              <div>
                <div className="flex items-center justify-between gap-2 mb-2">
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                      res.resource_type === 'model'
                        ? 'bg-amber-50 text-amber-800 border border-amber-200'
                        : 'bg-cyan-50 text-cyan-800 border border-cyan-200'
                    }`}
                  >
                    {res.resource_type} • {res.pipeline_tag}
                  </span>
                  <span className="text-[11px] font-mono text-slate-500">
                    ⬇ {res.downloads?.toLocaleString()} • ♥ {res.likes}
                  </span>
                </div>

                <h3 className="font-mono text-xs font-bold text-slate-900 break-all mb-1.5">{res.id}</h3>
                <p className="text-xs text-slate-600 leading-relaxed mb-3">{res.description}</p>
              </div>

              <div className="pt-2 border-t border-slate-200 flex items-center justify-between">
                <div className="flex flex-wrap gap-1">
                  {(res.linked_arxiv_ids || []).slice(0, 2).map((aid) => (
                    <span
                      key={aid}
                      className="px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200 font-mono text-[10px]"
                    >
                      arXiv:{aid}
                    </span>
                  ))}
                </div>
                <a
                  href={res.url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-xs font-semibold text-amber-700 hover:underline"
                >
                  Open on HF ↗
                </a>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

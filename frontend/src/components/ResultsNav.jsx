// The five result views, shown two ways: large cards at the top of the results, and a compact tab strip inside the
// sticky header so the views stay one click away while reading.

export function resultSections(result) {
  const gaps = result?.gap_matrix?.structured_gaps?.length || 0;
  return [
    {
      id: 'synthesis',
      icon: '📝',
      title: 'Literature Synthesis',
      short: 'Synthesis',
      desc: 'Cited answer, agreement meter and references',
      count: `${result?.bibliography?.length || 0} references`,
    },
    {
      id: 'evidence',
      icon: '📊',
      title: 'Evidence Table',
      short: 'Evidence',
      desc: 'Design, sample, results and appraisal of each study',
      count: `${result?.papers?.length || 0} studies`,
    },
    {
      id: 'gap_matrix',
      icon: '🧩',
      title: 'Knowledge Gaps',
      short: 'Gaps',
      desc: 'What is still unknown and where to look next',
      count: `${gaps} gap${gaps === 1 ? '' : 's'}`,
    },
    {
      id: 'network',
      icon: '🕸️',
      title: 'Literature Map',
      short: 'Map',
      desc: 'How the studies cite and cluster around each other',
      count: `${result?.graph?.nodes?.length || 0} nodes`,
    },
    {
      id: 'huggingface',
      icon: '🤗',
      title: 'Models & Datasets',
      short: 'Models',
      desc: 'Related Hugging Face models and datasets',
      count: `${result?.hf_resources?.length || 0} resources`,
    },
  ];
}

export function ResultsCards({ result, activeTab, onSelect }) {
  const sections = resultSections(result);
  return (
    <section aria-label="Result sections" className="space-y-2">
      <div className="flex items-baseline justify-between gap-2 px-1">
        <h2 className="text-sm font-bold text-slate-900">Your results: 5 views</h2>
        <span className="text-[11px] text-slate-500">Choose a view; they stay in the top bar while you scroll</span>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2.5" role="tablist">
        {sections.map((s, i) => {
          const active = activeTab === s.id;
          return (
            <button
              key={s.id}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => onSelect(s.id)}
              className={`text-left rounded-2xl border p-3 transition shadow-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500 ${
                active
                  ? 'bg-indigo-600 border-indigo-600 text-white shadow-md'
                  : 'bg-white border-slate-200 text-slate-800 hover:border-indigo-300 hover:bg-indigo-50/40'
              } ${i === 4 ? 'col-span-2 sm:col-span-1' : ''}`}
            >
              <div className="flex items-center gap-2">
                <span className="text-xl leading-none" aria-hidden="true">{s.icon}</span>
                <span className="text-sm font-bold leading-tight">{s.title}</span>
              </div>
              <div className={`mt-1 text-[11px] leading-snug ${active ? 'text-indigo-100' : 'text-slate-500'}`}>
                {s.desc}
              </div>
              <div
                className={`mt-2 inline-block whitespace-nowrap text-[10px] font-bold rounded-full px-2 py-0.5 ${
                  active ? 'bg-white/20 text-white' : 'bg-slate-100 text-slate-600'
                }`}
              >
                {s.count}
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}

export function ResultsTabStrip({ result, activeTab, onSelect }) {
  const sections = resultSections(result);
  return (
    <nav aria-label="Result sections" className="border-t border-slate-200 bg-white/95">
      <div className="max-w-[1440px] mx-auto px-4 flex gap-1 overflow-x-auto" style={{ scrollbarWidth: 'none' }}>
        {sections.map((s) => {
          const active = activeTab === s.id;
          return (
            <button
              key={s.id}
              type="button"
              onClick={() => onSelect(s.id)}
              aria-current={active ? 'page' : undefined}
              className={`shrink-0 inline-flex items-center gap-1.5 px-3 py-2 text-xs font-semibold border-b-2 transition ${
                active
                  ? 'border-indigo-600 text-indigo-700'
                  : 'border-transparent text-slate-600 hover:text-slate-900 hover:border-slate-300'
              }`}
            >
              <span aria-hidden="true">{s.icon}</span>
              <span className="hidden sm:inline">{s.title}</span>
              <span className="sm:hidden">{s.short}</span>
            </button>
          );
        })}
      </div>
    </nav>
  );
}

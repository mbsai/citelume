import React, { useState } from 'react';

// Where the included studies sit in the whole candidate pool: the records the searches found, grouped by their
// words. A large group with no included study is flagged. Clicking a group lists its records.

function ChasingSummary({ chasing }) {
  if (!chasing || !chasing.ran) return null;
  return (
    <p className="text-xs text-slate-600">
      <strong>Citation searching:</strong> from {chasing.seeds} selected {chasing.seeds === 1 ? 'study' : 'studies'},{' '}
      {chasing.seeds_found} found in Semantic Scholar ({chasing.seeds_with_references} with a reference list,{' '}
      {chasing.seeds_with_citations} cited by others). {chasing.added}{' '}
      {chasing.added === 1 ? 'paper' : 'papers'} linked to at least {chasing.min_links} of them and not already found
      {chasing.added === 1 ? ' was' : ' were'} added to the pool; {chasing.included} of these passed selection.
      {chasing.note && <span className="text-amber-700"> {chasing.note}</span>}
    </p>
  );
}

export default function PoolMap({ pool }) {
  const [open, setOpen] = useState(null);
  if (!pool || !pool.clusters?.length) return null;
  const max = Math.max(...pool.clusters.map((c) => c.size));
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm space-y-3">
      <div>
        <h3 className="text-sm font-bold text-slate-900">Where the included studies sit in the pool</h3>
        <p className="text-xs text-slate-500 mt-0.5">
          All {pool.records} records the searches found, grouped by the words in their titles and abstracts;{' '}
          {pool.included} were included. {pool.included_with_references} of the included studies had a reference list,
          which the citation links above depend on. Click a group to list its records.
        </p>
      </div>
      <ChasingSummary chasing={pool.citation_chasing} />
      {pool.note && (
        <p className="text-xs text-amber-900 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">{pool.note}</p>
      )}
      <ul className="space-y-1">
        {pool.clusters.map((c) => {
          const isOpen = open === c.cluster_id;
          return (
            <li key={c.cluster_id} className="border border-slate-100 rounded-lg">
              <button
                type="button"
                onClick={() => setOpen(isOpen ? null : c.cluster_id)}
                aria-expanded={isOpen}
                className="w-full text-left px-3 py-1.5 flex items-center gap-3 hover:bg-slate-50"
              >
                <span className="w-40 sm:w-56 shrink-0 text-xs font-semibold text-slate-800 truncate" title={c.label}>
                  {c.label}
                </span>
                <span className="flex-1 h-3 bg-slate-100 rounded overflow-hidden" aria-hidden="true">
                  <span className="flex h-full" style={{ width: `${(100 * c.size) / max}%` }}>
                    <span className="bg-indigo-600 h-full" style={{ width: `${(100 * c.included) / c.size}%` }} />
                    <span className={`${c.flagged ? 'bg-amber-400' : 'bg-slate-300'} h-full flex-1`} />
                  </span>
                </span>
                <span className="shrink-0 text-[11px] text-slate-600 w-28 text-right">
                  {c.included} of {c.size} included
                </span>
                {c.flagged && (
                  <span className="shrink-0 text-[10px] font-semibold text-amber-800 bg-amber-100 rounded px-1.5">
                    check
                  </span>
                )}
              </button>
              {isOpen && (
                <ul className="px-3 pb-2 pt-1 space-y-0.5 text-[11px] text-slate-700 max-h-72 overflow-y-auto">
                  {c.members.map((m) => (
                    <li key={m.id}>
                      {m.included ? (
                        <span className="font-mono text-indigo-700 font-semibold mr-1">[{m.citation_index}]</span>
                      ) : (
                        <span className="text-slate-400 mr-1">–</span>
                      )}
                      <span className={m.included ? 'font-semibold' : ''}>{m.title}</span>
                      {m.year ? ` (${m.year})` : ''}
                      {m.chased && <span className="ml-1 text-sky-700">· citation searching</span>}
                      {!m.included && m.excluded_reason && (
                        <span className="text-slate-500"> · {m.excluded_reason}</span>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </li>
          );
        })}
      </ul>
      <p className="text-[11px] text-slate-500">
        <span className="inline-block w-3 h-2 bg-indigo-600 align-middle" /> included{' '}
        <span className="inline-block w-3 h-2 bg-slate-300 align-middle ml-2" /> not included{' '}
        <span className="inline-block w-3 h-2 bg-amber-400 align-middle ml-2" /> large group with none included.
        Groups: {pool.method}.
      </p>
    </div>
  );
}

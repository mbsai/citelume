import React, { useState } from 'react';

// "How these studies were chosen": the selection protocol's funnel, scoring criteria, rules and per-study scores
const PARTS = [
  ['relevance', 'Relevance', 'bg-indigo-500'],
  ['evidence', 'Study type', 'bg-emerald-500'],
  ['impact', 'Citations', 'bg-amber-500'],
  ['recency', 'Recency', 'bg-sky-500'],
  ['agreement', 'Found by several searches', 'bg-slate-400'],
];

function ScoreBar({ breakdown }) {
  const total = breakdown?.total ?? 0;
  return (
    <div className="flex items-center gap-2 min-w-[150px]">
      <span className="w-9 text-right font-bold text-slate-800 tabular-nums">{total.toFixed(0)}</span>
      <div className="flex h-2.5 flex-1 rounded-full overflow-hidden bg-slate-100" title={PARTS.map(([k, l]) => `${l}: ${breakdown?.[k] ?? 0}`).join(' · ')}>
        {PARTS.map(([key, , color]) => (
          <div key={key} className={color} style={{ width: `${Math.max(0, breakdown?.[key] ?? 0)}%` }} />
        ))}
      </div>
    </div>
  );
}

function Step({ n, label, value, note }) {
  return (
    <div className="flex-1 min-w-[120px] rounded-xl border border-slate-200 bg-white px-3 py-2">
      <div className="text-[10px] font-bold uppercase tracking-wide text-slate-400">Step {n}</div>
      <div className="text-lg font-bold text-slate-900 leading-tight tabular-nums">{value}</div>
      <div className="text-[11px] text-slate-600 leading-snug">{label}</div>
      {note && <div className="text-[10px] text-slate-400 mt-0.5">{note}</div>}
    </div>
  );
}

export default function SelectionPanel({ report, papers, understanding }) {
  const [open, setOpen] = useState(false);
  if (!report || !report.included) return null;
  const excluded = Object.entries(report.excluded || {});
  const nExcluded = excluded.reduce((a, [, v]) => a + v, 0);
  const types = Object.entries(report.included_by_study_type || {});
  const aspects = Object.entries(report.included_by_aspect || {});
  const concepts = understanding?.concepts?.search_concepts || [];

  return (
    <section className="rounded-2xl border border-slate-200 bg-white shadow-sm">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="w-full flex flex-wrap items-center justify-between gap-2 px-4 py-3 text-left"
      >
        <span className="text-xs text-slate-700">
          <strong className="text-slate-900">How these {report.included} studies were chosen:</strong>{' '}
          {report.records_found} records found, {report.unique_records} after merging duplicates, {report.eligible}{' '}
          eligible, top {report.included} included by a published 100-point score.
        </span>
        <span className="text-xs font-semibold text-indigo-700 shrink-0">{open ? 'Hide details ▴' : 'Show details ▾'}</span>
      </button>

      {open && (
        <div className="border-t border-slate-200 px-4 py-4 space-y-5 text-xs text-slate-700">
          {(concepts.length > 0 || understanding?.breadth) && (
            <div className="space-y-1">
              <h4 className="font-bold text-slate-900">What was searched for</h4>
              <p>
                Read as a <strong>{understanding?.breadth === 'broad' ? 'broad topic to survey' : 'specific question'}</strong>
                {understanding?.planned_by ? ` (search planned by ${understanding.planned_by})` : ''}. Every search required:
              </p>
              <ul className="list-disc pl-5 space-y-0.5">
                {concepts.map((c) => (
                  <li key={c.name}>
                    <strong>{c.name}</strong>
                    {c.synonyms?.length ? ` (also: ${c.synonyms.join(', ')})` : ''}
                  </li>
                ))}
                {(understanding?.concepts?.required_constraints || []).map((c) => (
                  <li key={c.name}>
                    <strong>{c.name}</strong> ({c.kind})
                  </li>
                ))}
              </ul>
            </div>
          )}

          <div className="space-y-2">
            <h4 className="font-bold text-slate-900">From search results to included studies</h4>
            <div className="flex flex-wrap gap-2">
              <Step n={1} value={report.records_found} label="records returned by the searches" note={`${report.duplicates_merged} duplicates merged`} />
              <Step n={2} value={report.unique_records} label="unique records screened" note={`${nExcluded} removed`} />
              <Step n={3} value={report.eligible} label="eligible studies scored out of 100" />
              <Step n={4} value={report.included} label={`included (limit ${report.target_count})`} />
            </div>
            {excluded.length > 0 && (
              <ul className="list-disc pl-5 space-y-0.5">
                {excluded.map(([reason, n]) => (
                  <li key={reason}>
                    <strong>{n}</strong> removed: {reason}
                  </li>
                ))}
              </ul>
            )}
            {(report.notes || []).map((n) => (
              <p key={n} className="text-amber-800">{n}</p>
            ))}
          </div>

          <div className="space-y-2">
            <h4 className="font-bold text-slate-900">How each eligible study is scored (100 points)</h4>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <tbody>
                  {(report.criteria || []).map((c, i) => (
                    <tr key={c.key} className="border-t border-slate-100 align-top">
                      <td className="py-1.5 pr-3 whitespace-nowrap">
                        <span className={`inline-block w-2.5 h-2.5 rounded-sm mr-1.5 ${PARTS[i]?.[2] || 'bg-slate-300'}`} />
                        <strong>{c.label}</strong>
                      </td>
                      <td className="py-1.5 pr-3 whitespace-nowrap font-bold tabular-nums">{c.max_points}</td>
                      <td className="py-1.5 text-slate-600">{c.rule}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {report.citation_data && <p className="text-slate-500">{report.citation_data}</p>}
          </div>

          <div className="space-y-1">
            <h4 className="font-bold text-slate-900">Rules for filling the {report.target_count} places</h4>
            <ol className="list-decimal pl-5 space-y-0.5">
              {(report.selection_rules || []).map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ol>
          </div>

          {(types.length > 0 || aspects.length > 0) && (
            <div className="grid sm:grid-cols-2 gap-4">
              {types.length > 0 && (
                <div>
                  <h4 className="font-bold text-slate-900 mb-1">Included studies by type</h4>
                  <ul className="space-y-0.5">
                    {types.map(([t, n]) => (
                      <li key={t} className="flex justify-between gap-3 border-b border-slate-100 py-0.5">
                        <span>{t}</span>
                        <strong className="tabular-nums">{n}</strong>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {aspects.length > 0 && (
                <div>
                  <h4 className="font-bold text-slate-900 mb-1">Included studies covering each aspect</h4>
                  <ul className="space-y-0.5">
                    {aspects.map(([t, n]) => (
                      <li key={t} className="flex justify-between gap-3 border-b border-slate-100 py-0.5">
                        <span>{t}</span>
                        <strong className="tabular-nums">{n}</strong>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          <div className="space-y-2">
            <h4 className="font-bold text-slate-900">Score of every included study</h4>
            <div className="flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-slate-500">
              {PARTS.map(([key, label, color]) => (
                <span key={key} className="inline-flex items-center gap-1">
                  <span className={`inline-block w-2.5 h-2.5 rounded-sm ${color}`} />
                  {label}
                </span>
              ))}
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse min-w-[640px]">
                <thead>
                  <tr className="text-[10px] uppercase tracking-wide text-slate-400">
                    <th className="py-1 pr-2 font-bold">#</th>
                    <th className="py-1 pr-3 font-bold">Study</th>
                    <th className="py-1 pr-3 font-bold">Score</th>
                    <th className="py-1 pr-3 font-bold">Type</th>
                    <th className="py-1 font-bold">Why included</th>
                  </tr>
                </thead>
                <tbody>
                  {(papers || []).map((p) => (
                    <tr key={p.id} className="border-t border-slate-100 align-top">
                      <td className="py-1.5 pr-2 font-bold text-slate-500 tabular-nums">[{p.citation_index}]</td>
                      <td className="py-1.5 pr-3 max-w-[340px]">
                        <span className="text-slate-800">{p.title}</span>{' '}
                        <span className="text-slate-400">
                          ({p.year || 'n.d.'}
                          {p.citation_source ? `; ${p.citation_count} citations, ${p.citation_source}` : ''})
                        </span>
                      </td>
                      <td className="py-1.5 pr-3"><ScoreBar breakdown={p.selection_breakdown} /></td>
                      <td className="py-1.5 pr-3 text-slate-600">{(report.study_type_labels || {})[p.study_type] || p.study_type}</td>
                      <td className="py-1.5 text-slate-600">{p.selection_reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <p className="text-[11px] text-slate-400">Selection protocol {report.protocol_version}. Full description: docs/RANKING.md in the project repository.</p>
        </div>
      )}
    </section>
  );
}

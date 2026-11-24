import React, { useEffect, useMemo, useState } from 'react';
import { suggestGapMapAxes } from '../utils/api';

// Evidence gap map (backend: services/gap_map.py). The set-up proposes rows, columns and a population for the
// topic; the user edits them; the counts come from PubMed (by study design) and OpenAlex, with no model.

const splitList = (s) =>
  (s || '')
    .split(',')
    .map((x) => x.trim())
    .filter(Boolean);

const termToText = (t) => ({ name: t?.name || '', synonyms: (t?.synonyms || []).join(', ') });
const textToTerm = (t) => ({ name: t.name.trim(), synonyms: splitList(t.synonyms) });

const FLAG_STYLE = {
  none: 'bg-rose-100 text-rose-800 border-rose-300',
  none_pubmed: 'bg-rose-50 text-rose-800 border-rose-200',
  sparse: 'bg-orange-100 text-orange-800 border-orange-300',
  no_trials: 'bg-amber-100 text-amber-800 border-amber-300',
  unsynthesised: 'bg-sky-100 text-sky-800 border-sky-300',
};

function TermList({ title, hint, items, setItems, max }) {
  const update = (i, patch) => setItems(items.map((t, j) => (j === i ? { ...t, ...patch } : t)));
  return (
    <div className="space-y-1.5">
      <div className="text-xs font-bold text-slate-700 uppercase tracking-wide">{title}</div>
      <p className="text-[11px] text-slate-500">{hint}</p>
      {items.map((t, i) => (
        <div key={i} className="flex flex-wrap gap-1.5 items-center">
          <input
            value={t.name}
            onChange={(e) => update(i, { name: e.target.value })}
            placeholder="term"
            aria-label={`${title} ${i + 1}`}
            className="w-44 bg-white border border-slate-300 rounded-lg px-2 py-1 text-xs"
          />
          <input
            value={t.synonyms}
            onChange={(e) => update(i, { synonyms: e.target.value })}
            placeholder="synonyms, comma-separated"
            aria-label={`${title} ${i + 1} synonyms`}
            className="flex-1 min-w-[10rem] bg-white border border-slate-300 rounded-lg px-2 py-1 text-xs"
          />
          <button
            type="button"
            onClick={() => setItems(items.filter((_, j) => j !== i))}
            className="text-[11px] text-rose-700 hover:underline"
          >
            remove
          </button>
        </div>
      ))}
      {items.length < max && (
        <button
          type="button"
          onClick={() => setItems([...items, { name: '', synonyms: '' }])}
          className="text-[11px] font-semibold text-indigo-700 hover:underline"
        >
          + Add ({items.length} of {max})
        </button>
      )}
    </div>
  );
}

export function GapMapSetup({ topic, config, demo, running, onRun, onCancel }) {
  const max = demo?.demo ? 5 : 8;
  const [busy, setBusy] = useState(true);
  const [note, setNote] = useState('');
  const [error, setError] = useState('');
  const [population, setPopulation] = useState({ name: '', synonyms: '' });
  const [rows, setRows] = useState([]);
  const [cols, setCols] = useState([]);
  const [rowTitle, setRowTitle] = useState('Interventions or exposures');
  const [colTitle, setColTitle] = useState('Outcomes');
  const [yearFrom, setYearFrom] = useState('');
  const [yearTo, setYearTo] = useState('');

  useEffect(() => {
    let live = true;
    setBusy(true);
    suggestGapMapAxes({ topic, config })
      .then((s) => {
        if (!live) return;
        setPopulation(termToText(s.population));
        setRows((s.rows || []).slice(0, max).map(termToText));
        setCols((s.cols || []).slice(0, max).map(termToText));
        setNote(`${s.note}${s.source && s.source !== 'rules' ? ` (${s.source})` : ''}`);
      })
      .catch((e) => live && setError(e.message || 'Could not suggest the map; enter the rows and columns yourself.'))
      .finally(() => live && setBusy(false));
    return () => {
      live = false;
    };
  }, [topic]); // eslint-disable-line react-hooks/exhaustive-deps

  const goodRows = rows.filter((t) => t.name.trim());
  const goodCols = cols.filter((t) => t.name.trim());
  const queries = goodRows.length * goodCols.length * 5 + goodRows.length + goodCols.length;
  const ready = goodRows.length > 0 && goodCols.length > 0 && !busy && !running;

  return (
    <div className="bg-white border border-teal-200 rounded-2xl p-5 shadow-sm space-y-4" aria-label="Set up the evidence gap map">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-base font-bold text-slate-900">Set up the evidence gap map</h2>
          <p className="text-xs text-slate-600 mt-1">
            Each cell counts the records for one row and one column (within the population, if you give one) in PubMed,
            by study design, and in OpenAlex. No model is used for the counts, and every query is shown with its date.
          </p>
        </div>
        <button type="button" onClick={onCancel} className="text-xs text-slate-500 hover:text-slate-800">
          Cancel
        </button>
      </div>
      {busy && <p className="text-xs text-slate-500">Reading the topic...</p>}
      {note && !busy && <p className="text-[11px] text-teal-800 bg-teal-50 border border-teal-200 rounded-lg px-2.5 py-1.5">{note}</p>}
      {error && <p className="text-xs text-rose-700">{error}</p>}
      {!busy && (
        <>
          <div className="space-y-1.5">
            <div className="text-xs font-bold text-slate-700 uppercase tracking-wide">Population (optional)</div>
            <div className="flex flex-wrap gap-1.5">
              <input
                value={population.name}
                onChange={(e) => setPopulation({ ...population, name: e.target.value })}
                placeholder="e.g. older adults"
                aria-label="Population"
                className="w-44 bg-white border border-slate-300 rounded-lg px-2 py-1 text-xs"
              />
              <input
                value={population.synonyms}
                onChange={(e) => setPopulation({ ...population, synonyms: e.target.value })}
                placeholder="synonyms, comma-separated"
                aria-label="Population synonyms"
                className="flex-1 min-w-[10rem] bg-white border border-slate-300 rounded-lg px-2 py-1 text-xs"
              />
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-1">
              <input
                value={rowTitle}
                onChange={(e) => setRowTitle(e.target.value)}
                aria-label="Row heading"
                className="text-[11px] text-slate-500 bg-transparent border-b border-dashed border-slate-300 w-full"
              />
              <TermList title="Rows" hint="Interventions, exposures or approaches." items={rows} setItems={setRows} max={max} />
            </div>
            <div className="space-y-1">
              <input
                value={colTitle}
                onChange={(e) => setColTitle(e.target.value)}
                aria-label="Column heading"
                className="text-[11px] text-slate-500 bg-transparent border-b border-dashed border-slate-300 w-full"
              />
              <TermList title="Columns" hint="Outcomes that matter for the topic." items={cols} setItems={setCols} max={max} />
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-700">
            Published from
            <input value={yearFrom} onChange={(e) => setYearFrom(e.target.value.replace(/\D/g, '').slice(0, 4))} placeholder="any"
              aria-label="From year" className="w-16 bg-white border border-slate-300 rounded-lg px-2 py-1 text-xs" />
            to
            <input value={yearTo} onChange={(e) => setYearTo(e.target.value.replace(/\D/g, '').slice(0, 4))} placeholder="now"
              aria-label="To year" className="w-16 bg-white border border-slate-300 rounded-lg px-2 py-1 text-xs" />
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              disabled={!ready}
              onClick={() =>
                onRun({
                  topic,
                  population: population.name.trim() ? textToTerm(population) : null,
                  rows: goodRows.map(textToTerm),
                  cols: goodCols.map(textToTerm),
                  row_title: rowTitle.trim() || 'Rows',
                  col_title: colTitle.trim() || 'Columns',
                  year_from: yearFrom ? Number(yearFrom) : null,
                  year_to: yearTo ? Number(yearTo) : null,
                })
              }
              className="px-4 py-2 rounded-lg text-sm font-semibold bg-teal-600 hover:bg-teal-500 text-white disabled:opacity-50"
            >
              Count the records ({goodRows.length} × {goodCols.length})
            </button>
            <span className="text-[11px] text-slate-500">
              About {queries} database queries, roughly {Math.max(1, Math.round(queries * 0.4 / 60))} minute
              {Math.round(queries * 0.4 / 60) > 1 ? 's' : ''}.
              {demo?.demo ? ' The demo allows one gap map a day per address.' : ''}
            </span>
          </div>
        </>
      )}
    </div>
  );
}

const pubmedLink = (q) => `https://pubmed.ncbi.nlm.nih.gov/?term=${encodeURIComponent(q)}`;
const withDesign = (base, flt) => (flt ? `(${base}) AND ${flt}` : base);
const fmt = (n) => (n === null || n === undefined ? '–' : n.toLocaleString());

function shade(n, maxVal) {
  if (n === null || n === undefined) return 'bg-slate-100 text-slate-400';
  if (n === 0) return 'bg-rose-50 text-rose-800';
  const r = Math.log10(n + 1) / Math.log10(maxVal + 1);
  if (r < 0.25) return 'bg-teal-50 text-teal-900';
  if (r < 0.5) return 'bg-teal-100 text-teal-900';
  if (r < 0.75) return 'bg-teal-200 text-teal-950';
  return 'bg-teal-400 text-white';
}

function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function toCsv(m) {
  const designs = m.designs.map((d) => d.key);
  const head = ['row', 'column', ...designs.map((d) => `pubmed_${d}`), 'openalex_all', 'flags', 'pubmed_query', 'openalex_query', 'searched_at'];
  const esc = (v) => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const lines = [head.join(',')];
  m.cells.forEach((c) => {
    lines.push(
      [
        m.rows[c.row].name,
        m.cols[c.col].name,
        ...designs.map((d) => c.counts?.[d]?.pubmed ?? ''),
        c.openalex_all ?? '',
        c.flags.join(' '),
        c.pubmed_query,
        c.openalex_query,
        m.searched_at,
      ]
        .map(esc)
        .join(',')
    );
  });
  return lines.join('\n') + '\n';
}

export function GapMapView({ map }) {
  const [design, setDesign] = useState('all');
  const [sel, setSel] = useState(null);
  const cellAt = useMemo(() => {
    const o = {};
    map.cells.forEach((c) => (o[`${c.row}:${c.col}`] = c));
    return o;
  }, [map]);
  const val = (c) => (design === 'openalex' ? c.openalex_all : c.counts?.[design]?.pubmed);
  const maxVal = Math.max(1, ...map.cells.map((c) => val(c) || 0));
  const selected = sel ? cellAt[sel] : null;
  const flagText = (f) =>
    ({
      none: 'No records in any database',
      none_pubmed: 'None in PubMed',
      sparse: 'Few records',
      no_trials: 'No randomised trials',
      unsynthesised: 'Trials but no systematic review',
    }[f] || f);
  const slug = (map.topic || 'evidence-gap-map').toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 40);

  return (
    <div className="space-y-4" aria-label="Evidence gap map">
      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-3">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="text-base font-bold text-slate-900">Evidence gap map{map.topic ? `: ${map.topic}` : ''}</h2>
            <p className="text-xs text-slate-600 mt-0.5">
              {map.population ? `Population: ${map.population.name}. ` : ''}
              {map.year_from || map.year_to ? `Published ${map.year_from || '…'}–${map.year_to || 'now'}. ` : ''}
              Counted {map.searched_at}: {map.queries_run} database queries
              {map.queries_failed ? `, ${map.queries_failed} failed (shown as –)` : ''}, {Math.round(map.duration_s)} s.
            </p>
          </div>
          <div className="flex gap-2">
            <button type="button" onClick={() => download(`${slug}.csv`, toCsv(map), 'text/csv')}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-white">Download CSV</button>
            <button type="button" onClick={() => download(`${slug}.json`, JSON.stringify(map, null, 1), 'application/json')}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-white border border-slate-300 text-slate-700">JSON</button>
          </div>
        </div>
        {(map.notes || []).map((n, k) => (
          <p key={k} className="text-[11px] text-rose-800 bg-rose-50 border border-rose-200 rounded-lg px-2.5 py-1.5">{n}</p>
        ))}
        <p className="text-[11px] text-amber-900 bg-amber-50 border border-amber-200 rounded-lg px-2.5 py-1.5">
          Counts are records that match a query, not screened studies: a broad term over-counts and a missing synonym
          under-counts. Open a cell&apos;s query in PubMed to check what it finds.
        </p>
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          <span className="text-slate-500 font-medium">Show:</span>
          {[...map.designs.map((d) => [d.key, `${d.label} (PubMed)`]), ['openalex', 'All records (OpenAlex)']].map(([k, l]) => (
            <button key={k} type="button" onClick={() => setDesign(k)}
              className={`px-2.5 py-1 rounded-lg border font-semibold ${design === k ? 'bg-teal-600 text-white border-teal-600' : 'bg-white border-slate-300 text-slate-700'}`}>
              {l}
            </button>
          ))}
        </div>
        <div className="overflow-x-auto">
          <table className="border-collapse text-xs">
            <thead>
              <tr>
                <th className="p-2 text-left text-slate-500 font-semibold align-bottom">
                  {map.row_title} ↓ / {map.col_title} →
                </th>
                {map.cols.map((c, j) => (
                  <th key={j} className="p-2 text-left font-semibold text-slate-800 align-bottom min-w-[7rem]">
                    {c.name}
                    <div className="text-[10px] font-normal text-slate-500">{fmt(map.col_totals[j])} in PubMed</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {map.rows.map((r, i) => (
                <tr key={i}>
                  <th className="p-2 text-left font-semibold text-slate-800 min-w-[9rem]">
                    {r.name}
                    <div className="text-[10px] font-normal text-slate-500">{fmt(map.row_totals[i])} in PubMed</div>
                  </th>
                  {map.cols.map((_, j) => {
                    const c = cellAt[`${i}:${j}`];
                    const v = c ? val(c) : null;
                    return (
                      <td key={j} className="p-0.5">
                        <button
                          type="button"
                          onClick={() => setSel(`${i}:${j}`)}
                          aria-label={`${r.name} and ${map.cols[j].name}: ${fmt(v)}`}
                          className={`w-full text-left rounded-lg p-2 border ${sel === `${i}:${j}` ? 'ring-2 ring-teal-600' : 'border-transparent'} ${shade(v, maxVal)}`}
                        >
                          <div className="text-sm font-bold">{fmt(v)}</div>
                          <div className="flex flex-wrap gap-0.5 mt-0.5">
                            {(c?.flags || []).map((f) => (
                              <span key={f} className={`px-1 rounded border text-[9px] font-semibold ${FLAG_STYLE[f]}`}>{flagText(f)}</span>
                            ))}
                          </div>
                        </button>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {selected && (
        <div className="bg-white border border-teal-200 rounded-2xl p-5 shadow-sm space-y-2 text-xs">
          <h3 className="text-sm font-bold text-slate-900">
            {map.rows[selected.row].name} × {map.cols[selected.col].name}
          </h3>
          <table className="text-xs">
            <tbody>
              {map.designs.map((d) => (
                <tr key={d.key}>
                  <td className="pr-3 py-0.5 text-slate-600">{d.label}</td>
                  <td className="pr-3 font-semibold">{fmt(selected.counts?.[d.key]?.pubmed)}</td>
                  <td>
                    <a href={pubmedLink(withDesign(selected.pubmed_query, d.pubmed_filter))} target="_blank" rel="noreferrer"
                      className="text-indigo-700 hover:underline">open in PubMed ↗</a>
                  </td>
                </tr>
              ))}
              <tr>
                <td className="pr-3 py-0.5 text-slate-600">All records, OpenAlex</td>
                <td className="pr-3 font-semibold">{fmt(selected.openalex_all)}</td>
                <td />
              </tr>
            </tbody>
          </table>
          <div>
            <div className="font-semibold text-slate-700 mt-2">PubMed query (all records; each design adds its filter)</div>
            <code className="block bg-slate-50 border border-slate-200 rounded p-2 text-[11px] break-words">{selected.pubmed_query}</code>
            <div className="font-semibold text-slate-700 mt-2">OpenAlex title and abstract search</div>
            <code className="block bg-slate-50 border border-slate-200 rounded p-2 text-[11px] break-words">{selected.openalex_query}</code>
          </div>
        </div>
      )}

      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-2">
        <h3 className="text-sm font-bold text-slate-900">Where research looks thin ({map.gaps.length})</h3>
        {map.gaps.length === 0 ? (
          <p className="text-xs text-slate-600">No cell met a gap rule.</p>
        ) : (
          <ul className="text-xs space-y-1">
            {map.gaps.map((g, k) => (
              <li key={k}>
                <button type="button" onClick={() => setSel(`${g.row}:${g.col}`)} className="text-left hover:underline">
                  <span className={`px-1.5 py-0.5 mr-1.5 rounded border text-[10px] font-semibold ${FLAG_STYLE[g.flag]}`}>{g.text}</span>
                  {map.rows[g.row].name} × {map.cols[g.col].name}
                </button>
              </li>
            ))}
          </ul>
        )}
        <details className="text-[11px] text-slate-600">
          <summary className="cursor-pointer font-semibold text-slate-700">How this map was made</summary>
          <ul className="list-disc pl-5 space-y-1 mt-1">
            {map.method.map((m, k) => (
              <li key={k}>{m}</li>
            ))}
          </ul>
        </details>
      </div>
    </div>
  );
}

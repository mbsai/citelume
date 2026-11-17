import React, { useState } from 'react';

// After the guided review's search: how many studies met the criteria, and how many to analyse. Choosing fewer
// takes them from the same records (no new search); the rest are listed as eligible but not analysed.

const minutes = (n) => Math.round((1.5 + n / 20) * 2) / 2;

export default function AnalysisChoice({ summary, running, onAnalyse, onBack }) {
  const included = summary.included.length;
  const eligibleBased = summary.selection_method !== 'narrative';
  // An eligibility-based review can only go down (only screened studies can be chosen without the model); a
  // narrative review can be re-ranked up to the ceiling
  const max = eligibleBased ? included : Math.min(summary.max_study_limit, Math.max(summary.eligible, included));
  const [n, setN] = useState(included);
  const value = Math.max(Math.min(n || 0, max), Math.min(5, max));
  const notAnalysed = Math.max(0, summary.eligible - value);
  return (
    <div className="bg-emerald-50/60 border border-emerald-200 rounded-2xl p-5 space-y-3" aria-label="Choose how many studies to analyse">
      <h2 className="text-base font-bold text-slate-900">The search is done. How many studies should be analysed?</h2>
      <ul className="text-sm text-slate-700 space-y-0.5">
        <li>
          {summary.records_found.toLocaleString()} records found; {summary.unique_records.toLocaleString()} after merging duplicates.
        </li>
        <li>
          <strong>{summary.eligible.toLocaleString()}</strong> {eligibleBased ? 'met your eligibility criteria' : 'were eligible and ranked'}
          {eligibleBased && summary.eligible > included
            ? `; the ${included} with the highest scores are included (the most this install analyses for this review type is ${summary.max_study_limit}).`
            : eligibleBased
            ? '; all of them are included.'
            : `; the ranking chose ${included}.`}
        </li>
        {(summary.imports || []).map((imp) => (
          <li key={imp.filename}>
            Imported from {imp.source}: {imp.records.toLocaleString()} records ({imp.filename}).
          </li>
        ))}
        {(summary.user_pdfs || []).map((pdf) => (
          <li key={pdf.filename}>
            {pdf.filename}: {pdf.matched_record_id ? 'full text added to a study the search found' : 'added as a study'}
            {pdf.force_include ? ` (included regardless of the criteria: ${pdf.reason})` : ''}.
          </li>
        ))}
      </ul>
      <div className="flex flex-wrap items-center gap-3 text-sm text-slate-700">
        <label className="inline-flex items-center gap-2">
          Analyse
          <input
            type="number"
            min={Math.min(5, max)}
            max={max}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            className="w-20 bg-white border border-slate-300 rounded-lg px-2 py-1 text-sm"
            aria-label="Number of studies to analyse"
          />
          studies (5 to {max})
        </label>
        <span className="text-xs text-slate-500">
          About {minutes(value)} minutes with a free hosted model; every analysed study is read by the model.
        </span>
      </div>
      {notAnalysed > 0 && (
        <p className="text-xs text-slate-600">
          {eligibleBased
            ? `${notAnalysed} eligible ${notAnalysed === 1 ? 'study' : 'studies'} will be listed but not analysed.`
            : `${notAnalysed} eligible ${notAnalysed === 1 ? 'study is' : 'studies are'} not chosen by the ranking; they are counted under "How the studies were chosen".`}
        </p>
      )}
      <div className="flex gap-2">
        <button type="button" onClick={onBack} className="px-4 py-2 rounded-lg text-sm font-semibold bg-white border border-slate-300 text-slate-700 hover:bg-slate-50">
          Back to the set-up
        </button>
        <button
          type="button"
          disabled={running}
          onClick={() => onAnalyse(value)}
          className="px-4 py-2 rounded-lg text-sm font-semibold bg-indigo-600 hover:bg-indigo-500 text-white disabled:opacity-50"
        >
          {running ? 'Analysing...' : `Analyse ${value} ${value === 1 ? 'study' : 'studies'}`}
        </button>
      </div>
    </div>
  );
}

import React, { useState } from 'react';

// After the guided review's search: how many studies met the criteria, and how many to analyse. Choosing fewer
// takes them from the same records (no new search); the rest are listed as eligible but not analysed.

const minutes = (n) => Math.round((1.5 + n / 20) * 2) / 2;

// The model's screening decisions, for the user to confirm or overturn (each change needs a reason)
function ScreeningReview({ report, onApply, busy }) {
  const decisions = (report?.screening_decisions || []).filter((d) => d.screened);
  const [changes, setChanges] = useState({});
  const [show, setShow] = useState('excluded');
  if (!decisions.length) return null;
  const excluded = decisions.filter((d) => !(d.user_override ?? d.include));
  const included = decisions.filter((d) => d.user_override ?? d.include);
  const list = show === 'excluded' ? excluded : included;
  const pending = Object.entries(changes).filter(([, c]) => c.on);
  const ready = pending.length > 0 && pending.every(([, c]) => c.reason.trim());
  return (
    <div className="border-t border-emerald-200 pt-3 space-y-2">
      <h3 className="text-sm font-bold text-slate-800">Review the model&apos;s screening</h3>
      <p className="text-xs text-slate-600">
        The model ({report.screening_model}) answered each criterion for {decisions.length} records:{' '}
        {(report.screening_criteria || []).join('; ')}. A record is excluded if any criterion is clearly not met;
        &quot;unclear&quot; never excludes. Change any decision you disagree with; each change needs a reason and is recorded.
      </p>
      <div className="flex gap-2 text-xs">
        {[
          ['excluded', `Excluded by the model (${excluded.length})`],
          ['included', `Kept by the model (${included.length})`],
        ].map(([k, l]) => (
          <button
            key={k}
            type="button"
            onClick={() => setShow(k)}
            className={`px-2.5 py-1 rounded-lg border font-semibold ${show === k ? 'bg-slate-800 text-white border-slate-800' : 'bg-white border-slate-300 text-slate-700'}`}
          >
            {l}
          </button>
        ))}
      </div>
      <div className="max-h-80 overflow-auto space-y-1.5">
        {list.map((d) => {
          const c = changes[d.record_id] || { on: false, reason: '' };
          const set = (patch) => setChanges((prev) => ({ ...prev, [d.record_id]: { ...c, ...patch } }));
          return (
            <div key={d.record_id} className="text-xs border border-slate-200 rounded-lg p-2 bg-white space-y-1">
              <div className="font-medium text-slate-800">{d.title}</div>
              <div className="text-slate-600">
                {d.failed_criterion ? (
                  <>
                    <span className="text-rose-700 font-semibold">Not met: {d.failed_criterion}.</span> {d.reason}
                  </>
                ) : (
                  <>Meets the criteria (answers: {(d.answers || []).join(', ')}). {d.reason}</>
                )}
                {d.user_override != null && <span className="ml-1 text-indigo-700">(you changed this: {d.override_reason})</span>}
              </div>
              <label className="inline-flex items-center gap-1.5 text-slate-700">
                <input type="checkbox" checked={c.on} onChange={(e) => set({ on: e.target.checked })} />
                {show === 'excluded' ? 'Include it' : 'Exclude it'}
              </label>
              {c.on && (
                <input
                  value={c.reason}
                  onChange={(e) => set({ reason: e.target.value })}
                  placeholder="reason (recorded)"
                  className="ml-2 bg-white border border-slate-300 rounded px-2 py-0.5 text-xs w-72"
                />
              )}
            </div>
          );
        })}
      </div>
      <button
        type="button"
        disabled={!ready || busy}
        onClick={() =>
          onApply(
            pending.map(([record_id, c]) => ({
              record_id,
              include: !(decisions.find((d) => d.record_id === record_id)?.user_override ?? decisions.find((d) => d.record_id === record_id)?.include),
              reason: c.reason.trim(),
            }))
          )
        }
        className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-white disabled:opacity-40"
      >
        Apply my {pending.length || ''} change{pending.length === 1 ? '' : 's'}
      </button>
    </div>
  );
}

export default function AnalysisChoice({ summary, running, onAnalyse, onBack, onOverride }) {
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
      {notAnalysed > 0 && eligibleBased && (
        <div className="text-xs bg-amber-50 border border-amber-300 text-amber-900 rounded-lg p-2.5" role="alert">
          <strong>
            The review will not be complete: {value} of {summary.eligible} eligible studies analysed.
          </strong>{' '}
          A {summary.selection_method === 'scoping' ? 'scoping' : 'systematic'} review includes every eligible study.
          The other {notAnalysed} will be listed as eligible but not synthesised, and the review will be labelled a
          PRISMA-informed rapid review on the page and in every export.
          {value < included && ' Analyse all of them for a complete review.'}
          {included < summary.eligible &&
            ` This install analyses at most ${summary.max_study_limit} studies for this review type; a local install can raise the limit.`}
        </div>
      )}
      {notAnalysed > 0 && !eligibleBased && (
        <p className="text-xs text-slate-600">
          {`${notAnalysed} eligible ${notAnalysed === 1 ? 'study is' : 'studies are'} not chosen by the ranking; they are counted under "How the studies were chosen".`}
        </p>
      )}
      <ScreeningReview report={summary.selection_report} onApply={onOverride} busy={running} />
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

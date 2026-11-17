import React, { useState } from 'react';
import { editRiskOfBias } from '../utils/api';

// Suggested risk-of-bias judgements for the reviewer to confirm: a traffic-light table per tool, the quote behind
// each judgement (checked by code against the study's text), the RoB 2 answers for trials, and the reviewer's
// changes, each with a reason.

const ROB_STYLE = {
  Low: 'bg-emerald-500 text-white',
  'Some concerns': 'bg-amber-400 text-slate-900',
  High: 'bg-rose-600 text-white',
  'No information': 'bg-slate-300 text-slate-700',
};
const ROB_SYMBOL = { Low: '+', 'Some concerns': '!', High: '−', 'No information': '?' };
const JUDGEMENTS = ['Low', 'Some concerns', 'High', 'No information'];
const ANSWER_LABEL = { Y: 'Yes', PY: 'Probably yes', PN: 'Probably no', N: 'No', NI: 'No information', NA: 'Not applicable' };

function RobDot({ judgement, title }) {
  return (
    <span
      title={title ? `${title}: ${judgement}` : judgement}
      className={`inline-flex w-6 h-6 rounded-full items-center justify-center text-xs font-bold ${ROB_STYLE[judgement] || ROB_STYLE['No information']}`}
    >
      {ROB_SYMBOL[judgement] || '?'}
    </span>
  );
}

function Quote({ text, verified }) {
  if (!text) return <span className="text-slate-400">no quote</span>;
  return (
    <span>
      &ldquo;{text}&rdquo;{' '}
      <span className={verified ? 'text-emerald-700' : 'text-rose-700'}>
        {verified ? '✓ found in the study' : '✗ not found in the study (not used)'}
      </span>
    </span>
  );
}

function StudyDetail({ paper, corpusId, onChange }) {
  const rob = paper.elicit.risk_of_bias;
  const [domain, setDomain] = useState(rob.domains[0]?.domain || '');
  const [judgement, setJudgement] = useState('Low');
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const save = async () => {
    setBusy(true);
    setError('');
    try {
      onChange(paper.id, await editRiskOfBias(corpusId, { record_id: paper.id, domain, judgement, reason }));
      setReason('');
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="bg-slate-50 border-t border-slate-200 px-3 py-2 space-y-2 text-[11px] text-slate-700">
      <ul className="space-y-1">
        {rob.domains.map((d, i) => (
          <li key={d.domain}>
            <strong>
              D{i + 1} {d.domain}: {d.judgement}.
            </strong>{' '}
            <Quote text={d.quote} verified={d.quote_verified} />
          </li>
        ))}
      </ul>
      {rob.signalling?.length > 0 && (
        <details>
          <summary className="cursor-pointer font-semibold text-slate-700">
            RoB 2 signalling answers (the domains above are computed from these)
          </summary>
          <ul className="mt-1 space-y-0.5">
            {rob.signalling.map((s) => (
              <li key={s.id}>
                {s.id} {s.question} <strong>{ANSWER_LABEL[s.answer] || s.answer}</strong>
                {s.quote && (
                  <>
                    {' '}
                    — <Quote text={s.quote} verified={s.quote_verified} />
                  </>
                )}
              </li>
            ))}
          </ul>
        </details>
      )}
      {rob.edits?.length > 0 && (
        <ul className="text-indigo-800">
          {rob.edits.map((e, i) => (
            <li key={i}>
              You changed {e.domain} from {e.before} to {e.after}: {e.reason}
            </li>
          ))}
        </ul>
      )}
      {corpusId && (
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-semibold">Change a judgement:</span>
          <select value={domain} onChange={(e) => setDomain(e.target.value)} className="border border-slate-300 rounded px-1 py-0.5" aria-label="Domain">
            {rob.domains.map((d) => (
              <option key={d.domain}>{d.domain}</option>
            ))}
          </select>
          <select value={judgement} onChange={(e) => setJudgement(e.target.value)} className="border border-slate-300 rounded px-1 py-0.5" aria-label="Judgement">
            {JUDGEMENTS.map((j) => (
              <option key={j}>{j}</option>
            ))}
          </select>
          <input
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="reason (recorded)"
            className="border border-slate-300 rounded px-2 py-0.5 w-64"
          />
          <button
            type="button"
            disabled={busy || !reason.trim()}
            onClick={save}
            className="px-2 py-0.5 rounded bg-slate-800 text-white font-semibold disabled:opacity-40"
          >
            Save
          </button>
          {error && <span className="text-rose-700">{error}</span>}
        </div>
      )}
    </div>
  );
}

export default function RiskOfBiasTable({ papers, corpusId, onCitationClick, onRobChange }) {
  const [open, setOpen] = useState(null);
  const withRob = (papers || []).filter((p) => p.elicit?.risk_of_bias);
  const assessed = withRob.filter((p) => p.elicit.risk_of_bias.overall !== 'Not assessed');
  const notAssessed = withRob.filter((p) => p.elicit.risk_of_bias.overall === 'Not assessed');
  if (!withRob.length) return null;
  const groups = {};
  assessed.forEach((p) => {
    const t = p.elicit.risk_of_bias.tool;
    (groups[t] = groups[t] || []).push(p);
  });
  return (
    <div className="my-3 space-y-4">
      <p className="text-xs text-amber-900 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
        Suggested judgements for you to confirm. Each was made by an AI model from the full text and is backed by a quote
        that was checked against the study&apos;s text; an answer without such a quote counts as no information. Click a
        study to see the quotes and change a judgement.
      </p>
      {Object.entries(groups).map(([tool, ps]) => {
        const domains = ps[0].elicit.risk_of_bias.domains.map((d) => d.domain);
        return (
          <div key={tool} className="overflow-x-auto border border-slate-200 rounded-lg">
            <div className="px-3 py-1.5 bg-slate-50 border-b border-slate-200 text-xs font-semibold text-slate-700">
              {tool} · {ps.length} {ps.length === 1 ? 'study' : 'studies'}
            </div>
            <table className="text-xs w-full">
              <thead>
                <tr className="text-slate-600">
                  <th className="text-left px-2 py-1.5 font-semibold">Study</th>
                  {domains.map((d, i) => (
                    <th key={d} className="px-1 py-1.5 font-semibold text-center" title={d}>
                      D{i + 1}
                    </th>
                  ))}
                  <th className="px-2 py-1.5 font-semibold text-center">Overall</th>
                  <th className="text-left px-2 py-1.5 font-semibold">Main concern</th>
                </tr>
              </thead>
              <tbody>
                {ps.map((p) => {
                  const rob = p.elicit.risk_of_bias;
                  const isOpen = open === p.id;
                  return (
                    <React.Fragment key={p.id}>
                      <tr className="border-t border-slate-100 align-top">
                        <td className="px-2 py-1.5 whitespace-nowrap">
                          <button
                            type="button"
                            onClick={() => onCitationClick && onCitationClick(p.citation_index)}
                            className="font-mono text-indigo-700 font-semibold mr-1"
                          >
                            [{p.citation_index}]
                          </button>
                          {(p.authors?.[0] || p.source || '').split(' ').slice(-1)[0]} {p.year || ''}
                          <button
                            type="button"
                            onClick={() => setOpen(isOpen ? null : p.id)}
                            aria-expanded={isOpen}
                            className="ml-2 text-[11px] text-indigo-700 hover:underline"
                          >
                            {isOpen ? 'hide' : 'quotes'}
                          </button>
                        </td>
                        {rob.domains.map((d) => (
                          <td key={d.domain} className="px-1 py-1.5 text-center">
                            <RobDot judgement={d.judgement} title={d.domain} />
                          </td>
                        ))}
                        <td className="px-2 py-1.5 text-center">
                          <RobDot judgement={rob.overall} title="Overall" />
                        </td>
                        <td className="px-2 py-1.5 text-slate-600 min-w-[14rem]">
                          {rob.reason}
                          {rob.status === 'edited by the user' && <span className="text-indigo-700"> (changed by you)</span>}
                        </td>
                      </tr>
                      {isOpen && (
                        <tr>
                          <td colSpan={domains.length + 3}>
                            <StudyDetail paper={p} corpusId={corpusId} onChange={onRobChange || (() => {})} />
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
            <div className="px-3 py-1.5 border-t border-slate-100 text-[11px] text-slate-500">
              {domains.map((d, i) => `D${i + 1} ${d}`).join(' · ')}
            </div>
          </div>
        );
      })}
      {notAssessed.length > 0 && (
        <p className="text-xs text-slate-600">
          Not assessed (only the abstract was available): {notAssessed.map((p) => `[${p.citation_index}]`).join(', ')}.
        </p>
      )}
      <div className="flex flex-wrap gap-3 text-[11px] text-slate-600">
        {Object.keys(ROB_STYLE).map((j) => (
          <span key={j} className="inline-flex items-center gap-1">
            <RobDot judgement={j} /> {j}
          </span>
        ))}
      </div>
    </div>
  );
}

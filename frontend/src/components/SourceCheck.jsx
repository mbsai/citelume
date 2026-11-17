import React from 'react';

// One-click check of a claim against the study it cites: the sentences in that study's abstract (and open-access
// full text, when retrieved) that share the most words with the claim, and whether the claim's figures appear in
// the study at all. It is a reading aid: a close sentence does not prove the claim, so the reader judges.

const STOP = new Set(
  ('a an and are as at be been but by can did do does for from had has have in into is it its may more most no not ' +
    'of on or our over than that the their them then there these they this those to under up was were what when ' +
    'which while who will with within without also between both each other such study studies found showed reported ' +
    'results result evidence suggest suggests suggested associated association effect effects').split(' ')
);

const words = (t) => (t.toLowerCase().match(/[a-z0-9][a-z0-9-]*/g) || []).filter((w) => w.length > 2 && !STOP.has(w));
const figures = (t) => [...new Set((t.match(/\d+(?:[.·]\d+)?%?/g) || []).filter((n) => /[.%]/.test(n) || n.length >= 3))];

function sentencesOf(paper) {
  const parts = [
    ['abstract', paper.abstract || ''],
    ...Object.entries(paper.full_text_sections || {}).map(([k, v]) => [`full text, ${k}`, v || '']),
  ];
  const out = [];
  parts.forEach(([where, text]) => {
    text
      .replace(/\s+/g, ' ')
      .split(/(?<=[.!?])\s+(?=[A-Z0-9(])/)
      .forEach((s) => {
        if (s.length > 25) out.push({ where, text: s.trim() });
      });
  });
  return out;
}

export function closestSentences(claim, paper, n = 2) {
  const claimWords = new Set(words(claim.replace(/\[[\d,\s]+\]/g, ' ')));
  if (!claimWords.size) return [];
  return sentencesOf(paper)
    .map((s) => {
      const shared = [...new Set(words(s.text))].filter((w) => claimWords.has(w));
      return { ...s, shared, score: shared.length / Math.sqrt(claimWords.size) };
    })
    .filter((s) => s.shared.length >= 2)
    .sort((a, b) => b.score - a.score)
    .slice(0, n);
}

function Highlighted({ text, shared }) {
  const set = new Set(shared);
  return text.split(/(\s+)/).map((tok, i) => {
    const w = tok.toLowerCase().replace(/[^a-z0-9-]/g, '');
    return set.has(w) ? (
      <mark key={i} className="bg-amber-100 text-slate-900 rounded px-0.5">
        {tok}
      </mark>
    ) : (
      <React.Fragment key={i}>{tok}</React.Fragment>
    );
  });
}

export default function SourceCheck({ claim, paper, onClose, onShowReference }) {
  if (!paper) return null;
  const text = [paper.abstract || '', ...Object.values(paper.full_text_sections || {})].join(' ').replace(/·/g, '.');
  const claimFigures = figures(claim.replace(/\[[\d,\s]+\]/g, ' ').replace(/·/g, '.'));
  const matches = closestSentences(claim, paper);
  // Full text is read by the analysis but not sent to the page, so this check searches the abstract
  const hasFullText = !!paper.full_text_source;
  return (
    <div
      role="dialog"
      aria-label={`Source check for reference ${paper.citation_index}`}
      className="fixed inset-x-3 bottom-3 md:inset-x-auto md:right-4 md:bottom-4 md:w-[30rem] max-h-[70vh] overflow-auto z-50 bg-white border border-slate-300 rounded-2xl shadow-2xl p-4 space-y-3 text-sm"
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-xs font-bold uppercase tracking-wide text-indigo-700">Check against the source</div>
          <div className="font-semibold text-slate-900">
            [{paper.citation_index}] {paper.title}
          </div>
          <div className="text-xs text-slate-500">
            {(paper.authors || [])[0] || paper.source} {paper.year || ''} · this check searches the abstract
            {hasFullText ? ' (the analysis also read the full text)' : ' (no full text was available)'}
          </div>
        </div>
        <button type="button" onClick={onClose} aria-label="Close" className="text-slate-500 hover:text-slate-900 text-lg leading-none">
          ×
        </button>
      </div>

      <div className="text-xs bg-slate-50 border border-slate-200 rounded-lg p-2">
        <span className="font-semibold text-slate-700">Claim:</span> {claim}
      </div>

      {claimFigures.length > 0 && (
        <div className="text-xs space-x-1">
          <span className="font-semibold text-slate-700">Figures in the claim:</span>
          {claimFigures.map((f) => {
            const found = text.includes(f);
            return (
              <span
                key={f}
                className={`inline-block px-1.5 py-0.5 rounded border ${
                  found ? 'bg-emerald-50 border-emerald-200 text-emerald-800' : 'bg-rose-50 border-rose-200 text-rose-800'
                }`}
              >
                {f} {found ? 'found in this study' : 'not in this study'}
              </span>
            );
          })}
        </div>
      )}

      <div className="space-y-2">
        <div className="text-xs font-semibold text-slate-700">Closest sentences in this study (shared words highlighted):</div>
        {matches.length === 0 && (
          <p className="text-xs text-slate-600">
            No sentence in this study shares enough words with the claim. The claim may rest on another cited study,
            or may need checking in the full paper.
          </p>
        )}
        {matches.map((m, i) => (
          <blockquote key={i} className="text-xs border-l-4 border-amber-300 pl-2 text-slate-800">
            <Highlighted text={m.text} shared={m.shared} />
            <div className="text-[10px] text-slate-500 mt-0.5">from the {m.where}</div>
          </blockquote>
        ))}
      </div>

      <p className="text-[11px] text-slate-500">
        Sentences are matched by shared words, so a close sentence is not proof that the claim is right. Read it and
        judge.
      </p>
      <div className="flex gap-2">
        {paper.url && (
          <a href={paper.url} target="_blank" rel="noreferrer" className="text-xs font-semibold text-indigo-700 hover:underline">
            Open the paper ↗
          </a>
        )}
        <button type="button" onClick={onShowReference} className="text-xs font-semibold text-indigo-700 hover:underline">
          Show in the reference list
        </button>
      </div>
    </div>
  );
}

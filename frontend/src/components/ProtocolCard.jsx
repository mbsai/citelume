import React from 'react';

// The review protocol shown with the results: what was decided before searching, and by whom.

const ROLE_LABELS = {
  population: 'Population',
  intervention: 'Intervention',
  exposure: 'Exposure',
  comparison: 'Comparison',
  outcome: 'Outcome',
  topic: 'Topic',
  method: 'Method',
};
const TYPE_LABELS = {
  narrative: 'Narrative review',
  systematic_prisma: 'Systematic-style review (PRISMA 2020)',
  scoping: 'Scoping review (PRISMA-ScR)',
};
const DESIGN_LABELS = {
  rct: 'randomised trials',
  controlled_observational: 'cohort and controlled studies',
  descriptive_observational: 'cross-sectional studies',
  experimental_lab: 'laboratory studies',
  computational: 'computational studies',
  theoretical: 'theoretical studies',
  qualitative: 'qualitative studies',
  primary_other: 'other primary research',
  case_report: 'case reports',
  evidence_synthesis: 'systematic reviews',
  review: 'narrative reviews',
  commentary: 'commentaries',
};

function Who({ setBy }) {
  if (setBy === 'user') return <span className="ml-1 text-[10px] px-1.5 py-0.5 rounded bg-indigo-50 border border-indigo-200 text-indigo-700">set by you</span>;
  if (setBy === 'suggestion')
    return <span className="ml-1 text-[10px] px-1.5 py-0.5 rounded bg-amber-50 border border-amber-200 text-amber-800">model suggestion you accepted</span>;
  return null;
}

export default function ProtocolCard({ protocol }) {
  if (!protocol) return null;
  const e = protocol.eligibility || {};
  const years = e.year_from || e.year_to ? `${e.year_from || 'earliest'} to ${e.year_to || 'latest'}` : 'no limit';
  const designs = (e.study_designs || []).map((d) => DESIGN_LABELS[d] || d).join(', ') || 'all designs';
  const reviews = { default: protocol.review_type === 'systematic_prisma' ? 'excluded (cited in the Discussion)' : 'included', include: 'included', exclude: 'excluded' }[e.reviews] || e.reviews;
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-3">
      <div>
        <h2 className="text-base font-bold text-slate-900">Review Protocol</h2>
        <p className="text-xs text-slate-500">
          Decided before searching. The searches and the choice of studies followed it, and the same protocol on the
          same records always gives the same studies.
        </p>
      </div>
      {protocol.accepted_defaults && (
        <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
          The proposed settings were accepted without changes.
        </p>
      )}
      <dl className="text-sm text-slate-700 space-y-2">
        <div>
          <dt className="font-semibold text-slate-800 inline">Review type: </dt>
          <dd className="inline">
            {TYPE_LABELS[protocol.review_type] || protocol.review_type}. <span className="text-slate-500 text-xs">{protocol.review_type_reason}</span>
          </dd>
        </div>
        <div>
          <dt className="font-semibold text-slate-800">Parts of the question</dt>
          <dd>
            <ul className="text-sm space-y-0.5 mt-0.5">
              {(protocol.concepts || []).map((c, i) => (
                <li key={i}>
                  <span className="text-slate-500">{ROLE_LABELS[c.role] || c.role}:</span> <strong>{c.name}</strong>
                  {c.main && <span className="text-xs text-slate-500"> (main topic)</span>}
                  {c.synonyms?.length > 0 && <span className="text-slate-600"> — also {c.synonyms.join(', ')}</span>}
                  {c.role === 'comparison' && <span className="text-xs text-slate-500"> (recorded, not searched)</span>}
                  <Who setBy={c.set_by} />
                </li>
              ))}
            </ul>
          </dd>
        </div>
        {(protocol.populations_and_places || []).length > 0 && (
          <div>
            <dt className="font-semibold text-slate-800 inline">Every study must name: </dt>
            <dd className="inline">
              {protocol.populations_and_places.map((c, i) => (
                <span key={i}>
                  {i > 0 && '; '}
                  {c.name} <span className="text-xs text-slate-500">({(c.spellings || []).join(', ')})</span>
                  <Who setBy={c.set_by} />
                </span>
              ))}
            </dd>
          </div>
        )}
        {(protocol.aspects || []).length > 0 && (
          <div>
            <dt className="font-semibold text-slate-800 inline">Aspects searched separately: </dt>
            <dd className="inline">{protocol.aspects.map((a) => a.label).join('; ')}</dd>
          </div>
        )}
        <div>
          <dt className="font-semibold text-slate-800 inline">Eligibility: </dt>
          <dd className="inline">
            {designs}; years: {years}; existing reviews: {reviews}
            {(e.changed_by_user || []).length > 0 && <Who setBy="user" />}
          </dd>
        </div>
      </dl>
    </div>
  );
}

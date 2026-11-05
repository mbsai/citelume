import React, { useState } from 'react';

const CELL_CLASSIFICATION_STYLES = {
  'Direct evidence identified': {
    bg: 'bg-emerald-50/80 hover:bg-emerald-100/70 border-emerald-200 text-emerald-900',
    badge: 'bg-emerald-100 text-emerald-800 border-emerald-300',
    shortLabel: 'DIRECT EVIDENCE',
  },
  'Partial evidence identified': {
    bg: 'bg-amber-50/80 hover:bg-amber-100/70 border-amber-200 text-amber-900',
    badge: 'bg-amber-100 text-amber-800 border-amber-300',
    shortLabel: 'PARTIAL / SPARSE',
  },
  'Indirect evidence identified': {
    bg: 'bg-sky-50/80 hover:bg-sky-100/70 border-sky-200 text-sky-900',
    badge: 'bg-sky-100 text-sky-800 border-sky-300',
    shortLabel: 'INDIRECT ONLY',
  },
  'Conflicting evidence identified': {
    bg: 'bg-purple-50/80 hover:bg-purple-100/70 border-purple-300 text-purple-950',
    badge: 'bg-purple-100 text-purple-800 border-purple-300',
    shortLabel: 'CONFLICTING',
  },
  'No relevant evidence identified within the examined corpus': {
    bg: 'bg-rose-50/75 hover:bg-rose-100/65 border-rose-200 text-rose-900',
    badge: 'bg-rose-100 text-rose-800 border-rose-300',
    shortLabel: '0 IN CORPUS (SCOPE-BOUNDED)',
  },
  'Insufficient information to classify': {
    bg: 'bg-slate-100 hover:bg-slate-200/70 border-slate-300 text-slate-700',
    badge: 'bg-slate-200 text-slate-700 border-slate-300',
    shortLabel: 'INSUFFICIENT ACCESS',
  },
  'Not assessed': {
    bg: 'bg-slate-50 border-slate-200 text-slate-500',
    badge: 'bg-slate-100 text-slate-600 border-slate-200',
    shortLabel: 'NOT ASSESSED',
  },
};

const VERIFICATION_BADGE = {
  'Corroborated Within Search Scope': 'bg-indigo-50 text-indigo-800 border-indigo-200',
  'Partially Addressed': 'bg-amber-50 text-amber-800 border-amber-200',
  'Contradicted / Resolved by Retrieved Evidence': 'bg-emerald-50 text-emerald-800 border-emerald-200',
  'Candidate — Not Yet Verified': 'bg-slate-100 text-slate-700 border-slate-300',
  Inconclusive: 'bg-purple-50 text-purple-800 border-purple-200',
  'Verification Unavailable': 'bg-rose-50 text-rose-800 border-rose-200',
};

const PRIORITY_BADGE = {
  'High relevance to the research question': 'bg-rose-50 text-rose-800 border-rose-200',
  'Moderate relevance': 'bg-amber-50 text-amber-800 border-amber-200',
  'Contextual or exploratory relevance': 'bg-sky-50 text-sky-800 border-sky-200',
  'Insufficient information to prioritize': 'bg-slate-100 text-slate-600 border-slate-200',
};

export default function KnowledgeGapMatrix({ gapMatrix, papers = [] }) {
  const [selectedCell, setSelectedCell] = useState(
    gapMatrix?.top_opportunities?.[0] || gapMatrix?.cells?.[0]?.[0] || null
  );
  const [expandedGapIds, setExpandedGapIds] = useState({});
  const [activeSubView, setActiveSubView] = useState('gaps'); // 'gaps' | 'opportunities' | 'resolved'

  if (!gapMatrix || (!gapMatrix.rows?.length && !gapMatrix.subquestion_coverage?.length)) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-8 text-center text-slate-500">
        Run a research query to generate the Universal Evidence-Gap & Prioritization Matrix.
      </div>
    );
  }

  const toggleGapExpand = (gapId) => {
    setExpandedGapIds((prev) => ({ ...prev, [gapId]: !prev[gapId] }));
  };

  const structuredGaps = gapMatrix.structured_gaps || [];
  const resolvedGaps = gapMatrix.resolved_or_invalidated_gaps || [];
  const opportunities = gapMatrix.research_opportunities || [];
  const subqCoverage = gapMatrix.subquestion_coverage || [];
  const suppRecords = gapMatrix.supplementary_verification_records || [];
  const validation = gapMatrix.validation_report;

  return (
    <div className="space-y-6">
      {/* Top Banner: Scope Disclosure, Validation Status & Subquestion Coverage */}
      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-100 pb-3">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-sm font-bold text-slate-900">
                Universal Evidence-Gap Identification, Verification & Research Prioritization Engine
              </h2>
              {gapMatrix.domain_profile && (
                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-indigo-50 text-indigo-700 border border-indigo-200">
                  Domain Profile: {gapMatrix.domain_profile}
                </span>
              )}
              {validation && (
                <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-50 text-emerald-700 border border-emerald-200">
                  ✓ 5-Gate Adversarial & Cross-Section Verified
                </span>
              )}
            </div>
            <p className="text-xs text-slate-600 mt-1">
              {gapMatrix.corpus_adequacy_explanation ||
                'Distinguishes evidence-backed knowledge gaps (G1–G9) from methodological limitations, incomplete search scope, and constructive research opportunities.'}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-[11px]">
            <span className="px-2.5 py-1 rounded-lg bg-indigo-50 text-indigo-800 border border-indigo-200 font-semibold">
              Active Gaps: {structuredGaps.length}
            </span>
            <span className="px-2.5 py-1 rounded-lg bg-amber-50 text-amber-800 border border-amber-200 font-semibold">
              Research Opportunities: {opportunities.length}
            </span>
            {resolvedGaps.length > 0 && (
              <span className="px-2.5 py-1 rounded-lg bg-emerald-50 text-emerald-800 border border-emerald-200 font-semibold">
                Resolved in Verification: {resolvedGaps.length}
              </span>
            )}
          </div>
        </div>

        {/* Subquestion Evidence Coverage Strip (Phase 4) */}
        {subqCoverage.length > 0 && (
          <div className="space-y-2">
            <div className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
              Research-Question-Driven Evidence Coverage (Phase 4)
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {subqCoverage.map((sq) => (
                <div
                  key={sq.subquestion_id}
                  className="p-3 rounded-xl bg-slate-50 border border-slate-200 text-xs space-y-1"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-bold text-slate-900">
                      [{sq.subquestion_id}] {sq.subquestion_text}
                    </span>
                    <span
                      className={`px-2 py-0.5 rounded border text-[10px] font-semibold ${
                        sq.coverage_status === 'adequately_addressed'
                          ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                          : sq.coverage_status === 'conflicting_evidence'
                          ? 'bg-purple-50 text-purple-800 border-purple-200'
                          : sq.coverage_status === 'partially_addressed'
                          ? 'bg-amber-50 text-amber-800 border-amber-200'
                          : 'bg-rose-50 text-rose-700 border-rose-200'
                      }`}
                    >
                      {sq.coverage_status.replace(/_/g, ' ')}
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-600">
                    <strong>Evidence Units:</strong> {sq.relevant_unit_ids?.length || 0} (Refs:{' '}
                    {sq.relevant_study_indices?.map((i) => `[${i}]`).join(', ') || 'None'}) •{' '}
                    <strong>Claims:</strong> {sq.applicable_claim_ids?.join(', ') || 'None'}
                  </div>
                  {sq.missing_information?.length > 0 && (
                    <div className="text-[11px] text-amber-800">
                      <strong>Remaining Limitation:</strong> {sq.missing_information[0]}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Phase 12 Recommended Universal Knowledge Gap Matrix Table */}
      <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3">
          <div>
            <h3 className="text-sm font-bold text-slate-900">
              Structured Knowledge Gaps, Research Opportunities & Verification Audit Trail
            </h3>
            <p className="text-xs text-slate-500">
              Click any gap row to inspect its G1–G9 taxonomy classification, Section 1/2 evidence IDs, targeted verification queries, 6-dimension priority rationale, and 10-point adversarial check.
            </p>
          </div>
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => setActiveSubView('gaps')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition ${
                activeSubView === 'gaps'
                  ? 'bg-indigo-600 text-white border-indigo-600'
                  : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
              }`}
            >
              Verified / Active Knowledge Gaps ({structuredGaps.length})
            </button>
            <button
              type="button"
              onClick={() => setActiveSubView('opportunities')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition ${
                activeSubView === 'opportunities'
                  ? 'bg-indigo-600 text-white border-indigo-600'
                  : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
              }`}
            >
              Separated Research Opportunities ({opportunities.length})
            </button>
            <button
              type="button"
              onClick={() => setActiveSubView('resolved')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition ${
                activeSubView === 'resolved'
                  ? 'bg-indigo-600 text-white border-indigo-600'
                  : 'bg-slate-50 text-slate-700 border-slate-200 hover:bg-slate-100'
              }`}
            >
              Resolved / Invalidated Audit Trail ({resolvedGaps.length})
            </button>
          </div>
        </div>

        {activeSubView === 'gaps' && (
          <>
            {structuredGaps.length === 0 ? (
              <div className="p-6 rounded-xl bg-emerald-50/60 border border-emerald-200 text-xs text-emerald-900">
                <strong>No Unresolved Knowledge Gaps Fabricated:</strong>{' '}
                {gapMatrix.corpus_adequacy_explanation ||
                  'All candidate questions were adequately addressed by existing proofs/evidence or resolved during targeted verification.'}
              </div>
            ) : (
              <div className="overflow-x-auto border border-slate-200 rounded-xl">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-200 text-slate-700">
                      <th className="py-2.5 px-3 font-semibold min-w-[190px]">Knowledge Gap & Taxonomy</th>
                      <th className="py-2.5 px-3 font-semibold min-w-[190px]">What Is Known</th>
                      <th className="py-2.5 px-3 font-semibold min-w-[200px]">What Remains Unresolved</th>
                      <th className="py-2.5 px-3 font-semibold min-w-[150px]">Evidence Basis (Sec 1 & 2)</th>
                      <th className="py-2.5 px-3 font-semibold min-w-[170px]">Verification & Priority</th>
                      <th className="py-2.5 px-3 font-semibold min-w-[210px]">Research Direction</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-200">
                    {structuredGaps.map((g) => {
                      const isExp = !!expandedGapIds[g.gap_id];
                      return (
                        <React.Fragment key={g.gap_id}>
                          <tr
                            onClick={() => toggleGapExpand(g.gap_id)}
                            className="hover:bg-slate-50/90 transition align-top cursor-pointer"
                          >
                            <td className="py-3 px-3">
                              <div className="flex items-center gap-1.5">
                                <span className="font-mono font-bold text-indigo-600">[{g.gap_id}]</span>
                                <span className="px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-800 border border-indigo-200 text-[10px] font-bold">
                                  {g.taxonomy_code}: {g.primary_gap_type}
                                </span>
                              </div>
                              <div className="font-semibold text-slate-900 mt-1">{g.gap_title}</div>
                              <div className="text-[10px] text-slate-500 mt-1">
                                {isExp ? '▾ Hide verification & adversarial audit' : '▸ Inspect verification & priority details'}
                              </div>
                            </td>
                            <td className="py-3 px-3 text-slate-700 text-[11px]">{g.current_knowledge}</td>
                            <td className="py-3 px-3 text-slate-800 text-[11px]">
                              <div className="font-medium">{g.unresolved_issue}</div>
                              <div className="text-[10px] text-slate-500 mt-1">
                                <strong>Why it matters:</strong> {g.why_it_matters}
                              </div>
                            </td>
                            <td className="py-3 px-3 text-[11px]">
                              <div>
                                <strong>Studies:</strong>{' '}
                                {g.supporting_study_indices?.length > 0
                                  ? g.supporting_study_indices.map((i) => `[${i}]`).join(', ')
                                  : '0 in primary set'}
                              </div>
                              <div className="text-[10px] text-slate-500 mt-0.5">
                                <strong>Units:</strong> {g.supporting_evidence_unit_ids?.slice(0, 3).join(', ') || 'None'}
                              </div>
                              {g.related_claim_ids?.length > 0 && (
                                <div className="text-[10px] text-indigo-700 font-mono mt-0.5">
                                  Claims: {g.related_claim_ids.join(', ')}
                                </div>
                              )}
                            </td>
                            <td className="py-3 px-3 space-y-1.5">
                              <span
                                className={`inline-block px-2 py-0.5 rounded border text-[10px] font-semibold ${
                                  VERIFICATION_BADGE[g.verification_status] || VERIFICATION_BADGE['Candidate — Not Yet Verified']
                                }`}
                              >
                                {g.verification_status}
                              </span>
                              <span
                                className={`block px-2 py-0.5 rounded border text-[10px] font-medium ${
                                  PRIORITY_BADGE[g.priority_assessment] || PRIORITY_BADGE['Moderate relevance']
                                }`}
                              >
                                Priority: {g.priority_assessment}
                              </span>
                            </td>
                            <td className="py-3 px-3 text-[11px] text-slate-800">
                              <div>{g.recommended_study_or_approach}</div>
                              <div className="text-[10px] text-indigo-700 font-mono mt-1">
                                Verify: {g.verification_queries?.[0]}
                              </div>
                            </td>
                          </tr>
                          {isExp && (
                            <tr className="bg-slate-50/90">
                              <td colSpan={6} className="p-4">
                                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 text-[11px]">
                                  <div className="bg-white border border-slate-200 rounded-xl p-3 space-y-1.5">
                                    <div className="font-bold text-slate-900 border-b border-slate-100 pb-1">
                                      Phase 6 & 7: Targeted Gap Verification
                                    </div>
                                    <div>
                                      <strong>Verification Question:</strong> {g.verification_question}
                                    </div>
                                    <div>
                                      <strong>Executed Queries:</strong>{' '}
                                      <span className="font-mono text-indigo-700">
                                        {g.verification_queries?.join(' | ')}
                                      </span>
                                    </div>
                                    <div>
                                      <strong>Verification Outcome:</strong> {g.verification_outcome}
                                    </div>
                                    <div>
                                      <strong>Search Scope:</strong> {g.search_scope}
                                    </div>
                                  </div>
                                  <div className="bg-white border border-slate-200 rounded-xl p-3 space-y-1.5">
                                    <div className="font-bold text-slate-900 border-b border-slate-100 pb-1">
                                      Phase 11: Explainable Priority Evaluation
                                    </div>
                                    <div>
                                      <strong>Priority Category:</strong> {g.priority_evaluation?.priority_category}
                                    </div>
                                    <div>
                                      <strong>Consequence of Uncertainty:</strong>{' '}
                                      {g.priority_evaluation?.consequence_of_uncertainty}
                                    </div>
                                    <div>
                                      <strong>Feasibility Qualification:</strong>{' '}
                                      {g.priority_evaluation?.research_feasibility}
                                    </div>
                                    <div>
                                      <strong>Rationale:</strong> {g.priority_rationale}
                                    </div>
                                  </div>
                                  <div className="bg-white border border-slate-200 rounded-xl p-3 space-y-1.5">
                                    <div className="font-bold text-slate-900 border-b border-slate-100 pb-1">
                                      Phase 15: Adversarial Assumption Check
                                    </div>
                                    <div>
                                      <strong>Disposition:</strong>{' '}
                                      <span className="font-mono text-emerald-700">
                                        {g.adversarial_check?.disposition}
                                      </span>
                                    </div>
                                    <div>
                                      <strong>Challenge Summary:</strong> {g.adversarial_check?.challenge_summary}
                                    </div>
                                    <div>
                                      <strong>Source Access Note:</strong> {g.source_access_limitations}
                                    </div>
                                  </div>
                                </div>
                              </td>
                            </tr>
                          )}
                        </React.Fragment>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}

        {activeSubView === 'opportunities' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {opportunities.map((opp) => (
              <div key={opp.opportunity_id} className="p-4 rounded-xl bg-amber-50/40 border border-amber-200 space-y-2 text-xs">
                <div className="flex items-center justify-between">
                  <span className="font-bold text-amber-900">
                    [{opp.opportunity_id}] {opp.title}
                  </span>
                  <span className="px-2 py-0.5 rounded bg-amber-100 text-amber-800 text-[10px] font-semibold">
                    Constructive Research Opportunity
                  </span>
                </div>
                <p className="text-slate-700">{opp.description}</p>
                <div className="text-[11px] text-slate-600">
                  <strong>Methodological Note:</strong> {opp.distinction_note}
                </div>
                {opp.suggested_hf_models_or_datasets?.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 pt-1">
                    {opp.suggested_hf_models_or_datasets.map((m) => (
                      <a
                        key={m}
                        href={`https://huggingface.co/${m}`}
                        target="_blank"
                        rel="noreferrer"
                        className="px-2 py-0.5 rounded bg-white border border-amber-300 text-amber-900 font-mono text-[10px] hover:bg-amber-100"
                      >
                        {m} ↗
                      </a>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {activeSubView === 'resolved' && (
          <div className="space-y-3">
            {resolvedGaps.length === 0 ? (
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-600">
                No candidate gaps were invalidated during supplementary verification for this query.
              </div>
            ) : (
              resolvedGaps.map((rg) => (
                <div key={rg.gap_id} className="p-4 rounded-xl bg-emerald-50/50 border border-emerald-200 space-y-1.5 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-emerald-950">
                      [{rg.gap_id}] {rg.gap_title}
                    </span>
                    <span className="px-2 py-0.5 rounded bg-emerald-100 text-emerald-800 border border-emerald-300 text-[10px] font-bold">
                      {rg.verification_status}
                    </span>
                  </div>
                  <div className="text-slate-700">
                    <strong>Why Archived from Unresolved List:</strong> {rg.verification_outcome}
                  </div>
                  {suppRecords.filter((sr) => sr.gap_id === rg.gap_id).length > 0 && (
                    <div className="text-[11px] text-emerald-900 pt-1">
                      <strong>Resolving Supplementary Evidence (Outside Primary 20/60 Cap):</strong>{' '}
                      {suppRecords
                        .filter((sr) => sr.gap_id === rg.gap_id)
                        .map((sr) => `${sr.title} (${sr.year || 'n.d.'}, ${sr.persistent_id})`)
                        .join('; ')}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {/* Question-Adaptive Evidence Coverage Matrix Grid */}
      {gapMatrix.rows?.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <div className="lg:col-span-2 bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
            <div className="mb-4">
              <h3 className="text-sm font-bold text-slate-900">
                Question-Adaptive Evidence-Coverage Grid ({gapMatrix.row_dimension_title} × {gapMatrix.col_dimension_title})
              </h3>
              <p className="text-xs text-slate-600 mt-1">
                Visualizes independent evidence-unit coverage across domain-specific dimensions. Distinct colors and labels separate{' '}
                <span className="text-emerald-700 font-semibold">Direct Evidence</span>,{' '}
                <span className="text-amber-700 font-semibold">Partial/Sparse Coverage</span>,{' '}
                <span className="text-purple-700 font-semibold">Conflicting Evidence</span>, and{' '}
                <span className="text-rose-700 font-semibold">0-in-Corpus (Scope-Bounded)</span>.
              </p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-xs">
                <thead>
                  <tr>
                    <th className="p-3 bg-slate-100 border border-slate-200 text-left text-slate-600 font-semibold">
                      Approach ↓ \ Regime →
                    </th>
                    {gapMatrix.cols.map((col, cIdx) => (
                      <th
                        key={cIdx}
                        className="p-3 bg-slate-100 border border-slate-200 text-left text-slate-800 font-semibold min-w-[135px]"
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {gapMatrix.rows.map((rowName, rIdx) => (
                    <tr key={rIdx}>
                      <td className="p-3 bg-slate-50 border border-slate-200 font-semibold text-slate-800 min-w-[160px]">
                        {rowName}
                      </td>
                      {(gapMatrix.cells[rIdx] || []).map((cell, cIdx) => {
                        const isSelected = selectedCell && selectedCell.id === cell.id;
                        const covClass =
                          cell.coverage_classification ||
                          (cell.status === 'gap'
                            ? 'No relevant evidence identified within the examined corpus'
                            : cell.status === 'sparse'
                            ? 'Partial evidence identified'
                            : 'Direct evidence identified');
                        const cfg =
                          CELL_CLASSIFICATION_STYLES[covClass] ||
                          CELL_CLASSIFICATION_STYLES['Partial evidence identified'];
                        const unitCnt = cell.independent_unit_count ?? cell.paper_count ?? 0;

                        return (
                          <td
                            key={cIdx}
                            onClick={() => setSelectedCell(cell)}
                            className={`p-3 border cursor-pointer transition align-top ${cfg.bg} ${
                              isSelected ? 'ring-2 ring-indigo-600' : ''
                            }`}
                          >
                            <div className="flex items-center justify-between gap-1 mb-1">
                              <span className={`px-1.5 py-0.5 rounded border text-[9px] font-bold ${cfg.badge}`}>
                                {cfg.shortLabel}
                              </span>
                              <span className="text-[10px] font-semibold">
                                {unitCnt} unit{unitCnt === 1 ? '' : 's'}
                              </span>
                            </div>
                            {cell.paper_indices?.length > 0 ? (
                              <div className="text-[11px] font-mono">
                                Refs: {cell.paper_indices.map((i) => `[${i}]`).join(', ')}
                              </div>
                            ) : (
                              <div className="text-[10px] italic opacity-85">
                                {cell.verification_status || 'Scope-bounded'}
                              </div>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Selected Cell Evidence & Verification Inspector */}
          <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm flex flex-col justify-between">
            {selectedCell ? (
              <div className="space-y-4">
                <div>
                  <span className="inline-block px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase bg-indigo-50 text-indigo-800 border border-indigo-200">
                    {selectedCell.coverage_classification || selectedCell.status}
                  </span>
                  <h3 className="text-sm font-bold text-slate-900 mt-2">
                    {selectedCell.method_row} × {selectedCell.domain_col}
                  </h3>
                  <div className="text-[11px] text-slate-500 mt-1">
                    Primary Category: <strong>{selectedCell.gap_category}</strong> • Verification:{' '}
                    <strong>{selectedCell.verification_status}</strong>
                  </div>
                </div>

                <div className="space-y-3 text-xs border-t border-slate-200 pt-3">
                  <div>
                    <span className="text-slate-500 font-semibold block mb-1">
                      Evidence Basis & Deficiency Analysis:
                    </span>
                    <p className="text-slate-700 leading-relaxed">{selectedCell.why_gap_exists}</p>
                  </div>

                  <div>
                    <span className="text-indigo-700 font-semibold block mb-1">
                      Recommended Study / Verification Direction:
                    </span>
                    <p className="text-slate-800 leading-relaxed bg-indigo-50/50 p-3 rounded-lg border border-indigo-100">
                      {selectedCell.proposed_hypothesis}
                    </p>
                  </div>

                  {selectedCell.verification_search_query && (
                    <div>
                      <span className="text-slate-500 font-semibold block mb-1">
                        Targeted Verification Query:
                      </span>
                      <code className="block p-2 rounded bg-slate-100 text-indigo-800 text-[11px]">
                        {selectedCell.verification_search_query}
                      </code>
                    </div>
                  )}

                  {selectedCell.paper_indices?.length > 0 && (
                    <div>
                      <span className="text-slate-500 font-semibold block mb-1">
                        Studies in This Intersection ({selectedCell.independent_unit_count ?? selectedCell.paper_count}{' '}
                        independent unit(s)):
                      </span>
                      <ul className="space-y-1">
                        {selectedCell.paper_indices.map((idx) => {
                          const p = papers.find((item) => item.citation_index === idx);
                          return (
                            <li key={idx} className="text-slate-700">
                              <span className="text-indigo-700 font-mono font-bold mr-1">[{idx}]</span>
                              {p ? p.title : `Paper #${idx}`}
                            </li>
                          );
                        })}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
}

import React, { useState } from 'react';
import { downloadReportFromServer } from '../utils/api';
import ProtocolCard from './ProtocolCard';
import SourceCheck from './SourceCheck';
import RiskOfBiasTable from './RiskOfBiasTable';

// Each citation button passes the sentence it sits in, so the claim can be checked against the cited study
function renderCitationsOnly(text, onCitationClick, keyPrefix) {
  return text.split(/(?<=[.!?])(\s+)/).map((sentence, si) => (
    <React.Fragment key={`${keyPrefix}-s${si}`}>
      {renderSentenceCitations(sentence, onCitationClick, `${keyPrefix}-s${si}`)}
    </React.Fragment>
  ));
}

function renderSentenceCitations(sentence, onCitationClick, keyPrefix) {
  const parts = sentence.split(/(\[\d+(?:,\s*\d+)*\])/g);
  return parts.map((part, i) => {
    const idx = `${keyPrefix}-${i}`;
    const match = part.match(/^\[(\d+(?:,\s*\d+)*)\]$/);
    if (match) {
      const nums = match[1].split(',').map((s) => parseInt(s.trim(), 10));
      return (
        <span key={idx} className="inline-flex items-center gap-0.5 mx-0.5">
          {nums.map((num) => (
            <button
              key={num}
              onClick={() => onCitationClick && onCitationClick(num, sentence.trim())}
              className="px-1.5 py-0.5 text-[11px] font-mono font-semibold rounded bg-indigo-50 text-indigo-700 border border-indigo-200 hover:bg-indigo-100 transition"
              title={`Check this against study [${num}]`}
            >
              [{num}]
            </button>
          ))}
        </span>
      );
    }
    return <span key={idx}>{part}</span>;
  });
}

// Citations become buttons; **text** is shown in bold (used for the labelled parts of a PRISMA report)
export function renderCitedText(text, onCitationClick) {
  if (!text) return null;
  return text.split(/(\*\*[^*]+\*\*)/g).map((chunk, i) => {
    const bold = chunk.match(/^\*\*([^*]+)\*\*$/);
    if (bold) {
      return (
        <strong key={`b${i}`} className="font-semibold text-slate-900">
          {renderCitationsOnly(bold[1], onCitationClick, `b${i}`)}
        </strong>
      );
    }
    return <React.Fragment key={`t${i}`}>{renderCitationsOnly(chunk, onCitationClick, `t${i}`)}</React.Fragment>;
  });
}

const STAGE_LABELS = { appraisal: 'Appraisal', synthesis: 'Synthesis', reporting: 'Reporting', gap: 'Gaps' };

// PRISMA 2020 flow diagram for a systematic-style review: each stage with its exclusions beside it
function PrismaFlowFull({ flow }) {
  const box = 'rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs text-slate-800';
  const side = 'rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-[11px] text-slate-600';
  const arrow = <div className="text-center text-slate-400 text-xs leading-none py-1">▼</div>;
  const reasons = (obj) =>
    Object.entries(obj || {}).map(([r, n]) => (
      <div key={r}>
        {r}: <strong>{n}</strong>
      </div>
    ));
  const stage = (label, main, aside) => (
    <div className="grid grid-cols-[5.5rem_1fr] sm:grid-cols-[6.5rem_1fr_1fr] gap-2 items-stretch">
      <div className="rounded-md bg-indigo-600 text-white text-[10px] font-bold uppercase tracking-wider flex items-center justify-center text-center px-1">
        {label}
      </div>
      <div className={box}>{main}</div>
      <div className={`${side} col-start-2 sm:col-start-auto`}>{aside}</div>
    </div>
  );
  const bySource = Object.entries(flow.identified_by_source || {});
  const removedBefore = Object.values(flow.removed_before_screening || {}).reduce((a, b) => a + b, 0);
  const excludedElig = Object.values(flow.excluded_at_eligibility || {}).reduce((a, b) => a + b, 0);
  return (
    <div className="max-w-3xl mx-auto my-3 space-y-0">
      {stage(
        'Identification',
        <>
          <div className="font-semibold">Records identified: {flow.identified_total}</div>
          {bySource.map(([s, n]) => (
            <div key={s} className="text-[11px] text-slate-500">
              {s}: {n}
            </div>
          ))}
          {Object.entries(flow.identified_other_methods || {}).map(([s, n]) => (
            <div key={s} className="text-[11px] text-slate-700 mt-1">
              Identified by other methods ({s.toLowerCase()}): <strong>{n}</strong>
            </div>
          ))}
        </>,
        <>
          <div className="font-semibold text-slate-700">Removed before screening</div>
          <div>
            Duplicates: <strong>{flow.duplicates_removed}</strong>
          </div>
          {removedBefore > 0 && reasons(flow.removed_before_screening)}
        </>,
      )}
      {arrow}
      {stage(
        'Screening',
        <div className="font-semibold">Records screened (title & abstract): {flow.screened}</div>,
        <>
          <div className="font-semibold text-slate-700">Records excluded: {flow.excluded_at_screening}</div>
          {reasons(flow.excluded_at_screening_reasons)}
        </>,
      )}
      {arrow}
      {stage(
        'Eligibility',
        <div className="font-semibold">Records assessed for eligibility: {flow.assessed_for_eligibility}</div>,
        <>
          <div className="font-semibold text-slate-700">Excluded, with reasons: {excludedElig}</div>
          {reasons(flow.excluded_at_eligibility)}
        </>,
      )}
      {arrow}
      {stage(
        'Included',
        <>
          <div className="font-semibold">Eligible studies: {flow.eligible}</div>
          <div className="mt-1 font-semibold text-emerald-700">
            Studies included in the synthesis: {flow.included_in_synthesis || flow.included}
          </div>
          <div className="text-[11px] text-slate-500">Full text retrieved: {flow.full_text_retrieved}</div>
        </>,
        <>
          {flow.beyond_study_limit > 0 && (
            <div>
              Eligible but beyond the {flow.study_limit}-study limit: <strong>{flow.beyond_study_limit}</strong> (listed
              in the appendix)
            </div>
          )}
          {flow.excluded_at_extraction > 0 && (
            <div>
              Found not to address the question during data extraction: <strong>{flow.excluded_at_extraction}</strong>
            </div>
          )}
          {!flow.beyond_study_limit && !flow.excluded_at_extraction && <div>All eligible studies were synthesised.</div>}
        </>,
      )}
    </div>
  );
}

// PRISMA 2020-style flow of records, built from the pipeline's real counts
function PrismaFlow({ flow }) {
  if (flow.assessed_for_eligibility || flow.eligible) return <PrismaFlowFull flow={flow} />;
  const box = 'rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs text-slate-800 text-center';
  const arrow = <div className="text-center text-slate-400 text-xs leading-none py-0.5">▼</div>;
  const bySource = Object.entries(flow.identified_by_source || {})
    .map(([src, n]) => `${src}: ${n}`)
    .join(' · ');
  return (
    <div className="max-w-md mx-auto">
      <div className={box}>
        <div className="font-semibold">Records identified: {flow.identified_total}</div>
        {bySource && <div className="text-[11px] text-slate-500 mt-0.5">{bySource}</div>}
        {Object.entries(flow.identified_other_methods || {}).map(([s, n]) => (
          <div key={s} className="text-[11px] text-slate-700 mt-0.5">
            Identified by other methods ({s.toLowerCase()}): {n}
          </div>
        ))}
      </div>
      {arrow}
      <div className={box}>Duplicates removed: {flow.duplicates_removed}</div>
      {arrow}
      <div className={box}>
        Records screened (title & abstract): {flow.screened}
        <div className="text-[11px] text-slate-500 mt-0.5">Excluded by relevance ranking: {flow.excluded_at_screening}</div>
      </div>
      {arrow}
      <div className={`${box} border-emerald-400 bg-emerald-50 font-semibold`}>Studies included: {flow.included}</div>
    </div>
  );
}

// Review type, PRISMA flow, search strategy, and every source-cited rule applied to this search
function ReviewProtocolPanel({ protocol }) {
  const [showSearch, setShowSearch] = useState(false);
  const [showRules, setShowRules] = useState(false);
  if (!protocol) return null;
  const rules = protocol.rules_applied || [];
  const strategies = Object.entries(protocol.search_strategies || {});
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-4">
      <div>
        <h2 className="text-base font-bold text-slate-900">Review Method & Frameworks Applied</h2>
        <p className="text-xs text-slate-500 mt-0.5">
          {protocol.framework_label} · Findings organisation: {protocol.organization_label}
        </p>
      </div>

      {protocol.disclosure && (
        <div className="text-xs bg-amber-50 border border-amber-300 text-amber-900 rounded-lg px-3 py-2">
          {protocol.disclosure}
        </div>
      )}

      {protocol.flow && <PrismaFlow flow={protocol.flow} />}

      {(protocol.eligibility_criteria || []).length > 0 && (
        <div>
          <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-1">Eligibility & Screening</h3>
          <ul className="list-disc pl-5 text-xs text-slate-700 space-y-0.5">
            {protocol.eligibility_criteria.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        </div>
      )}

      {strategies.length > 0 && (
        <div>
          <button
            type="button"
            onClick={() => setShowSearch((v) => !v)}
            className="text-xs font-semibold text-indigo-700 hover:underline"
          >
            {showSearch ? '▲ Hide' : '▼ Show'} full search strategy ({strategies.length} sources)
          </button>
          {showSearch && (
            <div className="mt-2 space-y-2">
              {strategies.map(([src, queries]) => (
                <div key={src}>
                  <div className="text-xs font-semibold text-slate-800">{src}</div>
                  {queries.map((q) => (
                    <code key={q} className="block text-[11px] text-slate-600 bg-slate-50 rounded px-2 py-1 mt-1 break-words">
                      {q}
                    </code>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div>
        <button
          type="button"
          onClick={() => setShowRules((v) => !v)}
          className="text-xs font-semibold text-indigo-700 hover:underline"
        >
          {showRules ? '▲ Hide' : '▼ Show'} methodology rules applied ({rules.length})
        </button>
        {showRules && (
          <ul className="mt-2 space-y-1.5">
            {rules.map((r) => (
              <li key={r.rule_id} className="text-xs text-slate-700 border-l-2 border-indigo-200 pl-2">
                <span className="font-semibold text-slate-500">{STAGE_LABELS[r.stage] || r.stage}:</span> {r.text}{' '}
                <span className="text-slate-500">— {r.sources.join('; ')}</span>
                {r.enforcement !== 'prompt' && (
                  <span className="ml-1 px-1.5 py-0.5 rounded bg-emerald-50 border border-emerald-200 text-emerald-700 text-[10px] font-semibold">
                    checked by code
                  </span>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>

      {(protocol.framework_citations || []).length > 0 && (
        <div className="text-[11px] text-slate-500 space-y-0.5 border-t border-slate-100 pt-3">
          {protocol.framework_citations.map((c) => (
            <div key={c}>{c}</div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function SynthesisPanel({ data, onSelectPaper, onRobChange }) {
  const [bibStyle, setBibStyle] = useState('apa');
  const [exportingFormat, setExportingFormat] = useState('');
  const [highlightedRef, setHighlightedRef] = useState(null);
  const [sourceCheck, setSourceCheck] = useState(null);
  const [showDecomposition, setShowDecomposition] = useState(false);
  const [exportOpts, setExportOpts] = useState({
    include_evidence_table: true,
    include_gap_matrix: true,
    include_network_summary: true,
    include_hf_models: true,
  });

  if (!data) return null;
  const isPrisma = data.review_protocol?.framework_id === 'systematic_prisma';
  const {
    consensus = {},
    executive_answer,
    sections = [],
    bibliography = [],
    sub_queries = [],
    decomposition_plan,
  } = data;

  const handleExport = async (fmt) => {
    setExportingFormat(fmt);
    try {
      await downloadReportFromServer(fmt, data, exportOpts);
    } catch (err) {
      alert(`Export error: ${err.message}`);
    } finally {
      setExportingFormat('');
    }
  };

  const showReference = (idx) => {
    setHighlightedRef(idx);
    const el = document.getElementById(`bib-entry-${idx}`);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  // A citation opens the source check for the sentence it sits in; without a sentence it goes to the reference
  const handleCitationClick = (idx, sentence) => {
    const paper = (data.papers || []).find((p) => p.citation_index === idx);
    if (paper && onSelectPaper) onSelectPaper(paper);
    if (paper && sentence) {
      setHighlightedRef(idx);
      setSourceCheck({ idx, sentence });
    } else {
      showReference(idx);
    }
  };

  const toggleExportOpt = (key) => {
    setExportOpts((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div className="space-y-5">
      {/* Export & Report Configuration Bar */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200 mr-2">
              {data.mode === 'deep_research'
                ? 'Deep Research Literature Review (Up to 60 Papers)'
                : 'Standard Evidence Synthesis (Up to 20 Papers)'}
            </span>
            <span className="text-xs text-slate-600">
              Grounded across {bibliography.length} deduplicated papers with inline citations & full bibliography
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-slate-600 mr-1">Download Report:</span>
            {['txt', 'md', 'pdf'].map((fmt) => (
              <button
                key={fmt}
                onClick={() => handleExport(fmt)}
                disabled={exportingFormat === fmt}
                className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 hover:bg-indigo-700 text-white shadow-sm transition uppercase"
              >
                {exportingFormat === fmt ? `Exporting .${fmt}...` : `Download .${fmt}`}
              </button>
            ))}
          </div>
        </div>

        {/* Checkboxes for Optional Sections in Report Export */}
        <div className="flex flex-wrap items-center gap-3 pt-2 border-t border-slate-100 text-xs text-slate-700">
          <span className="font-semibold text-slate-500">Include in Downloaded Report:</span>
          <span className="px-2 py-0.5 rounded bg-slate-100 text-slate-600 font-medium">
            ✓ Cited Synthesis & Bibliography (Always Included)
          </span>
          {[
            ['include_evidence_table', 'Structured Evidence Table'],
            ['include_gap_matrix', 'Knowledge Gap Matrix'],
            ['include_network_summary', 'Literature Network Summary'],
            ['include_hf_models', 'Hugging Face Models'],
          ].map(([key, label]) => (
            <label
              key={key}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-50 hover:bg-indigo-50/50 border border-slate-200 cursor-pointer select-none"
            >
              <input
                type="checkbox"
                checked={!!exportOpts[key]}
                onChange={() => toggleExportOpt(key)}
                className="accent-indigo-600 rounded"
              />
              <span>{label}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Multi-API Query Decomposition Plan */}
      {decomposition_plan ? (
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-bold text-indigo-700 uppercase tracking-wider">
                How the question was searched
              </span>
              <span className="px-2 py-0.5 rounded bg-purple-50 text-purple-700 border border-purple-200 text-[11px] font-semibold">
                Domain: {decomposition_plan.detected_domain}
              </span>
              {(decomposition_plan.extracted_technical_entities || []).map((ent, i) => (
                <span
                  key={i}
                  className="px-2 py-0.5 rounded bg-white border border-slate-200 text-emerald-700 font-mono text-[11px]"
                >
                  {ent}
                </span>
              ))}
            </div>
            <button
              type="button"
              onClick={() => setShowDecomposition((v) => !v)}
              className="text-xs font-semibold text-indigo-700 hover:text-indigo-900 px-2.5 py-1 rounded bg-white border border-slate-200 shadow-sm"
            >
              {showDecomposition
                ? 'Hide the searches ▲'
                : `Show the ${decomposition_plan.search_facets?.length || 0} searches run on each database ▼`}
            </button>
          </div>

          {showDecomposition && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2 border-t border-slate-200">
              {(decomposition_plan.search_facets || []).map((facet, fIdx) => (
                <div
                  key={fIdx}
                  className="bg-white border border-slate-200 rounded-lg p-3 text-xs space-y-1.5 shadow-sm"
                >
                  <div className="font-bold text-slate-800 text-xs">{facet.facet_name}</div>
                  {facet.facet_intent && (
                    <div className="text-[11px] text-slate-500">{facet.facet_intent}</div>
                  )}
                  <div className="space-y-1 pt-1 font-mono text-[11px]">
                    <div>
                      <span className="text-blue-700 font-semibold">PubMed: </span>
                      <span className="text-slate-700">{facet.queries?.pubmed}</span>
                    </div>
                    <div>
                      <span className="text-purple-700 font-semibold">Semantic Scholar: </span>
                      <span className="text-slate-700">{facet.queries?.semantic_scholar}</span>
                    </div>
                    <div>
                      <span className="text-rose-700 font-semibold">arXiv: </span>
                      <span className="text-slate-700">{facet.queries?.arxiv}</span>
                    </div>
                    <div>
                      <span className="text-emerald-700 font-semibold">bioRxiv/medRxiv: </span>
                      <span className="text-slate-700">
                        {(facet.queries?.biorxiv_medrxiv || []).join(' | ')}
                      </span>
                    </div>
                    <div>
                      <span className="text-amber-700 font-semibold">Hugging Face: </span>
                      <span className="text-slate-700">{facet.queries?.hugging_face}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : null}

      {/* Evidence Agreement Meter */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div>
            <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
              Evidence Agreement Meter — {consensus.consensus_verdict || 'Synthesized'}
            </h3>
            <p className="text-xs text-slate-600 mt-1">
              {renderCitedText(consensus.consensus_summary, handleCitationClick)}
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs font-semibold">
            <span className="text-emerald-700">Supporting: {consensus.supporting_pct || 0}%</span>
            <span className="text-amber-700">Mixed: {consensus.mixed_pct || 0}%</span>
            <span className="text-rose-700">Contradicting: {consensus.contradicting_pct || 0}%</span>
          </div>
        </div>
        <div className="w-full h-3 rounded-full overflow-hidden bg-slate-100 flex border border-slate-200">
          <div
            style={{ width: `${consensus.supporting_pct || 0}%` }}
            className="bg-emerald-500 transition-all duration-500"
          />
          <div
            style={{ width: `${consensus.mixed_pct || 0}%` }}
            className="bg-amber-400 transition-all duration-500"
          />
          <div
            style={{ width: `${consensus.contradicting_pct || 0}%` }}
            className="bg-rose-500 transition-all duration-500"
          />
        </div>
      </div>

      {/* Executive Answer & Synthesis Sections */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-6 shadow-sm">
        <div>
          <h2 className="text-lg font-bold text-slate-900 mb-3">
            {isPrisma ? 'Abstract' : 'Executive Literature Synthesis'}
          </h2>
          <div className="text-sm text-slate-700 leading-relaxed whitespace-pre-line">
            {renderCitedText(executive_answer, handleCitationClick)}
          </div>
        </div>

        {sections.map((sec, idx) => (
          <div key={idx} className="border-t border-slate-200 pt-5">
            <h3 className="text-base font-bold text-indigo-800 mb-2">{sec.heading}</h3>
            <div className="text-sm text-slate-700 leading-relaxed whitespace-pre-line text-justify">
              {renderCitedText(sec.content, handleCitationClick)}
            </div>
            {sec.figure === 'prisma_flow' && data.review_protocol?.flow && (
              <PrismaFlow flow={data.review_protocol.flow} />
            )}
            {sec.figure === 'risk_of_bias' && (
              <RiskOfBiasTable
                papers={data.papers}
                corpusId={data.corpus_id}
                onCitationClick={handleCitationClick}
                onRobChange={onRobChange}
              />
            )}
          </div>
        ))}
      </div>

      <ProtocolCard protocol={data.protocol} />

      <ReviewProtocolPanel protocol={data.review_protocol} />

      {sourceCheck && (
        <SourceCheck
          claim={sourceCheck.sentence}
          paper={(data.papers || []).find((p) => p.citation_index === sourceCheck.idx)}
          onClose={() => setSourceCheck(null)}
          onShowReference={() => {
            const idx = sourceCheck.idx;
            setSourceCheck(null);
            showReference(idx);
          }}
        />
      )}

      {/* Mandatory Bibliography at the End of the Answer */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-4 mb-4">
          <div>
            <h2 className="text-base font-bold text-slate-900">
              References & Complete Bibliography ({bibliography.length})
            </h2>
            <p className="text-xs text-slate-500">
              All inline citations [1]..[{bibliography.length}] above resolve to the verified bibliography below.
            </p>
          </div>
          <div className="flex items-center gap-2">
            {['apa', 'ieee', 'bibtex'].map((st) => (
              <button
                key={st}
                onClick={() => setBibStyle(st)}
                className={`px-2.5 py-1 rounded text-xs font-semibold uppercase ${
                  bibStyle === st
                    ? 'bg-indigo-600 text-white'
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}
              >
                {st}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-2.5">
          {bibliography.map((bib) => (
            <div
              key={bib.index}
              id={`bib-entry-${bib.index}`}
              className={`p-3 rounded-lg border text-xs transition ${
                highlightedRef === bib.index
                  ? 'bg-indigo-50 border-indigo-400 text-slate-900'
                  : 'bg-slate-50/70 border-slate-200 text-slate-700'
              }`}
            >
              {bibStyle === 'bibtex' ? (
                <pre className="font-mono text-[11px] overflow-x-auto whitespace-pre-wrap text-slate-700">
                  {bib.bibtex}
                </pre>
              ) : (
                <div className="flex items-start justify-between gap-4">
                  <div className="leading-relaxed">
                    <span className="font-mono font-bold text-indigo-700 mr-1.5">[{bib.index}]</span>
                    <span>
                      {bibStyle === 'apa'
                        ? bib.apa_formatted.replace(/^\[\d+\]\s*/, '')
                        : bib.ieee_formatted.replace(/^\[\d+\]\s*/, '')}
                    </span>
                  </div>
                  {bib.url && (
                    <a
                      href={bib.url}
                      target="_blank"
                      rel="noreferrer"
                      className="shrink-0 px-2.5 py-1 rounded bg-white hover:bg-indigo-50 text-indigo-700 border border-slate-200 font-medium"
                    >
                      Open Source ↗
                    </a>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

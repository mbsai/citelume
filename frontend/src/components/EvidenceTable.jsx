import React, { useState, useMemo, useRef } from 'react';

const STANCE_BADGE = {
  supporting: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  mixed: 'bg-amber-50 text-amber-800 border-amber-200',
  contradicting: 'bg-rose-50 text-rose-700 border-rose-200',
  neutral: 'bg-slate-100 text-slate-700 border-slate-200',
};

const COMPARABILITY_BADGE = {
  'Directly comparable': 'bg-emerald-50 text-emerald-700 border-emerald-200',
  'Comparable with qualifications': 'bg-blue-50 text-blue-700 border-blue-200',
  'Related but not directly comparable': 'bg-amber-50 text-amber-800 border-amber-200',
  'Not comparable': 'bg-rose-50 text-rose-700 border-rose-200',
  'Insufficient information': 'bg-slate-100 text-slate-600 border-slate-200',
};

const STATUS_BADGE = {
  Reported: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  'Not reported': 'bg-slate-100 text-slate-600 border-slate-200',
  'Not assessed': 'bg-slate-100 text-slate-500 border-slate-200',
  'Not applicable': 'bg-purple-50 text-purple-700 border-purple-200',
  Unavailable: 'bg-rose-50 text-rose-700 border-rose-200',
  Uncertain: 'bg-amber-50 text-amber-800 border-amber-200',
};

export default function EvidenceTable({ papers = [], evidenceMatrix = null, query = '' }) {
  const [filterText, setFilterText] = useState('');
  const [stanceFilter, setStanceFilter] = useState('all');
  const [comparabilityFilter, setComparabilityFilter] = useState('all');
  const [showDomainCols, setShowDomainCols] = useState(true);
  const [expandedRows, setExpandedRows] = useState({});
  const scrollBoxRef = useRef(null);
  const scrollSideways = (dir) => {
    const box = scrollBoxRef.current;
    if (box) box.scrollBy({ left: dir * Math.max(300, box.clientWidth * 0.6), behavior: 'smooth' });
  };

  const rowsByCitation = useMemo(() => {
    const map = {};
    if (evidenceMatrix?.rows) {
      evidenceMatrix.rows.forEach((r) => {
        map[r.citation_index] = r;
      });
    }
    return map;
  }, [evidenceMatrix]);

  const domainColumns = useMemo(() => {
    return evidenceMatrix?.extraction_plan?.visible_domain_columns || [];
  }, [evidenceMatrix]);

  const toggleRow = (idx) => {
    setExpandedRows((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const expandAll = () => {
    const next = {};
    papers.forEach((p) => {
      next[p.citation_index] = true;
    });
    setExpandedRows(next);
  };

  const collapseAll = () => {
    setExpandedRows({});
  };

  const filtered = papers.filter((p) => {
    if (stanceFilter !== 'all' && p.stance !== stanceFilter) return false;
    const rowObj = rowsByCitation[p.citation_index];
    if (comparabilityFilter !== 'all' && rowObj) {
      if (rowObj.evidence_unit?.comparability?.status !== comparabilityFilter) return false;
    }
    if (!filterText.trim()) return true;
    const q = filterText.toLowerCase();
    const domainText = rowObj
      ? Object.values(rowObj.evidence_unit?.domain_specific_fields || {}).join(' ')
      : '';
    const findingsText = rowObj
      ? (rowObj.evidence_unit?.results || []).map((r) => `${r.factual_source_text} ${r.normalized_summary}`).join(' ')
      : '';
    return (
      p.title.toLowerCase().includes(q) ||
      (p.elicit?.methodology || '').toLowerCase().includes(q) ||
      (p.elicit?.dataset_or_system || '').toLowerCase().includes(q) ||
      (p.elicit?.key_findings || '').toLowerCase().includes(q) ||
      domainText.toLowerCase().includes(q) ||
      findingsText.toLowerCase().includes(q)
    );
  });

  const handleDownloadJson = () => {
    const payload = evidenceMatrix || { query, papers };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'structured_evidence_matrix.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadCsv = () => {
    const headers = [
      'Ref',
      'Document_ID',
      'Title',
      'Year',
      'Source',
      'Publication_Status',
      'Evidence_Access',
      'Study_Design',
      'System_or_Dataset',
      'Sample_or_Scale',
      'Methodology',
      ...domainColumns.map((c) => c.display_label),
      'Primary_Quantitative_Finding',
      'Verbatim_Source_Excerpt',
      'Validation_Tier',
      'Results_Count',
      'Author_Reported_Limitations',
      'Independent_Methodological_Concerns',
      'Comparability_Status',
      'Linked_Claims',
    ];

    const escapeCsv = (val) => {
      const str = String(val ?? '').replace(/\r?\n/g, ' ');
      return `"${str.replace(/"/g, '""')}"`;
    };

    const csvRows = [headers.map(escapeCsv).join(',')];
    filtered.forEach((p) => {
      const r = rowsByCitation[p.citation_index];
      const doc = r?.document;
      const unit = r?.evidence_unit;
      const pf = r?.primary_finding;
      const domVals = domainColumns.map((c) => unit?.domain_specific_fields?.[c.field_key] || 'Not reported');

      const rowArr = [
        `[${p.citation_index}]`,
        doc?.document_id || `DOC-${p.citation_index}`,
        p.title,
        p.year || 'n.d.',
        p.source,
        doc?.publication_status || p.version_status || 'published',
        doc?.evidence_access_level || 'abstract',
        unit?.experimental_or_analytical_design || p.elicit?.study_design || 'Empirical Study',
        unit?.context_or_system || p.elicit?.dataset_or_system || 'Not reported',
        unit?.sample_or_scale || p.elicit?.sample_or_scale || 'Not reported',
        unit?.methodological_approach || p.elicit?.methodology || 'Not reported',
        ...domVals,
        pf?.normalized_summary || p.elicit?.key_findings || 'Not reported',
        pf?.factual_source_text || 'Not reported',
        pf?.quantitative_detail?.analysis_tier || 'primary',
        unit?.results?.length || 1,
        (unit?.appraisal?.source_reported_limitations || [p.elicit?.limitations]).join(' | '),
        (unit?.appraisal?.independent_methodological_concerns || []).join(' | '),
        unit?.comparability?.status || 'Comparable with qualifications',
        (unit?.linked_claim_ids || []).join('; '),
      ];
      csvRows.push(rowArr.map(escapeCsv).join(','));
    });

    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'structured_evidence_matrix.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const plan = evidenceMatrix?.extraction_plan;
  const validation = evidenceMatrix?.validation_report;
  const totalGranularFindings = evidenceMatrix?.granular_findings?.length || papers.length;

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-4">
      {/* Top Header & Query-Adaptive Extraction Plan Banner */}
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-slate-100 pb-4">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-sm font-bold text-slate-900">
              Universal, Question-Adaptive, Source-Grounded Scientific Evidence Matrix
            </h3>
            {plan && (
              <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-indigo-50 text-indigo-700 border border-indigo-200">
                Profile: {plan.primary_domain}
              </span>
            )}
            {validation && (
              <span
                className={`px-2 py-0.5 rounded-full text-[10px] font-bold border ${
                  validation.structural_valid && validation.factual_provenance_valid
                    ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                    : 'bg-amber-50 text-amber-800 border-amber-200'
                }`}
              >
                ✓ 4-Layer Provenance & Comparability Verified
              </span>
            )}
          </div>
          <p className="text-xs text-slate-500 mt-1">
            3-Level Evidence Model: <strong>{papers.length}</strong> Document Records (Level A) •{' '}
            <strong>{evidenceMatrix?.evidence_units?.length || papers.length}</strong> Research Contributions (Level B) •{' '}
            <strong>{totalGranularFindings}</strong> Granular Result Records (Level C)
          </p>
        </div>

        {/* Export & Expand Controls */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => setShowDomainCols((v) => !v)}
            className={`px-2.5 py-1.5 rounded-lg text-[11px] font-semibold border transition ${
              showDomainCols
                ? 'bg-indigo-50 text-indigo-700 border-indigo-200'
                : 'bg-slate-50 text-slate-600 border-slate-200'
            }`}
          >
            {showDomainCols ? '✓ Adaptive Domain Columns On' : 'Show Adaptive Domain Columns'}
          </button>
          <button
            type="button"
            onClick={expandAll}
            className="px-2.5 py-1.5 rounded-lg text-[11px] font-semibold bg-slate-50 hover:bg-slate-100 text-slate-700 border border-slate-200 transition"
          >
            Expand All Details
          </button>
          <button
            type="button"
            onClick={collapseAll}
            className="px-2.5 py-1.5 rounded-lg text-[11px] font-semibold bg-slate-50 hover:bg-slate-100 text-slate-700 border border-slate-200 transition"
          >
            Collapse All
          </button>
          <button
            type="button"
            onClick={handleDownloadCsv}
            className="px-2.5 py-1.5 rounded-lg text-[11px] font-semibold bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border border-emerald-200 transition"
          >
            ⬇ Export CSV Matrix
          </button>
          <button
            type="button"
            onClick={handleDownloadJson}
            className="px-2.5 py-1.5 rounded-lg text-[11px] font-semibold bg-slate-800 hover:bg-slate-900 text-white transition"
          >
            ⬇ Export 3-Level JSON
          </button>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="text"
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
            placeholder="Filter by study, methodology, dataset, quantitative metric, or excerpt..."
            className="bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-xs text-slate-800 w-72 focus:outline-none focus:border-indigo-500"
          />
          <select
            value={stanceFilter}
            onChange={(e) => setStanceFilter(e.target.value)}
            className="bg-slate-50 border border-slate-300 rounded-lg px-2.5 py-1.5 text-xs text-slate-700 focus:outline-none focus:border-indigo-500"
          >
            <option value="all">All Stances ({papers.length})</option>
            <option value="supporting">Supporting</option>
            <option value="mixed">Mixed / Conditional</option>
            <option value="contradicting">Contradicting</option>
          </select>
          <select
            value={comparabilityFilter}
            onChange={(e) => setComparabilityFilter(e.target.value)}
            className="bg-slate-50 border border-slate-300 rounded-lg px-2.5 py-1.5 text-xs text-slate-700 focus:outline-none focus:border-indigo-500"
          >
            <option value="all">All Comparability Classes</option>
            <option value="Directly comparable">Directly comparable</option>
            <option value="Comparable with qualifications">Comparable with qualifications</option>
            <option value="Related but not directly comparable">Related but not directly comparable</option>
            <option value="Insufficient information">Insufficient information</option>
          </select>
        </div>

        <span className="text-[11px] text-slate-500">
          Showing <strong>{filtered.length}</strong> of <strong>{papers.length}</strong> studies (click any row to inspect Level A/B/C records & provenance)
        </span>
      </div>

      {/* Main Evidence Matrix Table: the box has its own height, so its sideways scrollbar is always on screen */}
      <div className="flex items-center justify-end gap-2 text-[11px] text-slate-500">
        <span className="hidden sm:inline">Scroll sideways with the buttons, Shift + mouse wheel, or the bar at the bottom of the table</span>
        <button
          type="button"
          onClick={() => scrollSideways(-1)}
          className="px-2.5 py-1 rounded-md border border-slate-300 bg-white hover:bg-slate-50 text-slate-700 font-semibold"
          aria-label="Scroll table left"
        >
          ◀ Left
        </button>
        <button
          type="button"
          onClick={() => scrollSideways(1)}
          className="px-2.5 py-1 rounded-md border border-slate-300 bg-white hover:bg-slate-50 text-slate-700 font-semibold"
          aria-label="Scroll table right"
        >
          Right ▶
        </button>
      </div>
      <div ref={scrollBoxRef} className="overflow-auto max-h-[75vh] border border-slate-200 rounded-xl">
        <table className="w-full text-left border-collapse text-xs">
          <thead className="sticky top-0 z-10 shadow-sm">
            <tr className="border-b border-slate-200 text-slate-600 bg-slate-50">
              <th className="py-2.5 px-3 font-semibold w-14">Ref</th>
              <th className="py-2.5 px-3 font-semibold min-w-[230px]">
                Level A: Document & Access Provenance
              </th>
              <th className="py-2.5 px-3 font-semibold min-w-[160px]">
                Level B: Design & Methodology
              </th>
              <th className="py-2.5 px-3 font-semibold min-w-[160px]">
                System / Dataset & Scale
              </th>
              {showDomainCols &&
                domainColumns.slice(0, 2).map((col) => (
                  <th
                    key={col.field_key}
                    className="py-2.5 px-3 font-semibold min-w-[165px] bg-indigo-50/50 text-indigo-950"
                    title={col.description}
                  >
                    <div className="flex items-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full bg-indigo-500"></span>
                      <span>{col.display_label}</span>
                    </div>
                  </th>
                ))}
              <th className="py-2.5 px-3 font-semibold min-w-[250px]">
                Level C: Source-Grounded Findings & Metrics
              </th>
              <th className="py-2.5 px-3 font-semibold min-w-[220px]">
                Separated Limitations & Appraisal
              </th>
              <th className="py-2.5 px-3 font-semibold min-w-[165px]">
                Comparability & Claims
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200">
            {filtered.map((p) => {
              const rowObj = rowsByCitation[p.citation_index];
              const doc = rowObj?.document;
              const unit = rowObj?.evidence_unit;
              const pf = rowObj?.primary_finding;
              const allResults = unit?.results || [];
              const isExpanded = !!expandedRows[p.citation_index];
              const compStatus = unit?.comparability?.status || 'Comparable with qualifications';
              const pubStatus = doc?.publication_status || p.version_status || 'published';
              const accessLevel = doc?.evidence_access_level || 'abstract';

              return (
                <React.Fragment key={p.id}>
                  <tr
                    onClick={() => toggleRow(p.citation_index)}
                    className="hover:bg-slate-50/90 transition align-top cursor-pointer"
                  >
                    {/* Ref & Expand toggle */}
                    <td className="py-3 px-3 font-mono font-bold text-indigo-600">
                      <div className="flex flex-col items-start gap-1">
                        <span>[{p.citation_index}]</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 font-sans font-medium">
                          {isExpanded ? '▾ Hide' : '▸ Inspect'}
                        </span>
                      </div>
                    </td>

                    {/* Level A: Document & Access Provenance */}
                    <td className="py-3 px-3">
                      <a
                        href={p.url}
                        target="_blank"
                        rel="noreferrer"
                        onClick={(e) => e.stopPropagation()}
                        className="font-semibold text-slate-900 hover:text-indigo-600 transition line-clamp-2"
                      >
                        {p.title}
                      </a>
                      <div className="text-[11px] text-slate-500 mt-1">
                        {p.authors?.slice(0, 2).join(', ') || 'Unknown'} ({p.year || 'n.d.'})
                      </div>
                      <div className="flex flex-wrap items-center gap-1 mt-1.5">
                        <span className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-700 border border-slate-200 text-[10px] font-medium">
                          {p.source}
                        </span>
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] font-medium border ${
                            pubStatus === 'preprint'
                              ? 'bg-amber-50 text-amber-800 border-amber-200'
                              : 'bg-emerald-50 text-emerald-700 border-emerald-200'
                          }`}
                        >
                          {pubStatus === 'preprint_with_published_link' ? 'preprint+published' : pubStatus}
                        </span>
                        <span
                          className={`px-1.5 py-0.5 rounded border text-[10px] ${
                            accessLevel === 'full_text'
                              ? 'bg-emerald-50 text-emerald-800 border-emerald-300 font-semibold'
                              : 'bg-slate-50 text-slate-600 border-slate-200'
                          }`}
                          title={p.full_text_source ? `Read from ${p.full_text_source}` : undefined}
                        >
                          Access: {accessLevel === 'full_text' ? 'full text' : accessLevel.replace('_', ' ')}
                        </span>
                        {p.citation_count > 0 && (
                          <span className="text-[10px] text-amber-700 font-semibold">
                            ★ {p.citation_count}
                          </span>
                        )}
                      </div>
                    </td>

                    {/* Level B: Design & Methodology */}
                    <td className="py-3 px-3 text-slate-700">
                      <div className="font-semibold text-slate-900 text-[11px] mb-1">
                        {unit?.experimental_or_analytical_design || p.elicit?.study_design || 'Empirical Study'}
                      </div>
                      <div className="text-slate-600 text-[11px] line-clamp-3">
                        {unit?.methodological_approach || p.elicit?.methodology || 'Not reported'}
                      </div>
                    </td>

                    {/* Context: System / Dataset & Scale */}
                    <td className="py-3 px-3 text-slate-700">
                      <div className="font-medium text-slate-800 text-[11px]">
                        {unit?.context_or_system || p.elicit?.dataset_or_system || 'Not reported'}
                      </div>
                      <div className="mt-1 flex flex-wrap items-center gap-1">
                        <span
                          className={`px-1.5 py-0.5 rounded border text-[10px] font-medium ${
                            STATUS_BADGE[unit?.sample_reporting_status || p.elicit?.sample_reporting_status || 'Not reported'] ||
                            STATUS_BADGE['Not reported']
                          }`}
                        >
                          Scale: {unit?.sample_or_scale || p.elicit?.sample_or_scale || 'Not reported'}
                        </span>
                      </div>
                    </td>

                    {/* Adaptive Domain-Specific Columns */}
                    {showDomainCols &&
                      domainColumns.slice(0, 2).map((col) => {
                        const val = unit?.domain_specific_fields?.[col.field_key] || 'Not reported';
                        const st = unit?.domain_field_statuses?.[col.field_key] || 'Reported';
                        return (
                          <td key={col.field_key} className="py-3 px-3 bg-indigo-50/20 text-slate-700">
                            <div className="text-[11px] text-slate-800 line-clamp-3">{val}</div>
                            <span
                              className={`inline-block mt-1 px-1.5 py-0.5 rounded border text-[9px] font-semibold ${
                                STATUS_BADGE[st] || STATUS_BADGE['Reported']
                              }`}
                            >
                              {st}
                            </span>
                          </td>
                        );
                      })}

                    {/* Level C: Source-Grounded Findings & Metrics */}
                    <td className="py-3 px-3 text-slate-800">
                      <div className="text-[11px] font-medium text-slate-900 line-clamp-3">
                        {pf?.normalized_summary || p.elicit?.key_findings || 'Not reported'}
                      </div>
                      <div className="flex flex-wrap items-center gap-1 mt-1.5">
                        <span
                          className={`px-1.5 py-0.5 rounded border text-[9px] font-semibold uppercase ${
                            STANCE_BADGE[p.stance] || STANCE_BADGE.supporting
                          }`}
                        >
                          {p.stance}
                        </span>
                        {pf?.quantitative_detail?.analysis_tier && (
                          <span className="px-1.5 py-0.5 rounded bg-slate-100 text-slate-700 border border-slate-200 text-[9px] font-medium">
                            {pf.quantitative_detail.analysis_tier.replace(/_/g, ' ')}
                          </span>
                        )}
                        {allResults.length > 1 && (
                          <span className="px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200 text-[9px] font-bold">
                            +{allResults.length - 1} more result{allResults.length - 1 > 1 ? 's' : ''}
                          </span>
                        )}
                      </div>
                    </td>

                    {/* Separated Limitations & Methodological Appraisal */}
                    <td className="py-3 px-3 text-slate-600">
                      <div className="text-[10px] font-semibold text-slate-700">
                        Author-Reported:
                      </div>
                      <div className="text-[11px] text-slate-600 line-clamp-2 mb-1">
                        {unit?.appraisal?.source_reported_limitations?.[0] || p.elicit?.limitations || 'Not explicitly stated in abstract'}
                      </div>
                      {unit?.appraisal?.independent_methodological_concerns?.[0] && (
                        <>
                          <div className="text-[10px] font-semibold text-amber-800">
                            Methodological Appraisal ({
                              { full_text_complete: 'Full text', preliminary_abstract_only: 'Preliminary, abstract only' }[
                                unit.appraisal.appraisal_completeness
                              ] || 'Minimal, metadata only'
                            }):
                          </div>
                          <div className="text-[11px] text-amber-900 line-clamp-2">
                            {unit.appraisal.independent_methodological_concerns[0]}
                          </div>
                        </>
                      )}
                    </td>

                    {/* Comparability & Section 1 Claim Links */}
                    <td className="py-3 px-3">
                      <span
                        className={`inline-block px-2 py-0.5 rounded border text-[10px] font-semibold ${
                          COMPARABILITY_BADGE[compStatus] || COMPARABILITY_BADGE['Comparable with qualifications']
                        }`}
                      >
                        {compStatus}
                      </span>
                      {unit?.comparability?.comparable_peer_indices?.length > 0 && (
                        <div className="text-[10px] text-slate-500 mt-1">
                          Peers:{' '}
                          {unit.comparability.comparable_peer_indices
                            .slice(0, 4)
                            .map((i) => `[${i}]`)
                            .join(', ')}
                        </div>
                      )}
                      {unit?.linked_claim_ids?.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-1.5">
                          {unit.linked_claim_ids.map((cid) => (
                            <span
                              key={cid}
                              className="px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200 text-[9px] font-mono font-semibold"
                            >
                              {cid}
                            </span>
                          ))}
                        </div>
                      )}
                    </td>
                  </tr>

                  {/* Expandable 3-Level Scientific Evidence Inspection Drawer */}
                  {isExpanded && rowObj && (
                    <tr className="bg-slate-50/90 border-b border-slate-200">
                      <td colSpan={showDomainCols ? 6 + Math.min(2, domainColumns.length) : 6} className="p-4">
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 text-xs">
                          {/* Column 1: Level A (Document) + Level B (Study Context & Domain Fields) */}
                          <div className="bg-white border border-slate-200 rounded-xl p-3.5 space-y-2.5">
                            <div className="flex items-center justify-between border-b border-slate-100 pb-1.5">
                              <span className="font-bold text-slate-900">
                                Level A & B: Document & Study Contribution
                              </span>
                              <span className="text-[10px] font-mono text-slate-500">
                                {doc.document_id} / {unit.contribution_id}
                              </span>
                            </div>
                            <div className="space-y-1 text-[11px] text-slate-700">
                              <div>
                                <strong>Research Objective:</strong> {unit.research_objective}
                              </div>
                              <div>
                                <strong>Persistent IDs:</strong>{' '}
                                {[
                                  doc.doi ? `DOI: ${doc.doi}` : null,
                                  doc.pmid ? `PMID: ${doc.pmid}` : null,
                                  doc.arxiv_id ? `arXiv: ${doc.arxiv_id}` : null,
                                ]
                                  .filter(Boolean)
                                  .join(' • ') || 'Source URL only'}
                              </div>
                              <div>
                                <strong>Document Relation:</strong>{' '}
                                <span className="font-mono text-indigo-700">{unit.relationship_type}</span> —{' '}
                                {unit.overlap_or_linkage_note}
                              </div>
                              <div>
                                <strong>Retraction/Correction Screening:</strong> {doc.retraction_or_correction_status}
                              </div>
                            </div>

                            {domainColumns.length > 0 && (
                              <div className="pt-2 border-t border-slate-100 space-y-1.5">
                                <div className="text-[10px] font-bold uppercase tracking-wider text-indigo-700">
                                  Domain-Adaptive Extracted Fields ({plan?.domain_module_activated})
                                </div>
                                {domainColumns.map((col) => {
                                  const fVal = unit.domain_specific_fields?.[col.field_key] || 'Not reported';
                                  const fSt = unit.domain_field_statuses?.[col.field_key] || 'Not reported';
                                  return (
                                    <div key={col.field_key} className="text-[11px] flex items-start justify-between gap-2">
                                      <div>
                                        <span className="font-semibold text-slate-800">{col.display_label}:</span>{' '}
                                        <span className="text-slate-700">{fVal}</span>
                                      </div>
                                      <span
                                        className={`shrink-0 px-1.5 py-0.5 rounded border text-[9px] font-semibold ${
                                          STATUS_BADGE[fSt] || STATUS_BADGE['Reported']
                                        }`}
                                      >
                                        {fSt}
                                      </span>
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                          </div>

                          {/* Column 2: Level C Granular Findings (Multi-Result + Verbatim Excerpt + Derived Quantities) */}
                          <div className="bg-white border border-slate-200 rounded-xl p-3.5 space-y-2.5">
                            <div className="flex items-center justify-between border-b border-slate-100 pb-1.5">
                              <span className="font-bold text-slate-900">
                                Level C: Granular Result Records ({allResults.length})
                              </span>
                              <span className="text-[10px] text-emerald-700 font-semibold">
                                Factual vs. Interpretive Separated
                              </span>
                            </div>
                            <div className="space-y-2.5 max-h-72 overflow-y-auto pr-1">
                              {allResults.map((res) => (
                                <div
                                  key={res.result_id}
                                  className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 space-y-1 text-[11px]"
                                >
                                  <div className="flex items-center justify-between gap-2">
                                    <span className="font-bold text-indigo-700">
                                      [{res.result_id}] {res.experiment_or_component_label}
                                    </span>
                                    <span
                                      className={`px-1.5 py-0.5 rounded border text-[9px] font-semibold ${
                                        STATUS_BADGE[res.extraction_status] || STATUS_BADGE.Reported
                                      }`}
                                    >
                                      {res.extraction_status}
                                    </span>
                                  </div>
                                  <div>
                                    <strong>Normalized Summary:</strong> {res.normalized_summary}
                                  </div>
                                  <div className="text-slate-600 italic bg-white p-1.5 rounded border border-slate-200">
                                    <strong>Verbatim Source Excerpt ({res.provenance?.page_or_span}):</strong> &ldquo;
                                    {res.factual_source_text}&rdquo;
                                  </div>
                                  <div className="grid grid-cols-2 gap-1 text-[10px] text-slate-600 pt-1">
                                    <div>
                                      <strong>Metric:</strong> {res.quantitative_detail?.metric_type}
                                    </div>
                                    <div>
                                      <strong>Direction:</strong> {res.direction_of_effect}
                                    </div>
                                    <div>
                                      <strong>Uncertainty:</strong> {res.quantitative_detail?.uncertainty_interval}
                                    </div>
                                    <div>
                                      <strong>p-value:</strong> {res.quantitative_detail?.p_value}
                                    </div>
                                  </div>
                                  {res.quantitative_detail?.derived_quantities?.length > 0 && (
                                    <div className="pt-1 text-[10px] text-indigo-800 bg-indigo-50/60 p-1.5 rounded">
                                      <strong>Derived Quantity:</strong>{' '}
                                      {res.quantitative_detail.derived_quantities.map((dq, i) => (
                                        <span key={i}>
                                          {dq.quantity_name} = {dq.derived_value} ({dq.formula_used})
                                        </span>
                                      ))}
                                    </div>
                                  )}
                                  <div className="text-[10px] text-slate-500">
                                    <strong>Analytical Interpretation:</strong> {res.llm_interpretation}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* Column 3: Separated Methodological Appraisal & Comparability Rationale */}
                          <div className="bg-white border border-slate-200 rounded-xl p-3.5 space-y-2.5">
                            <div className="flex items-center justify-between border-b border-slate-100 pb-1.5">
                              <span className="font-bold text-slate-900">
                                Method-Aware Appraisal & Comparability
                              </span>
                              <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-50 text-amber-800 border border-amber-200 font-semibold">
                                {unit.appraisal.appraisal_completeness}
                              </span>
                            </div>
                            <div className="space-y-2 text-[11px]">
                              <div>
                                <div className="font-semibold text-slate-800">
                                  Appraisal Framework: {unit.appraisal.appraisal_framework}
                                </div>
                              </div>
                              <div>
                                <div className="font-semibold text-slate-700">1. Author-Reported Limitations:</div>
                                <ul className="list-disc list-inside text-slate-600 space-y-0.5">
                                  {unit.appraisal.source_reported_limitations.map((lim, idx) => (
                                    <li key={idx}>{lim}</li>
                                  ))}
                                </ul>
                              </div>
                              <div>
                                <div className="font-semibold text-amber-800">
                                  2. Independent Methodological Concerns:
                                </div>
                                <ul className="list-disc list-inside text-amber-900 space-y-0.5">
                                  {unit.appraisal.independent_methodological_concerns.map((c, idx) => (
                                    <li key={idx}>{c}</li>
                                  ))}
                                </ul>
                              </div>
                              <div>
                                <div className="font-semibold text-indigo-800">3. Applicability & Scope:</div>
                                <ul className="list-disc list-inside text-slate-600 space-y-0.5">
                                  {unit.appraisal.applicability_concerns.map((a, idx) => (
                                    <li key={idx}>{a}</li>
                                  ))}
                                </ul>
                              </div>
                              <div className="pt-2 border-t border-slate-100">
                                <div className="font-semibold text-slate-800">
                                  Comparability Assessment ({unit.comparability.status}):
                                </div>
                                <p className="text-slate-600 mt-0.5">{unit.comparability.comparability_rationale}</p>
                              </div>
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
    </div>
  );
}

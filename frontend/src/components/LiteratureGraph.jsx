import React, { useState } from 'react';
import { expandPaperNode } from '../utils/api';

const CLUSTER_COLORS = ['#4f46e5', '#059669', '#0284c7', '#d97706', '#db2777', '#7c3aed'];

const ROLE_COLORS = {
  foundational_contribution: '#1d4ed8',
  citation_hub_in_context: '#0284c7',
  bridge_publication: '#7c3aed',
  recent_methodological_contribution: '#059669',
  evidence_central_publication: '#d97706',
  standard_included_study: '#475569',
};

const ROLE_LABELS = {
  foundational_contribution: 'Foundational Contribution',
  citation_hub_in_context: 'Citation Hub in Context',
  bridge_publication: 'Cross-Stream Bridge',
  recent_methodological_contribution: 'Recent Methodological Advance',
  evidence_central_publication: 'Evidence-Central Study',
  standard_included_study: 'Included Primary Study',
};

function triggerDownload(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function xmlEscape(val) {
  return String(val ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function csvEscape(val) {
  const s = String(val ?? '');
  if (s.includes(',') || s.includes('"') || s.includes('\n')) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

export default function LiteratureGraph({ graph, papers = [], config, onGraphExpand }) {
  const [layoutMode, setLayoutMode] = useState('timeline');
  const [colorMode, setColorMode] = useState('cluster');
  const [layerFilter, setLayerFilter] = useState('all');
  const [showContextNodes, setShowContextNodes] = useState(true);
  const [showClaimAndGapNodes, setShowClaimAndGapNodes] = useState(true);
  const [selectedNode, setSelectedNode] = useState(null);
  const [selectedEdge, setSelectedEdge] = useState(null);
  const [expanding, setExpanding] = useState(false);

  if (!graph || !graph.nodes || graph.nodes.length === 0) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-8 text-center text-slate-500">
        No literature network nodes available. Run a query to generate the Multilayer Scientific Literature Map.
      </div>
    );
  }

  const nodeMap = {};
  graph.nodes.forEach((n) => {
    nodeMap[n.id] = n;
  });

  // Filter visible nodes based on toggles
  const visibleNodes = graph.nodes.filter((n) => {
    if (!showContextNodes && (n.is_citation_context_only || n.inclusion_role === 'citation_context_publication')) {
      return false;
    }
    if (!showClaimAndGapNodes && (n.node_type === 'scientific_claim' || n.node_type === 'knowledge_gap')) {
      return false;
    }
    return true;
  });

  const visibleNodeIds = new Set(visibleNodes.map((n) => n.id));

  // Filter visible edges based on active layer filter & visible endpoints
  const visibleEdges = (graph.edges || []).filter((e) => {
    if (!visibleNodeIds.has(e.source) || !visibleNodeIds.has(e.target)) return false;
    if (layerFilter === 'all') return true;
    if (layerFilter === 'bibliographic') {
      return (
        e.network_layer === 'bibliographic' ||
        ['citation', 'bibliographic_coupling', 'co_citation'].includes(e.edge_type)
      );
    }
    if (layerFilter === 'semantic_conceptual') {
      return e.network_layer === 'semantic_conceptual' || e.edge_type === 'semantic';
    }
    if (layerFilter === 'scientific_evidence') {
      return (
        e.network_layer === 'scientific_evidence' ||
        e.network_layer === 'artifact_link' ||
        ['evidence_relationship', 'model_link'].includes(e.edge_type)
      );
    }
    if (layerFilter === 'research_evolution') {
      return e.network_layer === 'research_evolution' || e.edge_type === 'research_evolution';
    }
    return true;
  });

  const getNodeColor = (node) => {
    if (node.node_type === 'scientific_claim') return '#0f766e';
    if (node.node_type === 'knowledge_gap') return '#b45309';
    if (node.node_type !== 'paper') return '#d97706';

    if (colorMode === 'importance') {
      const topRole = (node.importance_roles && node.importance_roles[0]) || 'standard_included_study';
      return ROLE_COLORS[topRole] || '#475569';
    }
    if (colorMode === 'stance') {
      if (node.stance === 'supporting') return '#059669';
      if (node.stance === 'contradicting') return '#e11d48';
      return '#d97706';
    }
    if (colorMode === 'source') {
      if (node.source === 'PubMed') return '#2563eb';
      if (node.source === 'arXiv') return '#dc2626';
      if (node.source === 'bioRxiv') return '#059669';
      return '#7c3aed';
    }
    return CLUSTER_COLORS[Math.abs(node.cluster_id || 0) % CLUSTER_COLORS.length];
  };

  const getEdgeColor = (edge, isHighlighted) => {
    if (isHighlighted) return '#312e81';
    if (edge.edge_type === 'citation') return '#0284c7';
    if (edge.edge_type === 'bibliographic_coupling') return '#0369a1';
    if (edge.edge_type === 'co_citation') return '#0891b2';
    if (edge.edge_type === 'research_evolution') return '#7c3aed';
    if (edge.edge_type === 'evidence_relationship') {
      const sub = edge.relationship_subtype || edge.edge_subtype || '';
      if (sub === 'challenges_claim' || sub === 'contradictory_evidence_on_claim') return '#e11d48';
      if (sub === 'supports_claim') return '#059669';
      if (sub === 'motivates_knowledge_gap') return '#d97706';
      if (sub === 'uses_shared_dataset') return '#6366f1';
      return '#0f766e';
    }
    if (edge.edge_type === 'model_link') return '#d97706';
    return '#94a3b8';
  };

  const handleExpandNode = async () => {
    if (!selectedNode || selectedNode.node_type !== 'paper') return;
    const paperObj = papers.find((p) => p.id === selectedNode.id) || {};
    setExpanding(true);
    try {
      const res = await expandPaperNode({
        paper_id: selectedNode.id,
        title: selectedNode.full_title,
        doi: paperObj.doi,
        arxiv_id: paperObj.arxiv_id,
        pmid: paperObj.pmid,
        s2_id: paperObj.s2_id,
        config,
      });
      if (onGraphExpand && res.neighbors) {
        onGraphExpand(selectedNode, res.neighbors);
      }
    } catch (err) {
      alert(`Could not expand node: ${err.message}`);
    } finally {
      setExpanding(false);
    }
  };

  const handleExportGraphML = () => {
    const lines = [
      '<?xml version="1.0" encoding="UTF-8"?>',
      '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
      '  <key id="d0" for="node" attr.name="label" attr.type="string"/>',
      '  <key id="d1" for="node" attr.name="full_title" attr.type="string"/>',
      '  <key id="d2" for="node" attr.name="node_type" attr.type="string"/>',
      '  <key id="d3" for="node" attr.name="inclusion_role" attr.type="string"/>',
      '  <key id="d4" for="node" attr.name="year" attr.type="int"/>',
      '  <key id="d5" for="node" attr.name="citations" attr.type="int"/>',
      '  <key id="d6" for="node" attr.name="cluster_label" attr.type="string"/>',
      '  <key id="d7" for="node" attr.name="importance_roles" attr.type="string"/>',
      '  <key id="e0" for="edge" attr.name="edge_type" attr.type="string"/>',
      '  <key id="e1" for="edge" attr.name="edge_subtype" attr.type="string"/>',
      '  <key id="e2" for="edge" attr.name="network_layer" attr.type="string"/>',
      '  <key id="e3" for="edge" attr.name="weight" attr.type="double"/>',
      '  <key id="e4" for="edge" attr.name="verification_status" attr.type="string"/>',
      '  <key id="e5" for="edge" attr.name="explanation" attr.type="string"/>',
      '  <graph id="LiteratureNetwork" edgedefault="directed">',
    ];
    graph.nodes.forEach((n) => {
      lines.push(`    <node id="${xmlEscape(n.id)}">`);
      lines.push(`      <data key="d0">${xmlEscape(n.label)}</data>`);
      lines.push(`      <data key="d1">${xmlEscape(n.full_title)}</data>`);
      lines.push(`      <data key="d2">${xmlEscape(n.node_type)}</data>`);
      lines.push(`      <data key="d3">${xmlEscape(n.inclusion_role || 'included_synthesis_evidence')}</data>`);
      lines.push(`      <data key="d4">${n.year || 0}</data>`);
      lines.push(`      <data key="d5">${n.citations || 0}</data>`);
      lines.push(`      <data key="d6">${xmlEscape(n.cluster_label)}</data>`);
      lines.push(`      <data key="d7">${xmlEscape((n.importance_roles || []).join(';'))}</data>`);
      lines.push('    </node>');
    });
    graph.edges.forEach((e) => {
      lines.push(
        `    <edge id="${xmlEscape(e.id)}" source="${xmlEscape(e.source)}" target="${xmlEscape(e.target)}">`
      );
      lines.push(`      <data key="e0">${xmlEscape(e.edge_type)}</data>`);
      lines.push(`      <data key="e1">${xmlEscape(e.relationship_subtype || e.edge_subtype || e.edge_type)}</data>`);
      lines.push(`      <data key="e2">${xmlEscape(e.network_layer || 'bibliographic')}</data>`);
      lines.push(`      <data key="e3">${e.weight ?? 0.5}</data>`);
      lines.push(`      <data key="e4">${xmlEscape(e.verification_status || e.provenance)}</data>`);
      lines.push(`      <data key="e5">${xmlEscape(e.relationship_explanation || e.label)}</data>`);
      lines.push('    </edge>');
    });
    lines.push('  </graph>', '</graphml>');
    triggerDownload(lines.join('\n'), 'literature_network.graphml', 'application/xml');
  };

  const handleExportNodesCSV = () => {
    const headers = [
      'id',
      'label',
      'full_title',
      'node_type',
      'inclusion_role',
      'citation_index',
      'year',
      'citations',
      'source',
      'cluster_label',
      'bibliographic_cluster_label',
      'stance',
      'importance_roles',
      'importance_rationale',
      'metadata_completeness',
      'pagerank',
    ];
    const rows = [headers.join(',')];
    graph.nodes.forEach((n) => {
      rows.push(
        [
          n.id,
          n.label,
          n.full_title,
          n.node_type,
          n.inclusion_role || 'included_synthesis_evidence',
          n.citation_index ?? '',
          n.year ?? '',
          n.citations ?? 0,
          n.source,
          n.cluster_label,
          n.bibliographic_cluster_label || '',
          n.stance,
          (n.importance_roles || []).join(';'),
          n.importance_rationale || '',
          n.metadata_completeness || '',
          n.pagerank ?? 0,
        ]
          .map(csvEscape)
          .join(',')
      );
    });
    triggerDownload(rows.join('\n'), 'literature_network_nodes.csv', 'text/csv');
  };

  const handleExportEdgesCSV = () => {
    const headers = [
      'id',
      'source',
      'target',
      'edge_type',
      'edge_subtype',
      'network_layer',
      'directed',
      'raw_weight',
      'normalized_weight',
      'weighting_method',
      'provenance',
      'verification_status',
      'label',
      'relationship_explanation',
    ];
    const rows = [headers.join(',')];
    graph.edges.forEach((e) => {
      rows.push(
        [
          e.id,
          e.source,
          e.target,
          e.edge_type,
          e.relationship_subtype || e.edge_subtype || e.edge_type,
          e.network_layer || 'bibliographic',
          e.directed ?? false,
          e.raw_weight ?? e.weight,
          e.normalized_weight ?? e.weight,
          e.weighting_method || '',
          e.provenance,
          e.verification_status || '',
          e.label,
          e.relationship_explanation || '',
        ]
          .map(csvEscape)
          .join(',')
      );
    });
    triggerDownload(rows.join('\n'), 'literature_network_edges.csv', 'text/csv');
  };

  const handleExportJSON = () => {
    triggerDownload(JSON.stringify(graph, null, 2), 'literature_network.json', 'application/json');
  };

  const selectedPaperDetails =
    selectedNode && selectedNode.node_type === 'paper'
      ? papers.find((p) => p.id === selectedNode.id)
      : null;

  const cs = graph.coverage_stats || {};
  const interp = graph.interpretation || {};

  return (
    <div className="space-y-5">
      {/* Top Multilayer Filter & Export Bar */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-base font-bold text-slate-900">
                Multilayer Scientific Literature & Knowledge Network
              </h2>
              <span className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
                {visibleNodes.length} Visible Nodes • {visibleEdges.length} Active Links
              </span>
            </div>
            <p className="text-xs text-slate-500 mt-0.5">
              Separates verified bibliographic links (Layer A), computed semantic similarity (Layer B), scientific claims/gaps (Layer C), and research lineage (Layer D).
            </p>
          </div>

          {/* Direct Research Exports */}
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] font-semibold text-slate-500 mr-1">Export Graph:</span>
            <button
              onClick={handleExportGraphML}
              className="px-2.5 py-1 rounded-md text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition"
              title="Export standard GraphML for Gephi / Cytoscape / NetworkX"
            >
              GraphML
            </button>
            <button
              onClick={handleExportNodesCSV}
              className="px-2.5 py-1 rounded-md text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition"
            >
              Nodes CSV
            </button>
            <button
              onClick={handleExportEdgesCSV}
              className="px-2.5 py-1 rounded-md text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition"
            >
              Edges CSV
            </button>
            <button
              onClick={handleExportJSON}
              className="px-2.5 py-1 rounded-md text-xs font-semibold bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition"
            >
              Graph JSON
            </button>
          </div>
        </div>

        {/* Layer Tabs + View Controls */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-100">
          <div className="flex flex-wrap items-center gap-1.5">
            {[
              { id: 'all', label: `All Layers (${graph.edges.length})` },
              {
                id: 'bibliographic',
                label: `Layer A: Bibliographic (${
                  (cs.direct_citation_edges_count || 0) +
                  (cs.bibliographic_coupling_edges_count || 0) +
                  (cs.co_citation_edges_count || 0)
                })`,
              },
              {
                id: 'semantic_conceptual',
                label: `Layer B: Semantic (${cs.semantic_edges_count || 0})`,
              },
              {
                id: 'scientific_evidence',
                label: `Layer C: Claims & Gaps (${cs.evidence_edges_count || 0})`,
              },
              {
                id: 'research_evolution',
                label: `Layer D: Lineage (${cs.evolution_edges_count || 0})`,
              },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setLayerFilter(tab.id)}
                className={`px-2.5 py-1 rounded-lg text-xs font-semibold border transition ${
                  layerFilter === tab.id
                    ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                    : 'bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-100'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <label className="inline-flex items-center gap-1.5 text-xs text-slate-600 cursor-pointer">
              <input
                type="checkbox"
                checked={showClaimAndGapNodes}
                onChange={(e) => setShowClaimAndGapNodes(e.target.checked)}
                className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
              />
              Show Claim & Gap Nodes
            </label>
            {cs.total_context_only_papers > 0 && (
              <label className="inline-flex items-center gap-1.5 text-xs text-slate-600 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showContextNodes}
                  onChange={(e) => setShowContextNodes(e.target.checked)}
                  className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                />
                Context Papers ({cs.total_context_only_papers})
              </label>
            )}

            <div className="inline-flex rounded-lg bg-slate-100 p-0.5 border border-slate-200 text-xs">
              <button
                onClick={() => setLayoutMode('timeline')}
                className={`px-2.5 py-1 rounded-md font-semibold ${
                  layoutMode === 'timeline'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Timeline × Impact
              </button>
              <button
                onClick={() => setLayoutMode('force')}
                className={`px-2.5 py-1 rounded-md font-semibold ${
                  layoutMode === 'force'
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Force Cluster
              </button>
            </div>

            <select
              value={colorMode}
              onChange={(e) => setColorMode(e.target.value)}
              className="bg-slate-50 border border-slate-300 rounded-lg px-2.5 py-1 text-xs text-slate-700"
            >
              <option value="cluster">Color: Thematic Cluster</option>
              <option value="importance">Color: Network Role</option>
              <option value="stance">Color: Agreement Stance</option>
              <option value="source">Color: Database Source</option>
            </select>
          </div>
        </div>
      </div>

      {/* Main Canvas + Inspector Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-5">
        {/* Interactive SVG Network Canvas */}
        <div className="lg:col-span-3 bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex flex-col">
          <div className="relative w-full h-[540px] bg-slate-50 rounded-xl border border-slate-200 overflow-hidden">
            <svg viewBox="0 0 1000 560" className="w-full h-full select-none">
              <defs>
                <marker
                  id="arrow-citation"
                  viewBox="0 0 10 10"
                  refX="18"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 1 L 10 5 L 0 9 z" fill="#0284c7" />
                </marker>
                <marker
                  id="arrow-evolution"
                  viewBox="0 0 10 10"
                  refX="18"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 1 L 10 5 L 0 9 z" fill="#7c3aed" />
                </marker>
                <marker
                  id="arrow-evidence"
                  viewBox="0 0 10 10"
                  refX="18"
                  refY="5"
                  markerWidth="6"
                  markerHeight="6"
                  orient="auto-start-reverse"
                >
                  <path d="M 0 1 L 10 5 L 0 9 z" fill="#0f766e" />
                </marker>
              </defs>

              {layoutMode === 'timeline' && (
                <g className="opacity-70">
                  <line
                    x1="60"
                    y1="515"
                    x2="940"
                    y2="515"
                    stroke="#94a3b8"
                    strokeWidth="1"
                    strokeDasharray="4 4"
                  />
                  <text x="500" y="542" fill="#475569" fontSize="11" fontWeight="600" textAnchor="middle">
                    Publication Timeline (Older → Newer) • Claims & Gaps Anchored on Right
                  </text>
                  <text
                    x="22"
                    y="280"
                    fill="#475569"
                    fontSize="11"
                    fontWeight="600"
                    textAnchor="middle"
                    transform="rotate(-90, 22, 280)"
                  >
                    Citation & Relevance Impact ↑
                  </text>
                </g>
              )}

              {/* Edges */}
              {visibleEdges.map((edge) => {
                const s = nodeMap[edge.source];
                const t = nodeMap[edge.target];
                if (!s || !t) return null;
                const x1 = (layoutMode === 'timeline' ? s.x_timeline : s.x_force) * 1000;
                const y1 = (layoutMode === 'timeline' ? s.y_timeline : s.y_force) * 520;
                const x2 = (layoutMode === 'timeline' ? t.x_timeline : t.x_force) * 1000;
                const y2 = (layoutMode === 'timeline' ? t.y_timeline : t.y_force) * 520;
                const isSelectedEdge = selectedEdge && selectedEdge.id === edge.id;
                const isConnectedToSelectedNode =
                  selectedNode && (selectedNode.id === s.id || selectedNode.id === t.id);
                const isHighlighted = isSelectedEdge || isConnectedToSelectedNode;

                const strokeColor = getEdgeColor(edge, isHighlighted);
                let markerEnd = undefined;
                if (edge.directed || edge.edge_type === 'citation') {
                  if (edge.edge_type === 'research_evolution') markerEnd = 'url(#arrow-evolution)';
                  else if (edge.edge_type === 'evidence_relationship') markerEnd = 'url(#arrow-evidence)';
                  else if (edge.edge_type === 'citation') markerEnd = 'url(#arrow-citation)';
                }

                return (
                  <g
                    key={edge.id}
                    onClick={(ev) => {
                      ev.stopPropagation();
                      setSelectedEdge(edge);
                      setSelectedNode(null);
                    }}
                    className="cursor-pointer"
                  >
                    {/* Invisible wider hit target for easy clicking */}
                    <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="transparent" strokeWidth="9" />
                    <line
                      x1={x1}
                      y1={y1}
                      x2={x2}
                      y2={y2}
                      stroke={strokeColor}
                      strokeWidth={
                        isSelectedEdge
                          ? 3.2
                          : isHighlighted
                          ? 2.4
                          : edge.edge_type === 'citation' || edge.edge_type === 'research_evolution'
                          ? 1.9
                          : 1.3
                      }
                      strokeDasharray={
                        edge.edge_type === 'semantic'
                          ? '4 4'
                          : edge.edge_type === 'co_citation'
                          ? '2 3'
                          : undefined
                      }
                      markerEnd={markerEnd}
                      opacity={
                        (selectedNode || selectedEdge) && !isHighlighted ? 0.22 : 0.85
                      }
                    />
                  </g>
                );
              })}

              {/* Nodes */}
              {visibleNodes.map((node) => {
                const cx = (layoutMode === 'timeline' ? node.x_timeline : node.x_force) * 1000;
                const cy = (layoutMode === 'timeline' ? node.y_timeline : node.y_force) * 520;
                const radius =
                  node.node_type !== 'paper'
                    ? 11
                    : Math.min(22, Math.max(10, 9 + Math.log10((node.citations || 0) + 1) * 4.5));
                const fill = getNodeColor(node);
                const isSelected = selectedNode && selectedNode.id === node.id;
                const isContextOnly =
                  node.is_citation_context_only ||
                  node.inclusion_role === 'citation_context_publication';

                return (
                  <g
                    key={node.id}
                    onClick={() => {
                      setSelectedNode(node);
                      setSelectedEdge(null);
                    }}
                    className="cursor-pointer transition-transform"
                  >
                    {isSelected && (
                      <circle
                        cx={cx}
                        cy={cy}
                        r={radius + 6}
                        fill="none"
                        stroke="#1e1b4b"
                        strokeWidth="2"
                        strokeDasharray="3 3"
                      />
                    )}

                    {node.node_type === 'scientific_claim' ? (
                      <rect
                        x={cx - 12}
                        y={cy - 10}
                        width={24}
                        height={20}
                        rx={5}
                        fill={fill}
                        stroke={isSelected ? '#1e1b4b' : '#ffffff'}
                        strokeWidth={2}
                      />
                    ) : node.node_type === 'knowledge_gap' ? (
                      <polygon
                        points={`${cx},${cy - 12} ${cx + 12},${cy} ${cx},${cy + 12} ${cx - 12},${cy}`}
                        fill={fill}
                        stroke={isSelected ? '#1e1b4b' : '#ffffff'}
                        strokeWidth={2}
                      />
                    ) : (
                      <circle
                        cx={cx}
                        cy={cy}
                        r={radius}
                        fill={fill}
                        fillOpacity={isContextOnly ? 0.55 : 0.92}
                        stroke={isSelected ? '#1e1b4b' : isContextOnly ? '#475569' : '#ffffff'}
                        strokeWidth={isSelected ? 2.5 : 2}
                        strokeDasharray={isContextOnly ? '3 2' : undefined}
                      />
                    )}

                    <text
                      x={cx}
                      y={cy - radius - 5}
                      fill="#1e293b"
                      fontSize="10"
                      fontWeight="700"
                      textAnchor="middle"
                    >
                      {node.label}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Visual Legend */}
          <div className="mt-3 pt-2.5 border-t border-slate-100 flex flex-wrap items-center justify-between gap-2 text-[11px] text-slate-600">
            <div className="flex flex-wrap items-center gap-3">
              <span className="inline-flex items-center gap-1">
                <span className="w-3 h-0.5 bg-sky-600 inline-block" /> Directed Citation (Layer A)
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="w-3 h-0.5 bg-violet-600 inline-block" /> Lineage Extension (Layer D)
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="w-3 h-0.5 border-b border-dashed border-slate-500 inline-block" /> Semantic Similarity (Layer B)
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-sm bg-teal-700 inline-block" /> Synthesized Claim (Layer C)
              </span>
              <span className="inline-flex items-center gap-1">
                <span className="w-2.5 h-2.5 rotate-45 bg-amber-700 inline-block" /> Verified Gap (Layer C)
              </span>
            </div>
            <span className="text-slate-400">
              Tip: Click any node or edge line to inspect its provenance & explanation
            </span>
          </div>
        </div>

        {/* Right Column: Node Inspector OR Edge Inspector */}
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex flex-col justify-between">
          {selectedEdge ? (
            <div className="space-y-3 text-xs">
              <div className="flex items-center justify-between">
                <span className="inline-block px-2 py-0.5 rounded text-[10px] font-semibold uppercase bg-indigo-50 text-indigo-700 border border-indigo-200">
                  Edge Inspector • {selectedEdge.network_layer || 'bibliographic'}
                </span>
                <button
                  onClick={() => setSelectedEdge(null)}
                  className="text-slate-400 hover:text-slate-700 text-xs font-semibold"
                >
                  ✕ Clear
                </button>
              </div>

              <div>
                <h3 className="text-sm font-bold text-slate-900">
                  {(nodeMap[selectedEdge.source]?.label || selectedEdge.source)}{' '}
                  {selectedEdge.directed ? '→' : '↔'}{' '}
                  {(nodeMap[selectedEdge.target]?.label || selectedEdge.target)}
                </h3>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  Subtype: <code className="text-slate-700 font-semibold">{selectedEdge.relationship_subtype || selectedEdge.edge_subtype || selectedEdge.edge_type}</code>
                </p>
              </div>

              <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 space-y-1.5">
                <div>
                  <span className="text-slate-500 font-semibold block">Relationship Explanation:</span>
                  <span className="text-slate-800">
                    {selectedEdge.relationship_explanation || selectedEdge.label}
                  </span>
                </div>
                {selectedEdge.shared_concepts && selectedEdge.shared_concepts.length > 0 && (
                  <div>
                    <span className="text-slate-500 font-semibold block">Shared Concepts / Features:</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {selectedEdge.shared_concepts.map((c, idx) => (
                        <span
                          key={idx}
                          className="px-1.5 py-0.5 rounded bg-white border border-slate-200 text-[10px] text-slate-700"
                        >
                          {c}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="space-y-1.5 border-t border-slate-200 pt-2.5">
                <div className="flex justify-between">
                  <span className="text-slate-500">Verification Status:</span>
                  <span className="font-semibold text-slate-800">
                    {selectedEdge.verification_status || selectedEdge.provenance}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Raw Weight:</span>
                  <span className="font-mono text-slate-800">{selectedEdge.raw_weight ?? selectedEdge.weight}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Normalized Weight:</span>
                  <span className="font-mono text-slate-800">
                    {selectedEdge.normalized_weight ?? selectedEdge.weight}
                  </span>
                </div>
                <div>
                  <span className="text-slate-500 block">Weighting Method:</span>
                  <span className="text-slate-700 text-[11px]">
                    {selectedEdge.weighting_method || selectedEdge.provenance}
                  </span>
                </div>
              </div>
            </div>
          ) : selectedNode ? (
            <div className="space-y-3.5 text-xs">
              <div>
                <div className="flex items-center justify-between gap-1 mb-1.5">
                  <span className="inline-block px-2 py-0.5 rounded text-[10px] font-semibold uppercase bg-indigo-50 text-indigo-700 border border-indigo-200">
                    {selectedNode.node_type === 'paper'
                      ? selectedNode.is_citation_context_only
                        ? 'Citation-Context Paper'
                        : `Included Study [${selectedNode.citation_index || '—'}]`
                      : selectedNode.node_type === 'scientific_claim'
                      ? 'Synthesized Claim Node'
                      : selectedNode.node_type === 'knowledge_gap'
                      ? 'Verified Gap Node'
                      : 'Hugging Face Artifact'}
                  </span>
                  <button
                    onClick={() => setSelectedNode(null)}
                    className="text-slate-400 hover:text-slate-700 text-xs font-semibold"
                  >
                    ✕
                  </button>
                </div>

                <h3 className="text-sm font-bold text-slate-900 leading-snug">{selectedNode.full_title}</h3>
                <p className="text-xs text-slate-500 mt-1">
                  {selectedNode.year ? `Year: ${selectedNode.year} • ` : ''}
                  {selectedNode.node_type === 'paper'
                    ? `${(selectedNode.citations || 0).toLocaleString()} citations • ${selectedNode.source}`
                    : selectedNode.source}
                </p>
              </div>

              {/* Network Roles & Rationale */}
              {selectedNode.importance_roles && selectedNode.importance_roles.length > 0 && (
                <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 space-y-1.5">
                  <span className="text-[11px] font-semibold text-slate-600 block">
                    Network Structural Role(s):
                  </span>
                  <div className="flex flex-wrap gap-1">
                    {selectedNode.importance_roles.map((r) => (
                      <span
                        key={r}
                        className="px-2 py-0.5 rounded-full text-[10px] font-semibold text-white"
                        style={{ backgroundColor: ROLE_COLORS[r] || '#475569' }}
                      >
                        {ROLE_LABELS[r] || r}
                      </span>
                    ))}
                  </div>
                  {selectedNode.importance_rationale && (
                    <p className="text-[11px] text-slate-600 leading-relaxed pt-1">
                      {selectedNode.importance_rationale}
                    </p>
                  )}
                </div>
              )}

              {/* Cluster & Metadata Completeness */}
              <div className="space-y-1 border-t border-slate-200 pt-2.5 text-[11px]">
                <div className="flex justify-between">
                  <span className="text-slate-500">Semantic Theme:</span>
                  <span className="font-semibold text-slate-800">{selectedNode.cluster_label}</span>
                </div>
                {selectedNode.bibliographic_cluster_label && (
                  <div className="flex justify-between">
                    <span className="text-slate-500">Bibliographic Stream:</span>
                    <span className="font-semibold text-slate-800">
                      {selectedNode.bibliographic_cluster_label}
                    </span>
                  </div>
                )}
                {selectedNode.metadata_completeness && (
                  <div className="flex justify-between">
                    <span className="text-slate-500">Metadata Coverage:</span>
                    <span className="font-mono text-slate-700">{selectedNode.metadata_completeness}</span>
                  </div>
                )}
              </div>

              {selectedPaperDetails && (
                <div className="space-y-2 border-t border-slate-200 pt-2.5">
                  <div>
                    <span className="text-slate-500 font-semibold block">Methodology:</span>
                    <span className="text-slate-800">{selectedPaperDetails.elicit?.methodology}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 font-semibold block">Key Finding:</span>
                    <span className="text-slate-800">{selectedPaperDetails.elicit?.key_findings}</span>
                  </div>
                  <div>
                    <span className="text-slate-500 font-semibold block">Limitation:</span>
                    <span className="text-slate-600">{selectedPaperDetails.elicit?.limitations}</span>
                  </div>
                </div>
              )}

              <div className="pt-2.5 border-t border-slate-200 space-y-2">
                {selectedNode.node_type === 'paper' && (
                  <button
                    onClick={handleExpandNode}
                    disabled={expanding}
                    className="w-full py-2 px-3 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold shadow-sm transition"
                  >
                    {expanding ? 'Fetching Citation Network...' : '+ Expand Node Citations'}
                  </button>
                )}
                {selectedNode.url && (
                  <a
                    href={selectedNode.url}
                    target="_blank"
                    rel="noreferrer"
                    className="block w-full text-center py-2 px-3 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-800 text-xs font-medium"
                  >
                    Open Source Page ↗
                  </a>
                )}
              </div>
            </div>
          ) : (
            <div className="text-xs text-slate-500 space-y-3">
              <h3 className="text-sm font-bold text-slate-900">Multilayer Node & Edge Inspector</h3>
              <p>
                Click any <strong>Study</strong>, <strong>Synthesized Claim</strong>, <strong>Knowledge Gap</strong>, or <strong>Hugging Face Artifact</strong> node to inspect its structural role, rationale, and metadata completeness.
              </p>
              <p>
                Click any <strong>connecting edge line</strong> to inspect its layer (`bibliographic`, `semantic_conceptual`, `scientific_evidence`, `research_evolution`), raw/normalized weights, and scientific relationship explanation.
              </p>
              <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-200 space-y-1 text-[11px]">
                <div className="font-semibold text-slate-700">Network Coverage Summary</div>
                <div>• Included Synthesis Studies: {cs.included_studies_count ?? papers.length}</div>
                <div>• Verified Direct Citations: {cs.direct_citation_edges_count ?? 0}</div>
                <div>• Bibliographic Coupling / Co-Citation: {(cs.bibliographic_coupling_edges_count || 0) + (cs.co_citation_edges_count || 0)}</div>
                <div>• Semantic Similarity Links: {cs.semantic_edges_count ?? 0}</div>
                <div>• Evidence Claim / Dataset / Gap Links: {cs.evidence_edges_count ?? 0}</div>
                <div>• Isolated Included Studies: {cs.isolated_included_studies_count ?? 0}</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Scientific Network Interpretation & Thematic Structure Panel (Phase 17) */}
      {(interp.research_themes_summary || (graph.themes && graph.themes.length > 0)) && (
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Grounded Scientific Network Interpretation & Intellectual Structure
              </h3>
              <p className="text-xs text-slate-500">
                Computed strictly from observed network topology, citation metadata coverage, and cross-referenced Section 1–3 evidence.
              </p>
            </div>
            {graph.validation_report && (
              <span className="px-2.5 py-1 rounded-full text-[11px] font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                ✓ Bibliographic & Mathematical Graph Invariants Verified
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            {interp.research_themes_summary && (
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                <div className="font-bold text-slate-800 mb-1">1. Major Research Streams & Themes</div>
                <p className="text-slate-600 leading-relaxed">{interp.research_themes_summary}</p>
              </div>
            )}
            {interp.intellectual_foundations_summary && (
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                <div className="font-bold text-slate-800 mb-1">2. Foundational, Hub & Bridge Contributions</div>
                <p className="text-slate-600 leading-relaxed">
                  {interp.intellectual_foundations_summary}{' '}
                  {interp.cross_theme_connections_summary || ''}
                </p>
              </div>
            )}
            {interp.scientific_evidence_relationships_summary && (
              <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                <div className="font-bold text-slate-800 mb-1">
                  3. Evidence Convergence, Contradictions & Lineage
                </div>
                <p className="text-slate-600 leading-relaxed">
                  {interp.scientific_evidence_relationships_summary}{' '}
                  {interp.research_development_summary || ''}
                </p>
              </div>
            )}
            {interp.network_limitations_and_disclosures &&
              interp.network_limitations_and_disclosures.length > 0 && (
                <div className="p-3 rounded-lg bg-amber-50/60 border border-amber-200">
                  <div className="font-bold text-amber-900 mb-1">
                    4. Citation Coverage & Structural Sparsity Disclosures
                  </div>
                  <p className="text-amber-800 leading-relaxed">
                    {interp.network_limitations_and_disclosures.join(' ')}
                  </p>
                </div>
              )}
          </div>
        </div>
      )}
    </div>
  );
}

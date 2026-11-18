"""
The map's optional evidence layer (Layer C): study-to-claim links from the synthesis, shared-dataset links from the
evidence table and study-to-gap links from the gap matrix. Kept out of services/network_engine.py, whose default
map uses citation links only; services/graph_builder.py adds this layer when the user switches it on.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx

from models import (
    GraphEdge,
    GraphNode,
    KnowledgeGapMatrix,
    PaperRecord,
    StructuredEvidenceMatrix,
    SynthesisEvidencePackage,
)


def add_evidence_layer(
    papers: List[PaperRecord],
    edges: List[GraphEdge],
    G_layout: nx.Graph,
    module_key: str,
    synthesis_package: Optional[SynthesisEvidencePackage] = None,
    evidence_matrix: Optional[StructuredEvidenceMatrix] = None,
    gap_matrix: Optional[KnowledgeGapMatrix] = None,
) -> Tuple[List[GraphNode], List[GraphNode], Dict[int, List[str]], Dict[int, List[str]]]:
    """Adds the evidence edges to `edges` and `G_layout`; returns the claim nodes, gap nodes, and the claims and gaps
    each study (by citation index) bears on."""
    # ========================================================================
    # 4. LAYER C: Claim-Centered Scientific Evidence & Shared Dataset Network (Phase 9)
    # ========================================================================
    claim_nodes: List[GraphNode] = []
    gap_nodes: List[GraphNode] = []
    claims_by_study_idx: Dict[int, List[str]] = {}
    gaps_by_study_idx: Dict[int, List[str]] = {}

    if synthesis_package and synthesis_package.claims:
        for c_idx, claim in enumerate(synthesis_package.claims, start=1):
            c_node_id = f"claim:{claim.claim_id}"
            claim_nodes.append(
                GraphNode(
                    id=c_node_id,
                    label=f"Claim {claim.claim_id}",
                    full_title=claim.claim_text,
                    node_type="scientific_claim",
                    citation_index=None,
                    year=2025,
                    citations=len(claim.supporting_indices),
                    source="Section 1 Synthesis",
                    cluster_id=0,
                    cluster_label=claim.theme_label or "Synthesized Claims",
                    stance="supporting",
                    canonical_entity_id=claim.claim_id,
                    inclusion_role="synthesized_claim_node",
                    importance_roles=["contextual_or_artifact_node"],
                    importance_rationale=f"Section 1 verified claim ({claim.verification.state})",
                    domain=module_key,
                    related_claim_ids=[claim.claim_id],
                    x_timeline=0.52,
                    y_timeline=round(0.18 + (c_idx * 0.18) % 0.65, 4),
                    x_force=0.50,
                    y_force=round(0.20 + (c_idx * 0.18) % 0.65, 4),
                    pagerank=0.12,
                    provenance="Section 1 SynthesisEvidencePackage",
                )
            )
            G_layout.add_node(c_node_id)

            # Map supporting, challenging, inconclusive, and contextual study-to-claim edges
            pub_by_cit_idx = {p.citation_index: p for p in papers}
            role_lists = [
                ("supports_claim", "supports", claim.supporting_indices, claim.supporting_result_ids, 0.90),
                ("challenges_claim", "challenges", claim.challenging_indices, claim.challenging_result_ids, 0.90),
                ("inconclusive_for_claim", "inconclusive for", claim.inconclusive_indices, claim.inconclusive_result_ids, 0.65),
            ]
            for subtype, verb_lbl, s_idxs, r_ids, w_val in role_lists:
                for s_idx in s_idxs:
                    p_obj = pub_by_cit_idx.get(s_idx)
                    if not p_obj:
                        continue
                    claims_by_study_idx.setdefault(s_idx, []).append(claim.claim_id)
                    edges.append(
                        GraphEdge(
                            id=f"ev:{p_obj.id}->{c_node_id}:{subtype}",
                            source=p_obj.id,
                            target=c_node_id,
                            edge_type="evidence_relationship",
                            provenance="evidence_verified",
                            weight=w_val,
                            label=f"{verb_lbl} {claim.claim_id}",
                            relationship_subtype=subtype,
                            network_layer="scientific_evidence",
                            directed=True,
                            raw_weight=1.0,
                            normalized_weight=w_val,
                            weighting_method="Section 1 claim-evidence verification mapping",
                            construction_method="Bipartite Study -> Claim link from verified AtomicFinding records",
                            relationship_explanation=(
                                f"Study [{s_idx}] ({p_obj.title[:55]}) {verb_lbl} Claim {claim.claim_id}: "
                                f"\"{claim.claim_text[:110]}...\""
                            ),
                            supporting_source_ids=[p_obj.id],
                            supporting_result_ids=list(r_ids),
                            supporting_claim_ids=[claim.claim_id],
                            verification_status=claim.verification.state,
                        )
                    )
                    G_layout.add_edge(p_obj.id, c_node_id, weight=0.55)

            # Paper-to-paper evidence projection when one study supports and another challenges the same claim (Case G)
            for sup_i in claim.supporting_indices:
                for chal_j in claim.challenging_indices:
                    p_sup = pub_by_cit_idx.get(sup_i)
                    p_chal = pub_by_cit_idx.get(chal_j)
                    if p_sup and p_chal and p_sup.id != p_chal.id:
                        edges.append(
                            GraphEdge(
                                id=f"ev_disagree:{p_sup.id}<->{p_chal.id}:{claim.claim_id}",
                                source=p_sup.id,
                                target=p_chal.id,
                                edge_type="evidence_relationship",
                                provenance="evidence_verified",
                                weight=0.85,
                                label=f"divergent on {claim.claim_id}",
                                relationship_subtype="contradictory_evidence_on_claim",
                                network_layer="scientific_evidence",
                                directed=False,
                                raw_weight=1.0,
                                normalized_weight=0.85,
                                weighting_method="Bipartite claim-disagreement projection",
                                construction_method="Projected from opposing study stances on shared Section 1 claim",
                                relationship_explanation=(
                                    f"Study [{sup_i}] supports whereas Study [{chal_j}] challenges or qualifies "
                                    f"Claim {claim.claim_id} ({claim.theme_label})."
                                ),
                                supporting_source_ids=[p_sup.id, p_chal.id],
                                supporting_claim_ids=[claim.claim_id],
                                verification_status="Verified from Section 1 & 2 evidence records",
                            )
                        )

    # Shared dataset / cohort edges from Section 2 StructuredEvidenceMatrix (Case F)
    if evidence_matrix and evidence_matrix.evidence_units:
        ds_groups: Dict[str, List[int]] = {}
        for u in evidence_matrix.evidence_units:
            ds_norm = (u.context_or_system or "").strip().lower()
            if ds_norm and ds_norm not in ("not reported", "unavailable", "unspecified", "general domain"):
                ds_groups.setdefault(ds_norm, []).append(u.citation_index)
        pub_by_cit_idx = {p.citation_index: p for p in papers}
        for ds_name, c_idxs in ds_groups.items():
            if len(c_idxs) >= 2:
                for idx_a in range(len(c_idxs)):
                    for idx_b in range(idx_a + 1, len(c_idxs)):
                        pa = pub_by_cit_idx.get(c_idxs[idx_a])
                        pb = pub_by_cit_idx.get(c_idxs[idx_b])
                        if pa and pb:
                            edges.append(
                                GraphEdge(
                                    id=f"ev_dataset:{pa.id}<->{pb.id}",
                                    source=pa.id,
                                    target=pb.id,
                                    edge_type="evidence_relationship",
                                    provenance="evidence_verified",
                                    weight=0.80,
                                    label="shared dataset/system",
                                    relationship_subtype="uses_shared_dataset",
                                    network_layer="scientific_evidence",
                                    directed=False,
                                    raw_weight=1.0,
                                    normalized_weight=0.80,
                                    weighting_method="Section 2 context/dataset identity match",
                                    construction_method="Verified shared dataset/system in EvidenceUnitRecord",
                                    relationship_explanation=(
                                        f"Studies [{pa.citation_index}] and [{pb.citation_index}] evaluate the same underlying "
                                        f"dataset or experimental system ('{pa.elicit.dataset_or_system}'); tracked to prevent assuming independent cohort replication."
                                    ),
                                    supporting_source_ids=[pa.id, pb.id],
                                    supporting_result_ids=[f"R-{pa.citation_index}-1", f"R-{pb.citation_index}-1"],
                                    verification_status="Verified in Section 2 Evidence Matrix",
                                )
                            )

    # Knowledge Gap nodes & edges from Section 3 KnowledgeGapMatrix (Phase 20)
    if gap_matrix and gap_matrix.structured_gaps:
        pub_by_cit_idx = {p.citation_index: p for p in papers}
        for g_idx, sg in enumerate(gap_matrix.structured_gaps[:3], start=1):
            g_node_id = f"gap:{sg.gap_id}"
            gap_nodes.append(
                GraphNode(
                    id=g_node_id,
                    label=f"{sg.gap_id} ({sg.taxonomy_code})",
                    full_title=f"{sg.gap_title}: {sg.unresolved_issue}",
                    node_type="knowledge_gap",
                    citation_index=None,
                    year=2025,
                    citations=len(sg.supporting_study_indices),
                    source="Section 3 Gap Engine",
                    cluster_id=0,
                    cluster_label=sg.primary_gap_type,
                    stance="mixed",
                    canonical_entity_id=sg.gap_id,
                    inclusion_role="verified_gap_node",
                    importance_roles=["contextual_or_artifact_node"],
                    importance_rationale=f"{sg.verification_status} ({sg.priority_assessment})",
                    domain=module_key,
                    related_claim_ids=list(sg.related_claim_ids),
                    related_gap_ids=[sg.gap_id],
                    x_timeline=0.82,
                    y_timeline=round(0.22 + (g_idx * 0.22) % 0.65, 4),
                    x_force=0.82,
                    y_force=round(0.22 + (g_idx * 0.22) % 0.65, 4),
                    pagerank=0.09,
                    provenance="Section 3 KnowledgeGapMatrix",
                )
            )
            G_layout.add_node(g_node_id)
            for s_idx in sg.supporting_study_indices[:4]:
                p_obj = pub_by_cit_idx.get(s_idx)
                if p_obj:
                    gaps_by_study_idx.setdefault(s_idx, []).append(sg.gap_id)
                    edges.append(
                        GraphEdge(
                            id=f"ev_gap:{p_obj.id}->{g_node_id}",
                            source=p_obj.id,
                            target=g_node_id,
                            edge_type="evidence_relationship",
                            provenance="evidence_verified",
                            weight=0.75,
                            label=f"motivates {sg.gap_id}",
                            relationship_subtype="motivates_knowledge_gap",
                            network_layer="scientific_evidence",
                            directed=True,
                            raw_weight=1.0,
                            normalized_weight=0.75,
                            weighting_method="Section 3 evidence-gap provenance link",
                            construction_method="Linked via supporting_study_indices in StructuredKnowledgeGap",
                            relationship_explanation=(
                                f"Limitations or validation boundaries in Study [{s_idx}] motivate "
                                f"{sg.gap_id} ({sg.taxonomy_code}: {sg.primary_gap_type})."
                            ),
                            supporting_source_ids=[p_obj.id],
                            supporting_result_ids=list(sg.supporting_result_ids[:2]),
                            supporting_claim_ids=list(sg.related_claim_ids[:2]),
                            verification_status=sg.verification_status,
                        )
                    )
                    G_layout.add_edge(p_obj.id, g_node_id, weight=0.45)
    return claim_nodes, gap_nodes, claims_by_study_idx, gaps_by_study_idx

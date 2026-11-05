import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (
    ElicitExtraction,
    GraphEdge,
    HFResourceRecord,
    KnowledgeGapMatrix,
    PaperRecord,
    StructuredKnowledgeGap,
    SynthesisClaim,
    SynthesisEvidencePackage,
)
from services.evidence_table_engine import build_structured_evidence_matrix
from services.graph_builder import build_literature_graph
from services.network_engine import (
    build_multilayer_literature_network,
    export_network_edges_csv,
    export_network_json,
    export_network_nodes_csv,
    export_network_to_graphml,
    validate_literature_network,
)


class TestSection4LiteratureNetwork(unittest.TestCase):
    def _make_sample_corpus(self):
        p1 = PaperRecord(
            id="s2:1001",
            title="Foundational Graph Neural Network Architecture for Molecular Property Prediction",
            authors=["Alice Kipf", "Max Welling"],
            year=2018,
            venue="ICLR",
            source="Semantic Scholar",
            citation_count=950,
            citation_index=1,
            doi="10.1000/gnn2018",
            arxiv_id="1801.00001",
            s2_id="1001",
            stance="supporting",
            abstract=(
                "We introduce a foundational message passing graph neural network for molecular property "
                "prediction evaluated on the MoleculeNet QM9 benchmark achieving 4.2% MAE reduction."
            ),
            references=["doi:10.1000/chem2015", "doi:10.1000/message2016"],
            cited_by=["s2:1002", "s2:1003", "s2:9999"],
            elicit=ElicitExtraction(
                methodology="Message-passing graph neural network architecture",
                dataset_or_system="MoleculeNet QM9 Benchmark",
                key_findings="Achieved 4.2% MAE reduction on QM9 quantum chemical properties.",
                limitations="Limited scalability on large 3D protein complexes.",
            ),
        )
        p2 = PaperRecord(
            id="s2:1002",
            title="Equivariant Graph Neural Networks Extending Molecular Property Prediction on QM9",
            authors=["Victor Satorras", "Max Welling"],
            year=2022,
            venue="ICML",
            source="arXiv",
            citation_count=320,
            citation_index=2,
            doi="10.1000/egnn2022",
            arxiv_id="2201.00002",
            s2_id="1002",
            stance="supporting",
            abstract=(
                "Building on foundational message passing graph neural networks, we propose SE(3)-equivariant "
                "message passing for molecular property prediction on the MoleculeNet QM9 benchmark."
            ),
            references=["doi:10.1000/gnn2018", "doi:10.1000/chem2015", "doi:10.1000/message2016"],
            cited_by=["s2:1004", "s2:9999"],
            elicit=ElicitExtraction(
                methodology="SE(3)-equivariant graph neural network",
                dataset_or_system="MoleculeNet QM9 Benchmark",
                key_findings="Improves 3D conformer molecular property prediction accuracy by 11%.",
                limitations="Higher GPU memory overhead for long-range atomic interactions.",
            ),
        )
        p3 = PaperRecord(
            id="s2:1003",
            title="Critical Evaluation of Graph Neural Networks Under Scaffold Split in Molecular Property Prediction",
            authors=["Zhenqin Wu", "Regina Barzilay"],
            year=2024,
            venue="Nature Machine Intelligence",
            source="PubMed",
            citation_count=45,
            citation_index=3,
            doi="10.1000/crit2024",
            pmid="38001122",
            s2_id="1003",
            stance="contradicting",
            abstract=(
                "We evaluate message passing graph neural networks for molecular property prediction on the "
                "MoleculeNet QM9 benchmark and out-of-distribution scaffold splits, finding severe degradation."
            ),
            references=["doi:10.1000/gnn2018", "doi:10.1000/chem2015"],
            cited_by=["s2:9999"],
            elicit=ElicitExtraction(
                methodology="Out-of-distribution scaffold split benchmark evaluation",
                dataset_or_system="MoleculeNet QM9 Benchmark",
                key_findings="GNN performance drops by 19% under strict scaffold splitting compared to fingerprints.",
                limitations="Restricted to small organic molecules under 60 heavy atoms.",
            ),
        )
        p4 = PaperRecord(
            id="pubmed:2004",
            title="Quantum Density Functional Theory Corrections for Macrocyclic Transition Metal Complexes",
            authors=["Elena Rostova"],
            year=2025,
            venue="JACS",
            source="PubMed",
            citation_count=4,
            citation_index=4,
            pmid="39009988",
            stance="mixed",
            abstract=(
                "We formulate dispersion-corrected density functional theory approximations for macrocyclic "
                "organometallic complexes without citation overlap."
            ),
            references=[],
            cited_by=[],
            elicit=ElicitExtraction(
                methodology="Dispersion-corrected DFT orbital analysis",
                dataset_or_system="Transition Metal Macrocycle Set",
                key_findings="Reduces spin-state energetics error to 1.4 kcal/mol.",
                limitations="Computationally prohibitive above 200 atoms.",
            ),
        )
        return [p1, p2, p3, p4]

    def test_multilayer_bibliographic_and_evolution_edges(self):
        """
        Tests Layer A (verified directed citations, normalized bibliographic coupling, co-citation)
        and Layer D (documented chronological & methodological extension) without suppressing
        simultaneous relationships between the same pair of studies.
        """
        papers = self._make_sample_corpus()
        graph = build_literature_graph(
            papers=papers,
            hf_resources=[],
            query="graph neural networks for molecular property prediction",
            mode="standard",
        )

        # 1. Directed Citations: p2 (2022) -> p1 (2018) and p3 (2024) -> p1 (2018)
        cit_edges = [e for e in graph.edges if e.edge_type == "citation"]
        cit_pairs = {(e.source, e.target) for e in cit_edges}
        self.assertIn(("s2:1002", "s2:1001"), cit_pairs)
        self.assertIn(("s2:1003", "s2:1001"), cit_pairs)
        self.assertNotIn(("s2:1001", "s2:1002"), cit_pairs)
        for ce in cit_edges:
            self.assertTrue(ce.directed)
            self.assertEqual(ce.network_layer, "bibliographic")

        # 2. Bibliographic Coupling: p2 and p3 share references ("doi:10.1000/gnn2018", "doi:10.1000/chem2015")
        #    AND p1 and p2 also share references even though p2 directly cites p1!
        bc_edges = [e for e in graph.edges if e.edge_type == "bibliographic_coupling"]
        self.assertGreaterEqual(len(bc_edges), 2)
        for bce in bc_edges:
            self.assertFalse(bce.directed)
            self.assertGreaterEqual(bce.raw_weight, 1.0)
            self.assertGreater(bce.normalized_weight, 0.0)
            self.assertIn("Salton cosine", bce.weighting_method)

        # 3. Co-Citation: p1, p2, p3 are all cited by s2:9999
        cocit_edges = [e for e in graph.edges if e.edge_type == "co_citation"]
        self.assertGreaterEqual(len(cocit_edges), 1)
        for cce in cocit_edges:
            self.assertTrue(len(cce.citation_universe_scope) > 0)

        # 4. Layer D Research Evolution: p1 (2018) -> p2 (2022) grounded in verified citation + chronology
        evol_edges = [e for e in graph.edges if e.network_layer == "research_evolution"]
        self.assertGreaterEqual(len(evol_edges), 1)
        self.assertTrue(any(e.source == "s2:1001" and e.target == "s2:1002" for e in evol_edges))

        # 5. Partial metadata disclosure on p4 (no reference list retrieved from PubMed)
        node_by_id = {n.id: n for n in graph.nodes}
        self.assertIn("reference list unavailable", node_by_id["pubmed:2004"].metadata_completeness)
        self.assertIn("Complete", node_by_id["s2:1001"].metadata_completeness)

    def test_semantic_similarity_coexists_with_contradictory_evidence_and_shared_dataset(self):
        """
        Tests Case F (Shared Dataset) and Case G (High Semantic Similarity with Contradictory Findings):
        p2 (supporting) and p3 (contradicting) share high conceptual similarity AND evaluate on the
        same MoleculeNet QM9 Benchmark while having conflicting stances on Claim C1.
        """
        papers = self._make_sample_corpus()
        synth_pkg = SynthesisEvidencePackage(
            claims=[
                SynthesisClaim(
                    claim_id="C1",
                    claim_text="Message-passing GNNs improve accuracy on QM9 but degrade under scaffold splitting.",
                    theme_label="GNN Generalization",
                    supporting_indices=[1, 2],
                    challenging_indices=[3],
                )
            ]
        )
        ev_matrix = build_structured_evidence_matrix(
            query="graph neural networks for molecular property prediction",
            papers=papers,
            synthesis_package=synth_pkg,
        )
        gap_matrix = KnowledgeGapMatrix(
            structured_gaps=[
                StructuredKnowledgeGap(
                    gap_id="GAP-1",
                    gap_title="Scalability of Equivariant GNNs to Large 3D Protein Complexes",
                    gap_statement="Equivariant GNNs have not been validated on large 3D protein complexes above 200 atoms.",
                    supporting_study_indices=[1, 2],
                    priority_assessment="High relevance to the research question",
                )
            ]
        )

        graph = build_multilayer_literature_network(
            papers=papers,
            hf_resources=[
                HFResourceRecord(
                    id="deepchem/gnn-qm9",
                    resource_type="model",
                    pipeline_tag="graph-ml",
                    downloads=12500,
                    url="https://huggingface.co/deepchem/gnn-qm9",
                    linked_arxiv_ids=["1801.00001"],
                )
            ],
            query="graph neural networks for molecular property prediction",
            mode="standard",
            synthesis_package=synth_pkg,
            evidence_matrix=ev_matrix,
            gap_matrix=gap_matrix,
        )

        # Verify Claim Node & Gap Node exist in Layer C
        node_types = {n.id: n.node_type for n in graph.nodes}
        self.assertEqual(node_types.get("claim:C1"), "scientific_claim")
        self.assertEqual(node_types.get("gap:GAP-1"), "knowledge_gap")

        # Verify supports_claim, challenges_claim, and contradictory_evidence_on_claim edges
        subtypes = {e.relationship_subtype for e in graph.edges}
        self.assertIn("supports_claim", subtypes)
        self.assertIn("challenges_claim", subtypes)
        self.assertIn("contradictory_evidence_on_claim", subtypes)
        self.assertIn("uses_shared_dataset", subtypes)
        self.assertIn("motivates_knowledge_gap", subtypes)
        self.assertIn("verified_arxiv_implementation", subtypes)

        # Verify Semantic Similarity (Layer B) coexists with Contradictory Evidence (Layer C) between s2:1002 and s2:1003
        sem_edges = [
            e for e in graph.edges
            if e.network_layer == "semantic_conceptual"
            and {e.source, e.target} == {"s2:1002", "s2:1003"}
        ]
        contra_edges = [
            e for e in graph.edges
            if e.relationship_subtype == "contradictory_evidence_on_claim"
            and {e.source, e.target} == {"s2:1002", "s2:1003"}
        ]
        self.assertEqual(len(sem_edges), 1)
        self.assertEqual(len(contra_edges), 1)
        self.assertIn("does not imply scientific agreement", sem_edges[0].relationship_explanation)

    def test_context_only_separation_small_network_safeguard_and_exports(self):
        """
        Tests Case D (Extended Citation Context separation from Included Synthesis Studies),
        Case K (Small-network clustering safeguard for N < 5), deterministic validation repair,
        and GraphML / CSV / JSON exports.
        """
        papers = self._make_sample_corpus()  # 4 included studies (< 5 triggers small-network safeguard)
        context_paper = PaperRecord(
            id="s2:9999",
            title="Comprehensive Review of Geometric Deep Learning in Chemistry",
            authors=["Michael Bronstein"],
            year=2023,
            source="Semantic Scholar",
            citation_count=410,
            s2_id="9999",
            is_citation_context_only=True,
            references=["doi:10.1000/gnn2018", "doi:10.1000/egnn2022", "doi:10.1000/crit2024"],
        )

        graph = build_multilayer_literature_network(
            papers=papers,
            hf_resources=[],
            query="graph neural networks for molecular property prediction",
            mode="standard",
            context_papers=[context_paper],
        )

        # Included study count remains strictly 4 while context paper is tracked separately
        self.assertEqual(graph.coverage_stats.included_studies_count, 4)
        self.assertEqual(graph.coverage_stats.context_publications_count, 1)
        self.assertEqual(graph.metadata.included_study_count, 4)
        self.assertEqual(graph.metadata.context_document_count, 1)
        self.assertTrue(graph.validation_report.small_network_safeguards_applied)

        ctx_node = next(n for n in graph.nodes if n.id == "s2:9999")
        self.assertTrue(ctx_node.is_citation_context_only)
        self.assertEqual(ctx_node.inclusion_role, "citation_context_publication")

        # Inject a self-loop and dangling edge to test deterministic validation repair
        graph.edges.append(
            GraphEdge(
                id="bad:self",
                source="s2:1001",
                target="s2:1001",
                edge_type="citation",
            )
        )
        graph.edges.append(
            GraphEdge(
                id="bad:dangling",
                source="s2:1001",
                target="nonexistent:node",
                edge_type="semantic",
            )
        )
        val_report = validate_literature_network(graph, expected_included_count=4)
        self.assertTrue(val_report.bibliographic_valid)
        self.assertTrue(val_report.mathematical_network_valid)
        self.assertGreaterEqual(val_report.repaired_elements_count, 2)
        edge_ids = {e.id for e in graph.edges}
        self.assertNotIn("bad:self", edge_ids)
        self.assertNotIn("bad:dangling", edge_ids)

        # Verify GraphML, Nodes CSV, Edges CSV, and JSON exports
        graphml_str = export_network_to_graphml(graph)
        self.assertIn("<graphml", graphml_str)
        self.assertIn('node id="s2:1001"', graphml_str)

        nodes_csv = export_network_nodes_csv(graph)
        self.assertIn("Inclusion_Role", nodes_csv)
        self.assertIn("s2:1001", nodes_csv)

        edges_csv = export_network_edges_csv(graph)
        self.assertIn("Relationship_Explanation", edges_csv)
        self.assertIn("bibliographic", edges_csv)

        json_str = export_network_json(graph)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["coverage_stats"]["included_studies_count"], 4)


if __name__ == "__main__":
    unittest.main()

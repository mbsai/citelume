"""The map's optional evidence layer (services/network_evidence_layer.py): claim, shared-dataset and gap links,
shown only when the user switches the map to view "all". Kept apart from the map tests, which need no synthesis or
gap models."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (
    HFResourceRecord,
    KnowledgeGapMatrix,
    StructuredKnowledgeGap,
    SynthesisClaim,
    SynthesisEvidencePackage,
)
from services.evidence_table_engine import build_structured_evidence_matrix
from services.graph_builder import build_literature_graph
from tests.test_section4_literature_network import sample_papers


class TestMapEvidenceLayer(unittest.TestCase):
    def _make_sample_corpus(self):
        return sample_papers()

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

        graph = build_literature_graph(
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
            view="all",
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

    def test_hidden_in_the_default_view(self):
        papers = self._make_sample_corpus()
        synth = SynthesisEvidencePackage(claims=[SynthesisClaim(claim_id="C1", claim_text="x", theme_label="t",
                                                                supporting_indices=[1, 2])])
        graph = build_literature_graph(papers=papers, hf_resources=[], synthesis_package=synth)
        self.assertFalse(any(n.node_type == "scientific_claim" for n in graph.nodes))
        self.assertEqual({e.network_layer for e in graph.edges}, {"bibliographic"})


if __name__ == "__main__":
    unittest.main()

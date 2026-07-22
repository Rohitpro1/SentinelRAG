"""
Unit 3.2 tests -- PlannerNode: the first genuinely executable
orchestration node. Covers exactly the six areas instruction 6 lists:
empty queries, whitespace normalization, normal queries, serialization
through GraphState, node execution within the graph, plus classification
rule coverage and settings-driven configurability.
"""
import pytest

from app.core.settings.planner import PlannerSettings
from app.orchestration.graph_builder import GraphBuilder
from app.orchestration.graph_state import GraphState
from app.orchestration.nodes.planner import PlannerNode
from app.schemas.planning import PlanningMetadata, QueryClassification


# --- Empty queries ---

@pytest.mark.asyncio
async def test_empty_string_classifies_as_empty():
    node = PlannerNode()
    result = await node(GraphState(original_query=""))
    assert result.planning_metadata.classification == QueryClassification.EMPTY
    assert result.planning_metadata.word_count == 0
    assert result.planning_metadata.character_count == 0


@pytest.mark.asyncio
async def test_whitespace_only_string_classifies_as_empty():
    """Whitespace-only input normalizes to an empty string, not a single-space TOO_SHORT."""
    node = PlannerNode()
    result = await node(GraphState(original_query="     \n\t  "))
    assert result.planning_metadata.normalized_query == ""
    assert result.planning_metadata.classification == QueryClassification.EMPTY


# --- Whitespace normalization ---

@pytest.mark.asyncio
async def test_leading_and_trailing_whitespace_stripped():
    node = PlannerNode()
    result = await node(GraphState(original_query="   what is the refund policy?   "))
    assert result.planning_metadata.normalized_query == "what is the refund policy?"


@pytest.mark.asyncio
async def test_internal_whitespace_collapsed_to_single_spaces():
    node = PlannerNode()
    result = await node(GraphState(original_query="what   is\tthe\n\nrefund policy?"))
    assert result.planning_metadata.normalized_query == "what is the refund policy?"


@pytest.mark.asyncio
async def test_normalization_preserves_original_casing():
    node = PlannerNode()
    result = await node(GraphState(original_query="  What IS the Refund Policy?  "))
    assert result.planning_metadata.normalized_query == "What IS the Refund Policy?"


# --- Normal queries / classification rules ---

@pytest.mark.asyncio
async def test_question_ending_in_question_mark_classifies_as_question():
    node = PlannerNode()
    result = await node(GraphState(original_query="Refunds take how long?"))
    assert result.planning_metadata.classification == QueryClassification.QUESTION


@pytest.mark.asyncio
async def test_query_starting_with_question_word_classifies_as_question_even_without_mark():
    node = PlannerNode()
    result = await node(GraphState(original_query="What is the refund policy"))
    assert result.planning_metadata.classification == QueryClassification.QUESTION


@pytest.mark.asyncio
async def test_declarative_sentence_classifies_as_statement():
    node = PlannerNode()
    result = await node(GraphState(original_query="Refunds are processed within five business days."))
    assert result.planning_metadata.classification == QueryClassification.STATEMENT


@pytest.mark.asyncio
async def test_too_short_query_below_threshold():
    node = PlannerNode()
    result = await node(GraphState(original_query="refunds"))
    assert result.planning_metadata.classification == QueryClassification.TOO_SHORT


@pytest.mark.asyncio
async def test_multiple_question_marks_classify_as_multi_part():
    node = PlannerNode()
    result = await node(GraphState(original_query="What is the refund policy? What about shipping?"))
    assert result.planning_metadata.classification == QueryClassification.MULTI_PART


@pytest.mark.asyncio
async def test_word_count_and_character_count_are_accurate():
    node = PlannerNode()
    result = await node(GraphState(original_query="refund policy details"))
    assert result.planning_metadata.word_count == 3
    assert result.planning_metadata.character_count == len("refund policy details")


# --- Settings-driven configurability ---

@pytest.mark.asyncio
async def test_min_words_threshold_is_configurable():
    settings = PlannerSettings(min_words_threshold=5)
    node = PlannerNode(settings)
    result = await node(GraphState(original_query="refund policy details here"))  # 4 words < 5
    assert result.planning_metadata.classification == QueryClassification.TOO_SHORT


@pytest.mark.asyncio
async def test_multi_part_threshold_is_configurable():
    settings = PlannerSettings(multi_part_question_mark_threshold=3)
    node = PlannerNode(settings)
    result = await node(GraphState(original_query="What about A? What about B?"))  # only 2 '?' now below threshold of 3
    assert result.planning_metadata.classification == QueryClassification.QUESTION


# --- rewritten_query left unset ---

@pytest.mark.asyncio
async def test_rewritten_query_remains_unset():
    node = PlannerNode()
    result = await node(GraphState(original_query="what is the refund policy?"))
    assert result.rewritten_query is None


@pytest.mark.asyncio
async def test_rewritten_query_untouched_if_already_set():
    """PlannerNode must not clobber a pre-existing rewritten_query (e.g. from a future retry pass)."""
    node = PlannerNode()
    result = await node(GraphState(original_query="q", rewritten_query="already rewritten"))
    assert result.rewritten_query == "already rewritten"


# --- Serialization through GraphState ---

@pytest.mark.asyncio
async def test_planning_metadata_survives_graphstate_serialization_round_trip():
    node = PlannerNode()
    result = await node(GraphState(original_query="what is the refund policy?"))
    dumped = result.model_dump()
    restored = GraphState.model_validate(dumped)
    assert restored.planning_metadata == result.planning_metadata
    assert restored.planning_metadata.classification == QueryClassification.QUESTION


@pytest.mark.asyncio
async def test_planning_metadata_survives_json_round_trip():
    node = PlannerNode()
    result = await node(GraphState(original_query="refund policy"))
    restored = GraphState.model_validate_json(result.model_dump_json())
    assert restored.planning_metadata.normalized_query == "refund policy"


# --- Node execution within the graph ---

@pytest.mark.asyncio
async def test_planner_node_executes_when_run_through_the_compiled_graph():
    compiled = GraphBuilder().build()
    result = await compiled.ainvoke(GraphState(original_query="  what   is the refund policy?  "))
    reconstructed = GraphState(**result)
    assert reconstructed.planning_metadata is not None
    assert reconstructed.planning_metadata.normalized_query == "what is the refund policy?"
    assert reconstructed.planning_metadata.classification == QueryClassification.QUESTION


@pytest.mark.asyncio
async def test_planner_node_via_graph_leaves_retrieval_and_verification_untouched():
    """
    Rescoped for Unit 3.5: GraphBuilder() now defaults a real DecisionNode
    into the graph alongside PlannerNode (both are cheap/pure, same
    reasoning), so `decision` is no longer None by default -- it's
    genuinely computed from the (empty, since no retrieval/verification
    ran) evidence. This test now asserts what's actually still true:
    retrieval_result and verification_result remain untouched, since no
    RetrievalNode/VerificationNode was injected.
    """
    compiled = GraphBuilder().build()
    result = await compiled.ainvoke(GraphState(original_query="q", retry_count=2))
    reconstructed = GraphState(**result)
    assert reconstructed.retry_count == 2
    assert reconstructed.retrieval_result is None
    assert reconstructed.verification_result is None


@pytest.mark.asyncio
async def test_graph_uses_injected_custom_planner_node():
    """Confirms GraphBuilder actually uses the injected planner_node, not always a hardcoded default."""
    custom_settings = PlannerSettings(min_words_threshold=100)  # everything short of 100 words -> TOO_SHORT
    custom_node = PlannerNode(custom_settings)
    compiled = GraphBuilder(planner_node=custom_node).build()
    result = await compiled.ainvoke(GraphState(original_query="what is the refund policy?"))
    reconstructed = GraphState(**result)
    assert reconstructed.planning_metadata.classification == QueryClassification.TOO_SHORT

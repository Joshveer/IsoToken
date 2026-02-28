"""
Task 2.6: Fracture produces valid DAG (no cycles).
Test: For a set of prompts, every emitted PEP has acyclic depends_on.
"""

import pytest
from fracture import fracture, validate_pep, is_pep_dag


def test_fracture_outputs_acyclic_pep_for_multi_question():
    pep = fracture("What is A? What is B?")
    validate_pep(pep)
    assert is_pep_dag(pep)


def test_fracture_outputs_acyclic_pep_for_compare():
    pep = fracture("Compare X and Y.")
    validate_pep(pep)
    assert is_pep_dag(pep)


def test_fracture_outputs_acyclic_pep_for_pros_cons():
    pep = fracture("List pros and cons of Z.")
    validate_pep(pep)
    assert is_pep_dag(pep)


def test_fracture_outputs_acyclic_pep_for_verify_critique():
    pep = fracture("Solve it. Then verify your answer.")
    validate_pep(pep)
    assert is_pep_dag(pep)


def test_fracture_outputs_acyclic_pep_for_single_prompt():
    pep = fracture("What is 2+2?")
    validate_pep(pep)
    assert is_pep_dag(pep)


def test_pep_with_cycle_fails_dag_check():
    """Manually constructed PEP with cycle must fail is_pep_dag."""
    pep = {
        "task_id": "x", "global_context": "", "nodes": [
            {"node_id": "n1", "type": "a", "adapter": "a", "prompt": "p", "depends_on": ["n2"]},
            {"node_id": "n2", "type": "a", "adapter": "a", "prompt": "p", "depends_on": ["n1"]},
        ],
        "aggregation": {"strategy": "vote"},
    }
    assert not is_pep_dag(pep)

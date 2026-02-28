"""
Task 4.2: Critic-synthesis flow: parallel outputs → Critic LoRA reviews → Synthesizer produces final answer.
Test: Given a set of node outputs, critic-synthesis path returns a single final answer; critic and synthesizer invoked in order.
"""


def test_critic_synthesis_returns_single_final_answer():
    """aggregate_synthesize(outputs, critic_fn, synthesizer_fn) returns one final answer."""
    from aggregate import aggregate_synthesize
    call_order = []

    def critic(outputs):
        call_order.append("critic")
        return "review"

    def synthesizer(review):
        call_order.append("synthesizer")
        return "final answer"

    result = aggregate_synthesize(["out1", "out2"], critic_fn=critic, synthesizer_fn=synthesizer)
    assert result == "final answer"
    assert call_order == ["critic", "synthesizer"]


def test_critic_receives_outputs_synthesizer_receives_review():
    """Critic is called with outputs; synthesizer is called with critic result."""
    from aggregate import aggregate_synthesize

    def critic(outputs):
        assert outputs == ["a", "b"]
        return "reviewed"

    def synthesizer(review):
        assert review == "reviewed"
        return "synthesis"

    assert aggregate_synthesize(["a", "b"], critic_fn=critic, synthesizer_fn=synthesizer) == "synthesis"

from agents.base.rag_orchestrator import RAGOrchestrator

def test_rag_orchestrator_end_to_end():
    orchestrator = RAGOrchestrator()
    workflow_id = 'test1'
    orchestrator.start_workflow(workflow_id, {'query': '  test query  ', 'context': 'test context'})
    result = orchestrator.run_workflow(workflow_id)
    assert 'final_output' in result
    assert 'evaluation' in result
    assert 'log' in result
    steps = [entry['step'] for entry in result['log']]
    assert steps == [
        'query_rewriting',
        'context_assessment',
        'source_selection',
        'prompt_building',
        'llm_call',
        'response_evaluation'
    ]

def test_rag_orchestrator_user_feedback():
    orchestrator = RAGOrchestrator()
    workflow_id = 'wf-feedback'
    orchestrator.start_workflow(workflow_id, {'query': 'short', 'context': 'ctx'})
    orchestrator.run_workflow(workflow_id)
    feedback = {'rating': 1.0, 'comment': 'Great!'}
    orchestrator.add_user_feedback(workflow_id, feedback)
    feedbacks = orchestrator.get_user_feedback(workflow_id)
    assert feedbacks and feedbacks[0]['feedback']['rating'] == 1.0
    # Run again and check confidence is higher due to positive feedback
    result2 = orchestrator.run_workflow(workflow_id)
    assert result2['evaluation']['confidence'] > 0.3  # Should be adjusted upward by feedback 
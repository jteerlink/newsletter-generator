from agents.base.agentic_logger import AgenticLogger
import time

def test_agentic_logger_logging_and_metrics():
    logger = AgenticLogger()
    logger.log_action('test_action', {'foo': 'bar'})
    logs = logger.get_logs()
    assert logs and logs[0]['action'] == 'test_action'
    # Start/end workflow and check metrics
    logger.start_workflow('wf1')
    time.sleep(0.01)
    logger.end_workflow('wf1', iteration_count=2, answer_quality=0.8, escalated=True)
    metrics = logger.get_metrics()
    assert metrics['total_workflows'] == 1
    assert metrics['avg_latency'] is not None
    assert metrics['avg_answer_quality'] == 0.8
    assert metrics['avg_iterations'] == 2
    assert metrics['escalation_rate'] == 1.0 
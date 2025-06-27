import time
from typing import Dict, Any, List

class AgenticLogger:
    """
    Centralized logger and metrics tracker for agentic RAG workflows.
    Logs actions, tool calls, decisions, and tracks metrics.
    """
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.metrics = {
            'retrieval_latency': [],
            'answer_quality': [],
            'iteration_counts': [],
            'escalation_rates': 0,
            'total_workflows': 0
        }
        self._workflow_start_times = {}

    def log_action(self, action: str, details: Dict[str, Any]):
        entry = {'action': action, 'details': details, 'timestamp': time.time()}
        self.logs.append(entry)

    def start_workflow(self, workflow_id: str):
        self._workflow_start_times[workflow_id] = time.time()
        self.metrics['total_workflows'] += 1

    def end_workflow(self, workflow_id: str, iteration_count: int, answer_quality: float = None, escalated: bool = False):
        start = self._workflow_start_times.pop(workflow_id, None)
        if start:
            latency = time.time() - start
            self.metrics['retrieval_latency'].append(latency)
        self.metrics['iteration_counts'].append(iteration_count)
        if answer_quality is not None:
            self.metrics['answer_quality'].append(answer_quality)
        if escalated:
            self.metrics['escalation_rates'] += 1

    def get_logs(self) -> List[Dict[str, Any]]:
        return self.logs

    def get_metrics(self) -> Dict[str, Any]:
        # Compute averages for reporting
        metrics = self.metrics.copy()
        if metrics['retrieval_latency']:
            metrics['avg_latency'] = sum(metrics['retrieval_latency']) / len(metrics['retrieval_latency'])
        else:
            metrics['avg_latency'] = None
        if metrics['answer_quality']:
            metrics['avg_answer_quality'] = sum(metrics['answer_quality']) / len(metrics['answer_quality'])
        else:
            metrics['avg_answer_quality'] = None
        if metrics['iteration_counts']:
            metrics['avg_iterations'] = sum(metrics['iteration_counts']) / len(metrics['iteration_counts'])
        else:
            metrics['avg_iterations'] = None
        if metrics['total_workflows']:
            metrics['escalation_rate'] = metrics['escalation_rates'] / metrics['total_workflows']
        else:
            metrics['escalation_rate'] = None
        return metrics 
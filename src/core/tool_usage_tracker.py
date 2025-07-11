"""
Tool Usage Tracking System

Provides comprehensive tracking of tool usage across the newsletter generation system,
including performance metrics, success rates, usage patterns, and analytics.
"""

import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class ToolExecutionStatus(Enum):
    """Status of tool execution"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class ToolUsageEntry:
    """Individual tool usage record"""
    timestamp: datetime
    tool_name: str
    agent_name: str
    execution_time: float
    status: ToolExecutionStatus
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    error_message: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    workflow_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'tool_name': self.tool_name,
            'agent_name': self.agent_name,
            'execution_time': self.execution_time,
            'status': self.status.value,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'error_message': self.error_message,
            'context': self.context or {},
            'session_id': self.session_id,
            'workflow_id': self.workflow_id
        }

@dataclass
class ToolPerformanceMetrics:
    """Performance metrics for a specific tool"""
    tool_name: str
    total_invocations: int
    successful_invocations: int
    failed_invocations: int
    average_execution_time: float
    min_execution_time: float
    max_execution_time: float
    success_rate: float
    last_used: datetime
    error_patterns: Dict[str, int]
    usage_trends: Dict[str, int]  # usage count by time period

@dataclass
class ToolUsageAnalytics:
    """Comprehensive analytics for tool usage"""
    total_tools_tracked: int
    total_invocations: int
    average_success_rate: float
    most_used_tools: List[str]
    least_reliable_tools: List[str]
    fastest_tools: List[str]
    slowest_tools: List[str]
    error_summary: Dict[str, int]
    usage_by_agent: Dict[str, int]
    time_period_summary: Dict[str, Dict[str, Any]]

class ToolUsageLogger:
    """
    Centralized tool usage tracking and analytics system
    """
    
    def __init__(self, 
                 log_file: str = "logs/tool_usage.json",
                 metrics_file: str = "logs/tool_metrics.json",
                 max_entries: int = 10000,
                 analytics_window_hours: int = 24):
        
        self.log_file = Path(log_file)
        self.metrics_file = Path(metrics_file)
        self.max_entries = max_entries
        self.analytics_window_hours = analytics_window_hours
        
        # Thread-safe operations
        self._lock = threading.Lock()
        
        # In-memory cache for recent entries
        self._recent_entries: List[ToolUsageEntry] = []
        self._performance_cache: Dict[str, ToolPerformanceMetrics] = {}
        
        # Ensure log directories exist
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize files if they don't exist
        self._initialize_files()
        
        # Load existing data
        self._load_recent_entries()
        
        logger.info(f"ToolUsageLogger initialized - tracking to {self.log_file}")
    
    def _initialize_files(self):
        """Initialize log files if they don't exist"""
        if not self.log_file.exists():
            initial_data = {
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "entries": []
            }
            with open(self.log_file, 'w') as f:
                json.dump(initial_data, f, indent=2)
        
        if not self.metrics_file.exists():
            initial_metrics = {
                "last_updated": datetime.now().isoformat(),
                "version": "1.0",
                "tool_metrics": {},
                "system_analytics": {}
            }
            with open(self.metrics_file, 'w') as f:
                json.dump(initial_metrics, f, indent=2)
    
    def _load_recent_entries(self):
        """Load recent entries from log file"""
        try:
            with open(self.log_file, 'r') as f:
                data = json.load(f)
            
            # Load recent entries (last 1000)
            recent_entries = data.get("entries", [])[-1000:]
            self._recent_entries = []
            
            for entry_data in recent_entries:
                try:
                    entry = ToolUsageEntry(
                        timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                        tool_name=entry_data["tool_name"],
                        agent_name=entry_data["agent_name"],
                        execution_time=entry_data["execution_time"],
                        status=ToolExecutionStatus(entry_data["status"]),
                        input_data=entry_data["input_data"],
                        output_data=entry_data["output_data"],
                        error_message=entry_data.get("error_message"),
                        context=entry_data.get("context"),
                        session_id=entry_data.get("session_id"),
                        workflow_id=entry_data.get("workflow_id")
                    )
                    self._recent_entries.append(entry)
                except Exception as e:
                    logger.warning(f"Failed to load entry: {e}")
            
            logger.info(f"Loaded {len(self._recent_entries)} recent tool usage entries")
        
        except Exception as e:
            logger.warning(f"Failed to load existing tool usage data: {e}")
            self._recent_entries = []
    
    @contextmanager
    def track_tool_usage(self, 
                        tool_name: str, 
                        agent_name: str,
                        input_data: Dict[str, Any] = None,
                        context: Dict[str, Any] = None,
                        session_id: str = None,
                        workflow_id: str = None):
        """
        Context manager for tracking tool usage
        
        Usage:
            with tracker.track_tool_usage("search_web", "ResearchAgent", {"query": "AI news"}):
                result = search_web("AI news")
        """
        
        start_time = time.time()
        status = ToolExecutionStatus.SUCCESS
        error_message = None
        output_data = {}
        
        try:
            logger.debug(f"Starting tool usage tracking: {tool_name} by {agent_name}")
            yield
            
        except Exception as e:
            status = ToolExecutionStatus.FAILURE
            error_message = str(e)
            logger.error(f"Tool execution failed: {tool_name} - {error_message}")
            raise
        
        finally:
            execution_time = time.time() - start_time
            
            # Create usage entry
            entry = ToolUsageEntry(
                timestamp=datetime.now(),
                tool_name=tool_name,
                agent_name=agent_name,
                execution_time=execution_time,
                status=status,
                input_data=input_data or {},
                output_data=output_data,
                error_message=error_message,
                context=context,
                session_id=session_id,
                workflow_id=workflow_id
            )
            
            # Log the entry
            self.log_tool_usage(entry)
    
    def log_tool_usage(self, entry: ToolUsageEntry):
        """Log a tool usage entry"""
        with self._lock:
            # Add to recent entries
            self._recent_entries.append(entry)
            
            # Maintain max entries limit
            if len(self._recent_entries) > self.max_entries:
                self._recent_entries = self._recent_entries[-self.max_entries:]
            
            # Update performance cache
            self._update_performance_cache(entry)
            
            # Persist to file (async to avoid blocking)
            self._persist_entry(entry)
            
            logger.debug(f"Logged tool usage: {entry.tool_name} by {entry.agent_name} - {entry.status.value}")
    
    def _update_performance_cache(self, entry: ToolUsageEntry):
        """Update in-memory performance metrics"""
        tool_name = entry.tool_name
        
        if tool_name not in self._performance_cache:
            self._performance_cache[tool_name] = ToolPerformanceMetrics(
                tool_name=tool_name,
                total_invocations=0,
                successful_invocations=0,
                failed_invocations=0,
                average_execution_time=0.0,
                min_execution_time=float('inf'),
                max_execution_time=0.0,
                success_rate=0.0,
                last_used=entry.timestamp,
                error_patterns={},
                usage_trends={}
            )
        
        metrics = self._performance_cache[tool_name]
        
        # Update basic metrics
        metrics.total_invocations += 1
        if entry.status == ToolExecutionStatus.SUCCESS:
            metrics.successful_invocations += 1
        else:
            metrics.failed_invocations += 1
        
        # Update timing metrics
        metrics.average_execution_time = (
            (metrics.average_execution_time * (metrics.total_invocations - 1) + entry.execution_time) 
            / metrics.total_invocations
        )
        metrics.min_execution_time = min(metrics.min_execution_time, entry.execution_time)
        metrics.max_execution_time = max(metrics.max_execution_time, entry.execution_time)
        
        # Update success rate
        metrics.success_rate = metrics.successful_invocations / metrics.total_invocations
        
        # Update last used
        metrics.last_used = entry.timestamp
        
        # Track error patterns
        if entry.error_message:
            error_key = entry.error_message[:100]  # First 100 chars of error
            metrics.error_patterns[error_key] = metrics.error_patterns.get(error_key, 0) + 1
        
        # Track usage trends (hourly)
        hour_key = entry.timestamp.strftime("%Y-%m-%d %H:00")
        metrics.usage_trends[hour_key] = metrics.usage_trends.get(hour_key, 0) + 1
    
    def _persist_entry(self, entry: ToolUsageEntry):
        """Persist entry to log file"""
        try:
            # Read existing data
            with open(self.log_file, 'r') as f:
                data = json.load(f)
            
            # Add new entry
            data["entries"].append(entry.to_dict())
            
            # Maintain max entries in file
            if len(data["entries"]) > self.max_entries:
                data["entries"] = data["entries"][-self.max_entries:]
            
            data["last_updated"] = datetime.now().isoformat()
            
            # Write back to file
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to persist tool usage entry: {e}")
    
    def get_tool_performance(self, tool_name: str) -> Optional[ToolPerformanceMetrics]:
        """Get performance metrics for a specific tool"""
        return self._performance_cache.get(tool_name)
    
    def get_all_tool_metrics(self) -> Dict[str, ToolPerformanceMetrics]:
        """Get performance metrics for all tools"""
        return self._performance_cache.copy()
    
    def generate_usage_analytics(self, 
                                hours_back: int = None) -> ToolUsageAnalytics:
        """Generate comprehensive usage analytics"""
        if hours_back is None:
            hours_back = self.analytics_window_hours
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        # Filter recent entries
        recent_entries = [
            entry for entry in self._recent_entries 
            if entry.timestamp >= cutoff_time
        ]
        
        if not recent_entries:
            return ToolUsageAnalytics(
                total_tools_tracked=0,
                total_invocations=0,
                average_success_rate=0.0,
                most_used_tools=[],
                least_reliable_tools=[],
                fastest_tools=[],
                slowest_tools=[],
                error_summary={},
                usage_by_agent={},
                time_period_summary={}
            )
        
        # Calculate analytics
        tool_usage_counts = defaultdict(int)
        tool_success_counts = defaultdict(int)
        tool_execution_times = defaultdict(list)
        error_summary = defaultdict(int)
        usage_by_agent = defaultdict(int)
        
        for entry in recent_entries:
            tool_usage_counts[entry.tool_name] += 1
            usage_by_agent[entry.agent_name] += 1
            
            if entry.status == ToolExecutionStatus.SUCCESS:
                tool_success_counts[entry.tool_name] += 1
            
            tool_execution_times[entry.tool_name].append(entry.execution_time)
            
            if entry.error_message:
                error_key = entry.error_message.split('\n')[0][:50]  # First line, truncated
                error_summary[error_key] += 1
        
        # Calculate success rates
        tool_success_rates = {}
        for tool in tool_usage_counts:
            success_count = tool_success_counts.get(tool, 0)
            total_count = tool_usage_counts[tool]
            tool_success_rates[tool] = success_count / total_count if total_count > 0 else 0.0
        
        # Calculate average execution times
        tool_avg_times = {}
        for tool, times in tool_execution_times.items():
            tool_avg_times[tool] = sum(times) / len(times) if times else 0.0
        
        # Most used tools
        most_used_tools = sorted(tool_usage_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        most_used_tools = [tool for tool, count in most_used_tools]
        
        # Least reliable tools
        least_reliable_tools = sorted(tool_success_rates.items(), key=lambda x: x[1])[:5]
        least_reliable_tools = [tool for tool, rate in least_reliable_tools if rate < 0.9]
        
        # Fastest and slowest tools
        fastest_tools = sorted(tool_avg_times.items(), key=lambda x: x[1])[:5]
        fastest_tools = [tool for tool, time in fastest_tools]
        
        slowest_tools = sorted(tool_avg_times.items(), key=lambda x: x[1], reverse=True)[:5]
        slowest_tools = [tool for tool, time in slowest_tools]
        
        # Time period summary
        time_period_summary = self._generate_time_period_summary(recent_entries)
        
        return ToolUsageAnalytics(
            total_tools_tracked=len(tool_usage_counts),
            total_invocations=len(recent_entries),
            average_success_rate=sum(tool_success_rates.values()) / len(tool_success_rates) if tool_success_rates else 0.0,
            most_used_tools=most_used_tools,
            least_reliable_tools=least_reliable_tools,
            fastest_tools=fastest_tools,
            slowest_tools=slowest_tools,
            error_summary=dict(error_summary),
            usage_by_agent=dict(usage_by_agent),
            time_period_summary=time_period_summary
        )
    
    def _generate_time_period_summary(self, entries: List[ToolUsageEntry]) -> Dict[str, Dict[str, Any]]:
        """Generate time-based usage summary"""
        hourly_data = defaultdict(lambda: {"count": 0, "success_count": 0, "avg_time": 0.0, "tools": set()})
        
        for entry in entries:
            hour_key = entry.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_data[hour_key]["count"] += 1
            hourly_data[hour_key]["tools"].add(entry.tool_name)
            
            if entry.status == ToolExecutionStatus.SUCCESS:
                hourly_data[hour_key]["success_count"] += 1
        
        # Convert to serializable format
        summary = {}
        for hour, data in hourly_data.items():
            summary[hour] = {
                "total_invocations": data["count"],
                "successful_invocations": data["success_count"],
                "success_rate": data["success_count"] / data["count"] if data["count"] > 0 else 0.0,
                "unique_tools_used": len(data["tools"])
            }
        
        return summary
    
    def get_tool_usage_history(self, 
                              tool_name: str = None,
                              agent_name: str = None,
                              hours_back: int = 24) -> List[ToolUsageEntry]:
        """Get filtered tool usage history"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        filtered_entries = []
        for entry in self._recent_entries:
            if entry.timestamp < cutoff_time:
                continue
            
            if tool_name and entry.tool_name != tool_name:
                continue
            
            if agent_name and entry.agent_name != agent_name:
                continue
            
            filtered_entries.append(entry)
        
        return filtered_entries
    
    def save_metrics_report(self, output_file: str = None):
        """Save comprehensive metrics report to file"""
        if output_file is None:
            output_file = f"logs/tool_usage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        analytics = self.generate_usage_analytics()
        tool_metrics = {name: asdict(metrics) for name, metrics in self._performance_cache.items()}
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "analytics_window_hours": self.analytics_window_hours,
            "system_analytics": asdict(analytics),
            "tool_performance_metrics": tool_metrics,
            "recent_entries_count": len(self._recent_entries)
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Tool usage metrics report saved to {output_file}")
        return output_file

# Global instance for easy access
_global_tool_tracker: Optional[ToolUsageLogger] = None

def get_tool_tracker() -> ToolUsageLogger:
    """Get global tool usage tracker instance"""
    global _global_tool_tracker
    if _global_tool_tracker is None:
        _global_tool_tracker = ToolUsageLogger()
    return _global_tool_tracker

def track_tool_call(tool_name: str, 
                   agent_name: str,
                   input_data: Dict[str, Any] = None,
                   context: Dict[str, Any] = None,
                   session_id: str = None,
                   workflow_id: str = None):
    """Decorator/context manager for tracking tool calls"""
    tracker = get_tool_tracker()
    return tracker.track_tool_usage(
        tool_name=tool_name,
        agent_name=agent_name,
        input_data=input_data,
        context=context,
        session_id=session_id,
        workflow_id=workflow_id
    ) 
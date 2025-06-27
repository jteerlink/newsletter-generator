import sqlite3
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AgenticPersistence:
    """
    Persistence layer for agentic RAG system.
    Stores agent memory, workflow state, logs, and metrics in SQLite database.
    Provides JSON backup functionality for human-readable data.
    """
    
    def __init__(self, db_path: str = "agentic_rag.db", backup_dir: str = "backups"):
        self.db_path = db_path
        self.backup_dir = backup_dir
        os.makedirs(backup_dir, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Agent memory table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            ''')
            
            # Workflow state table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workflow_state (
                    workflow_id TEXT PRIMARY KEY,
                    state_data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            ''')
            
            # Logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    details TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            ''')
            
            # Metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
            ''')
            
            # User feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT NOT NULL,
                    feedback_data TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            ''')
            
            conn.commit()
    
    def save_agent_memory(self, agent_id: str, memory_type: str, data: Dict[str, Any]):
        """Save agent memory to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO agent_memory (agent_id, memory_type, data, timestamp) VALUES (?, ?, ?, ?)',
                (agent_id, memory_type, json.dumps(data), datetime.now().timestamp())
            )
            conn.commit()
    
    def load_agent_memory(self, agent_id: str, memory_type: str) -> List[Dict[str, Any]]:
        """Load agent memory from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT data FROM agent_memory WHERE agent_id = ? AND memory_type = ? ORDER BY timestamp',
                (agent_id, memory_type)
            )
            results = cursor.fetchall()
            return [json.loads(row[0]) for row in results]
    
    def save_workflow_state(self, workflow_id: str, state_data: Dict[str, Any]):
        """Save workflow state to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            now = datetime.now().timestamp()
            cursor.execute(
                '''INSERT OR REPLACE INTO workflow_state 
                   (workflow_id, state_data, created_at, updated_at) 
                   VALUES (?, ?, ?, ?)''',
                (workflow_id, json.dumps(state_data), now, now)
            )
            conn.commit()
    
    def load_workflow_state(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Load workflow state from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT state_data FROM workflow_state WHERE workflow_id = ?',
                (workflow_id,)
            )
            result = cursor.fetchone()
            return json.loads(result[0]) if result else None
    
    def save_log(self, action: str, details: Dict[str, Any]):
        """Save log entry to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO logs (action, details, timestamp) VALUES (?, ?, ?)',
                (action, json.dumps(details), datetime.now().timestamp())
            )
            conn.commit()
    
    def load_logs(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Load recent logs from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT action, details, timestamp FROM logs ORDER BY timestamp DESC LIMIT ?',
                (limit,)
            )
            results = cursor.fetchall()
            return [
                {
                    'action': row[0],
                    'details': json.loads(row[1]),
                    'timestamp': row[2]
                }
                for row in results
            ]
    
    def save_metric(self, metric_type: str, value: float):
        """Save metric to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO metrics (metric_type, value, timestamp) VALUES (?, ?, ?)',
                (metric_type, value, datetime.now().timestamp())
            )
            conn.commit()
    
    def load_metrics(self, metric_type: str, limit: int = 100) -> List[float]:
        """Load recent metrics of a specific type."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT value FROM metrics WHERE metric_type = ? ORDER BY timestamp DESC LIMIT ?',
                (metric_type, limit)
            )
            return [row[0] for row in cursor.fetchall()]
    
    def save_user_feedback(self, workflow_id: str, feedback_data: Dict[str, Any]):
        """Save user feedback to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO user_feedback (workflow_id, feedback_data, timestamp) VALUES (?, ?, ?)',
                (workflow_id, json.dumps(feedback_data), datetime.now().timestamp())
            )
            conn.commit()
    
    def load_user_feedback(self, workflow_id: str = None) -> List[Dict[str, Any]]:
        """Load user feedback from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if workflow_id:
                cursor.execute(
                    'SELECT workflow_id, feedback_data, timestamp FROM user_feedback WHERE workflow_id = ? ORDER BY timestamp',
                    (workflow_id,)
                )
            else:
                cursor.execute(
                    'SELECT workflow_id, feedback_data, timestamp FROM user_feedback ORDER BY timestamp'
                )
            results = cursor.fetchall()
            return [
                {
                    'workflow_id': row[0],
                    'feedback_data': json.loads(row[1]),
                    'timestamp': row[2]
                }
                for row in results
            ]
    
    def create_backup(self) -> str:
        """Create a JSON backup of all data."""
        backup_data = {
            'agent_memory': {},
            'workflow_states': {},
            'logs': self.load_logs(),
            'metrics': {},
            'user_feedback': self.load_user_feedback(),
            'backup_timestamp': datetime.now().isoformat()
        }
        
        # Get all agent memory
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT agent_id FROM agent_memory')
            agent_ids = [row[0] for row in cursor.fetchall()]
            
            for agent_id in agent_ids:
                backup_data['agent_memory'][agent_id] = {
                    'queries': self.load_agent_memory(agent_id, 'queries'),
                    'responses': self.load_agent_memory(agent_id, 'responses'),
                    'contexts': self.load_agent_memory(agent_id, 'contexts'),
                    'evaluations': self.load_agent_memory(agent_id, 'evaluations'),
                    'feedback': self.load_agent_memory(agent_id, 'feedback')
                }
        
        # Get all workflow states
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT workflow_id, state_data FROM workflow_state')
            for row in cursor.fetchall():
                backup_data['workflow_states'][row[0]] = json.loads(row[1])
        
        # Get all metrics
        metric_types = ['retrieval_latency', 'answer_quality', 'iteration_counts']
        for metric_type in metric_types:
            backup_data['metrics'][metric_type] = self.load_metrics(metric_type)
        
        # Save backup file
        backup_filename = f"agentic_rag_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        backup_path = os.path.join(self.backup_dir, backup_filename)
        
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        logger.info(f"Backup created: {backup_path}")
        return backup_path 
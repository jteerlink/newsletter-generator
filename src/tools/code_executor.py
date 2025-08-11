"""
Code Executor for Safe Testing of Generated Code Examples

This module provides secure execution and testing of generated Python code examples
to ensure they work correctly before inclusion in newsletters.
"""

import ast
import logging
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of code execution"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    SECURITY_VIOLATION = "security_violation"
    DEPENDENCY_ERROR = "dependency_error"


class SecurityLevel(Enum):
    """Security levels for code execution"""
    UNSAFE = "unsafe"          # No restrictions (not recommended)
    BASIC = "basic"            # Basic restrictions 
    SECURE = "secure"          # Secure sandbox environment
    ISOLATED = "isolated"      # Full isolation (subprocess)


@dataclass
class ExecutionResult:
    """Result of code execution"""
    status: ExecutionStatus
    stdout: str
    stderr: str
    execution_time: float
    return_value: Any = None
    exception: Optional[Exception] = None
    security_warnings: List[str] = None
    resource_usage: Dict[str, Any] = None


@dataclass
class ExecutionConfig:
    """Configuration for code execution"""
    timeout: float = 10.0
    max_memory_mb: int = 100
    security_level: SecurityLevel = SecurityLevel.SECURE
    allowed_imports: List[str] = None
    blocked_imports: List[str] = None
    capture_output: bool = True
    enable_network: bool = False


class SafeCodeExecutor:
    """Safe executor for Python code with security and resource controls"""
    
    def __init__(self, config: ExecutionConfig = None):
        self.config = config or ExecutionConfig()
        
        # Default safe imports for AI/ML code
        self.default_allowed_imports = [
            'numpy', 'np',
            'pandas', 'pd', 
            'matplotlib', 'plt',
            'seaborn', 'sns',
            'sklearn', 'scikit-learn',
            'torch', 'torchvision',
            'tensorflow', 'tf',
            'transformers',
            'scipy',
            'joblib',
            'pickle',
            'json',
            'csv',
            'math',
            'random',
            'datetime',
            'collections',
            'itertools',
            'functools',
            'typing',
            're',
            'os.path',  # Limited os access
            'sys.version',  # Limited sys access
        ]
        
        # Dangerous imports to block
        self.blocked_imports = [
            'subprocess',
            'os.system', 'os.exec', 'os.spawn',
            'sys.exit',
            'eval', 'exec',
            'compile',
            '__import__',
            'importlib',
            'socket',
            'urllib',
            'requests',  # Unless network is enabled
            'http',
            'ftplib',
            'smtplib',
            'multiprocessing',
            'threading',
            'ctypes',
            'gc',
        ]
        
        if not self.config.enable_network:
            self.blocked_imports.extend(['requests', 'urllib', 'http'])
    
    def execute(self, code: str, context: Dict[str, Any] = None) -> ExecutionResult:
        """
        Execute Python code safely with configured restrictions
        
        Args:
            code: Python code to execute
            context: Optional context variables to make available
            
        Returns:
            ExecutionResult with execution details
        """
        logger.info(f"Executing code with {self.config.security_level.value} security level")
        
        start_time = time.time()
        
        # Security validation
        security_check = self._validate_security(code)
        if not security_check.is_safe:
            return ExecutionResult(
                status=ExecutionStatus.SECURITY_VIOLATION,
                stdout="",
                stderr="Security validation failed",
                execution_time=time.time() - start_time,
                security_warnings=security_check.warnings
            )
        
        # Choose execution method based on security level
        if self.config.security_level == SecurityLevel.ISOLATED:
            return self._execute_isolated(code, context, start_time)
        elif self.config.security_level in [SecurityLevel.SECURE, SecurityLevel.BASIC]:
            return self._execute_sandboxed(code, context, start_time)
        else:  # UNSAFE
            return self._execute_direct(code, context, start_time)
    
    def test_code_example(self, code: str, expected_outputs: List[str] = None) -> ExecutionResult:
        """
        Test a code example and optionally validate expected outputs
        
        Args:
            code: Code to test
            expected_outputs: Optional list of expected output patterns
            
        Returns:
            ExecutionResult with test results
        """
        result = self.execute(code)
        
        if result.status == ExecutionStatus.SUCCESS and expected_outputs:
            # Validate expected outputs
            output_text = result.stdout.lower()
            for expected in expected_outputs:
                if expected.lower() not in output_text:
                    result.security_warnings = result.security_warnings or []
                    result.security_warnings.append(f"Expected output '{expected}' not found")
        
        return result
    
    def _validate_security(self, code: str) -> 'SecurityCheckResult':
        """Validate code for security issues"""
        warnings = []
        is_safe = True
        
        try:
            # Parse code to AST for analysis
            tree = ast.parse(code)
            
            # Check for dangerous patterns
            class SecurityVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.issues = []
                
                def visit_Import(self, node):
                    for alias in node.names:
                        if alias.name in self.get_blocked_imports():
                            self.issues.append(f"Blocked import: {alias.name}")
                
                def visit_ImportFrom(self, node):
                    if node.module in self.get_blocked_imports():
                        self.issues.append(f"Blocked import: {node.module}")
                
                def visit_Call(self, node):
                    # Check for dangerous function calls
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec', 'compile', '__import__']:
                            self.issues.append(f"Dangerous function call: {node.func.id}")
                    elif isinstance(node.func, ast.Attribute):
                        if (isinstance(node.func.value, ast.Name) and 
                            node.func.value.id == 'os' and 
                            node.func.attr in ['system', 'exec', 'spawn']):
                            self.issues.append(f"Dangerous os call: os.{node.func.attr}")
                    
                    self.generic_visit(node)
                
                def get_blocked_imports(self):
                    return self.get_outer_blocked_imports()
            
            # Monkey patch to access outer scope
            SecurityVisitor.get_outer_blocked_imports = lambda self: self.blocked_imports
            
            visitor = SecurityVisitor()
            visitor.visit(tree)
            
            if visitor.issues:
                warnings.extend(visitor.issues)
                if self.config.security_level in [SecurityLevel.SECURE, SecurityLevel.ISOLATED]:
                    is_safe = False
            
        except SyntaxError as e:
            warnings.append(f"Syntax error in security check: {e}")
            is_safe = False
        except Exception as e:
            warnings.append(f"Security validation error: {e}")
            # Allow execution but with warning
        
        return SecurityCheckResult(is_safe=is_safe, warnings=warnings)
    
    def _execute_isolated(self, code: str, context: Dict[str, Any], start_time: float) -> ExecutionResult:
        """Execute code in completely isolated subprocess"""
        try:
            # Create temporary file with code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Add context variables if provided
                if context:
                    for key, value in context.items():
                        f.write(f"{key} = {repr(value)}\n")
                f.write(code)
                temp_file = f.name
            
            # Execute in subprocess with restrictions
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                cwd=tempfile.gettempdir()  # Run in temp directory
            )
            
            # Clean up
            Path(temp_file).unlink(missing_ok=True)
            
            status = ExecutionStatus.SUCCESS if result.returncode == 0 else ExecutionStatus.FAILURE
            
            return ExecutionResult(
                status=status,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=time.time() - start_time
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                stdout="",
                stderr="Code execution timed out",
                execution_time=self.config.timeout
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILURE,
                stdout="",
                stderr=str(e),
                execution_time=time.time() - start_time,
                exception=e
            )
    
    def _execute_sandboxed(self, code: str, context: Dict[str, Any], start_time: float) -> ExecutionResult:
        """Execute code in sandboxed environment with restricted globals"""
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        try:
            # Create restricted globals
            safe_globals = self._create_safe_globals()
            
            # Add context variables
            if context:
                safe_globals.update(context)
            
            # Execute with timeout and output capture
            with self._timeout_context(self.config.timeout):
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    result = eval(compile(code, '<string>', 'exec'), safe_globals)
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                execution_time=time.time() - start_time,
                return_value=result
            )
            
        except TimeoutError:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                execution_time=self.config.timeout
            )
        except ImportError as e:
            return ExecutionResult(
                status=ExecutionStatus.DEPENDENCY_ERROR,
                stdout=stdout_capture.getvalue(),
                stderr=f"Import error: {str(e)}",
                execution_time=time.time() - start_time,
                exception=e
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILURE,
                stdout=stdout_capture.getvalue(),
                stderr=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time,
                exception=e
            )
    
    def _execute_direct(self, code: str, context: Dict[str, Any], start_time: float) -> ExecutionResult:
        """Execute code directly with minimal restrictions (UNSAFE)"""
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        try:
            # Create globals with context
            execution_globals = globals().copy()
            if context:
                execution_globals.update(context)
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                result = exec(code, execution_globals)
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                execution_time=time.time() - start_time,
                return_value=result
            )
            
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILURE,
                stdout=stdout_capture.getvalue(),
                stderr=f"Execution error: {str(e)}",
                execution_time=time.time() - start_time,
                exception=e
            )
    
    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create a restricted global namespace for safe execution"""
        safe_builtins = {
            # Safe built-in functions
            'abs', 'all', 'any', 'bin', 'bool', 'bytearray', 'bytes',
            'chr', 'dict', 'divmod', 'enumerate', 'filter', 'float',
            'format', 'frozenset', 'hasattr', 'hash', 'hex', 'id',
            'int', 'isinstance', 'issubclass', 'iter', 'len', 'list',
            'map', 'max', 'min', 'next', 'oct', 'ord', 'pow', 'print',
            'range', 'repr', 'reversed', 'round', 'set', 'slice',
            'sorted', 'str', 'sum', 'tuple', 'type', 'zip',
        }
        
        # Create restricted builtins
        restricted_builtins = {}
        import builtins
        for name in safe_builtins:
            if hasattr(builtins, name):
                restricted_builtins[name] = getattr(builtins, name)
        
        # Safe imports
        safe_modules = {}
        
        # Try to import safe modules
        safe_import_names = self.config.allowed_imports or self.default_allowed_imports
        
        for module_name in safe_import_names:
            try:
                if module_name in ['np', 'pd', 'plt', 'sns', 'tf']:
                    # Handle common aliases
                    real_names = {
                        'np': 'numpy',
                        'pd': 'pandas', 
                        'plt': 'matplotlib.pyplot',
                        'sns': 'seaborn',
                        'tf': 'tensorflow'
                    }
                    real_module = __import__(real_names.get(module_name, module_name))
                    safe_modules[module_name] = real_module
                else:
                    safe_modules[module_name] = __import__(module_name)
            except ImportError:
                logger.debug(f"Module {module_name} not available for safe execution")
        
        return {
            '__builtins__': restricted_builtins,
            **safe_modules
        }
    
    @contextmanager
    def _timeout_context(self, timeout: float):
        """Context manager for execution timeout"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")
        
        # Set up timeout (Unix only)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # Fallback for Windows - no timeout
            yield


@dataclass
class SecurityCheckResult:
    """Result of security validation"""
    is_safe: bool
    warnings: List[str]


class CodeExampleTester:
    """High-level tester for code examples with common AI/ML patterns"""
    
    def __init__(self, executor: SafeCodeExecutor = None):
        self.executor = executor or SafeCodeExecutor()
    
    def test_pytorch_example(self, code: str) -> ExecutionResult:
        """Test PyTorch code example"""
        # Add common PyTorch test patterns
        test_code = f"""
import torch
import torch.nn as nn
try:
    {code}
    print("PyTorch example executed successfully")
except Exception as e:
    print(f"PyTorch example failed: {{e}}")
    raise
"""
        return self.executor.execute(test_code)
    
    def test_tensorflow_example(self, code: str) -> ExecutionResult:
        """Test TensorFlow code example"""
        test_code = f"""
import tensorflow as tf
try:
    {code}
    print("TensorFlow example executed successfully")
except Exception as e:
    print(f"TensorFlow example failed: {{e}}")
    raise
"""
        return self.executor.execute(test_code)
    
    def test_sklearn_example(self, code: str) -> ExecutionResult:
        """Test scikit-learn code example"""
        test_code = f"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
try:
    {code}
    print("Scikit-learn example executed successfully")
except Exception as e:
    print(f"Scikit-learn example failed: {{e}}")
    raise
"""
        return self.executor.execute(test_code)
    
    def test_pandas_example(self, code: str) -> ExecutionResult:
        """Test pandas code example"""
        test_code = f"""
import pandas as pd
import numpy as np
try:
    {code}
    print("Pandas example executed successfully")
except Exception as e:
    print(f"Pandas example failed: {{e}}")
    raise
"""
        return self.executor.execute(test_code)
    
    def test_basic_python(self, code: str) -> ExecutionResult:
        """Test basic Python code"""
        test_code = f"""
try:
    {code}
    print("Python code executed successfully")
except Exception as e:
    print(f"Python code failed: {{e}}")
    raise
"""
        return self.executor.execute(test_code)
    
    def generate_test_report(self, results: List[Tuple[str, ExecutionResult]]) -> str:
        """Generate a comprehensive test report"""
        report_lines = []
        
        report_lines.append("=" * 60)
        report_lines.append("CODE EXECUTION TEST REPORT")
        report_lines.append("=" * 60)
        report_lines.append()
        
        total_tests = len(results)
        passed_tests = sum(1 for _, result in results if result.status == ExecutionStatus.SUCCESS)
        
        report_lines.append(f"Total Tests: {total_tests}")
        report_lines.append(f"Passed: {passed_tests}")
        report_lines.append(f"Failed: {total_tests - passed_tests}")
        report_lines.append(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        report_lines.append()
        
        # Individual test results
        for i, (test_name, result) in enumerate(results, 1):
            status_icon = "✅" if result.status == ExecutionStatus.SUCCESS else "❌"
            report_lines.append(f"{i:2d}. {status_icon} {test_name}")
            report_lines.append(f"    Status: {result.status.value}")
            report_lines.append(f"    Execution Time: {result.execution_time:.3f}s")
            
            if result.stdout:
                report_lines.append(f"    Output: {result.stdout.strip()}")
            
            if result.stderr:
                report_lines.append(f"    Errors: {result.stderr.strip()}")
            
            if result.security_warnings:
                report_lines.append(f"    Warnings: {', '.join(result.security_warnings)}")
            
            report_lines.append()
        
        return '\n'.join(report_lines)


# Convenience function for quick testing
def test_code_example(code: str, framework: str = "python", 
                     security_level: SecurityLevel = SecurityLevel.SECURE,
                     timeout: float = 10.0) -> ExecutionResult:
    """
    Quick function to test a code example
    
    Args:
        code: Code to test
        framework: Framework type (python, pytorch, tensorflow, sklearn, pandas)
        security_level: Security level for execution
        timeout: Execution timeout in seconds
        
    Returns:
        ExecutionResult
    """
    config = ExecutionConfig(
        timeout=timeout,
        security_level=security_level
    )
    
    executor = SafeCodeExecutor(config)
    tester = CodeExampleTester(executor)
    
    if framework == "pytorch":
        return tester.test_pytorch_example(code)
    elif framework == "tensorflow":
        return tester.test_tensorflow_example(code)
    elif framework == "sklearn":
        return tester.test_sklearn_example(code)
    elif framework == "pandas":
        return tester.test_pandas_example(code)
    else:
        return tester.test_basic_python(code)

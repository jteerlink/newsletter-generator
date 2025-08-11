"""
Syntax Validator for Code Examples in Newsletter Generation

This module provides comprehensive syntax validation, style checking, and best practices
analysis for generated code examples to ensure high-quality technical content.
"""

import ast
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Levels of validation strictness"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"


class IssueType(Enum):
    """Types of code issues that can be detected"""
    SYNTAX_ERROR = "syntax_error"
    STYLE_VIOLATION = "style_violation"
    BEST_PRACTICE = "best_practice"
    SECURITY_CONCERN = "security_concern"
    PERFORMANCE_ISSUE = "performance_issue"
    DOCUMENTATION = "documentation"


class IssueSeverity(Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in code"""
    line_number: int
    column: int
    issue_type: IssueType
    severity: IssueSeverity
    message: str
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None


@dataclass
class ValidationResult:
    """Complete validation result for a code example"""
    is_valid: bool
    syntax_score: float  # 0.0 to 1.0
    style_score: float   # 0.0 to 1.0
    overall_score: float # 0.0 to 1.0
    issues: List[ValidationIssue]
    has_imports: bool
    has_docstrings: bool
    has_comments: bool
    line_count: int
    complexity_score: float


class SyntaxValidator:
    """Advanced syntax and style validator for Python code examples"""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.style_patterns = self._initialize_style_patterns()
        self.best_practice_patterns = self._initialize_best_practice_patterns()
        self.security_patterns = self._initialize_security_patterns()
    
    def validate(self, code: str, filename: str = "<string>") -> ValidationResult:
        """
        Perform comprehensive validation of Python code
        
        Args:
            code: The Python code to validate
            filename: Optional filename for context
            
        Returns:
            ValidationResult with detailed analysis
        """
        logger.info(f"Validating code with {self.validation_level.value} level")
        
        issues = []
        
        # Basic syntax validation
        syntax_valid, syntax_issues = self._validate_syntax(code, filename)
        issues.extend(syntax_issues)
        
        if syntax_valid:
            # Style validation
            style_issues = self._validate_style(code)
            issues.extend(style_issues)
            
            # Best practices validation
            if self.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
                best_practice_issues = self._validate_best_practices(code)
                issues.extend(best_practice_issues)
            
            # Security validation
            if self.validation_level == ValidationLevel.STRICT:
                security_issues = self._validate_security(code)
                issues.extend(security_issues)
        
        # Calculate scores and metrics
        result = self._calculate_scores(code, issues, syntax_valid)
        result.issues = issues
        
        return result
    
    def _validate_syntax(self, code: str, filename: str) -> Tuple[bool, List[ValidationIssue]]:
        """Validate Python syntax using AST parsing"""
        issues = []
        
        try:
            # Parse the code into an AST
            tree = ast.parse(code, filename=filename)
            
            # Additional AST-based checks
            issues.extend(self._check_ast_patterns(tree, code))
            
            return True, issues
            
        except SyntaxError as e:
            # Handle syntax errors
            issue = ValidationIssue(
                line_number=e.lineno or 1,
                column=e.offset or 0,
                issue_type=IssueType.SYNTAX_ERROR,
                severity=IssueSeverity.CRITICAL,
                message=f"Syntax error: {e.msg}",
                suggestion="Fix the syntax error before proceeding",
                code_snippet=self._get_code_snippet(code, e.lineno or 1)
            )
            issues.append(issue)
            return False, issues
            
        except Exception as e:
            # Handle other parsing errors
            issue = ValidationIssue(
                line_number=1,
                column=0,
                issue_type=IssueType.SYNTAX_ERROR,
                severity=IssueSeverity.CRITICAL,
                message=f"Code parsing error: {str(e)}",
                suggestion="Check code structure and syntax"
            )
            issues.append(issue)
            return False, issues
    
    def _validate_style(self, code: str) -> List[ValidationIssue]:
        """Validate code style using PEP 8 guidelines"""
        issues = []
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Check line length
            if len(line) > 88:  # Relaxed from 79 for readability
                issues.append(ValidationIssue(
                    line_number=line_num,
                    column=88,
                    issue_type=IssueType.STYLE_VIOLATION,
                    severity=IssueSeverity.MEDIUM,
                    message="Line too long (>88 characters)",
                    suggestion="Break long lines for better readability",
                    code_snippet=line[:50] + "..." if len(line) > 50 else line
                ))
            
            # Check for style patterns
            for pattern, issue_info in self.style_patterns.items():
                if re.search(pattern, line):
                    issues.append(ValidationIssue(
                        line_number=line_num,
                        column=0,
                        issue_type=IssueType.STYLE_VIOLATION,
                        severity=issue_info['severity'],
                        message=issue_info['message'],
                        suggestion=issue_info['suggestion'],
                        code_snippet=line.strip()
                    ))
        
        return issues
    
    def _validate_best_practices(self, code: str) -> List[ValidationIssue]:
        """Validate against Python best practices"""
        issues = []
        lines = code.split('\n')
        
        # Check for docstrings
        if not re.search(r'""".*?"""', code, re.DOTALL) and len(lines) > 10:
            issues.append(ValidationIssue(
                line_number=1,
                column=0,
                issue_type=IssueType.BEST_PRACTICE,
                severity=IssueSeverity.LOW,
                message="Consider adding docstrings for better documentation",
                suggestion="Add module and function docstrings"
            ))
        
        # Check for best practice patterns
        for line_num, line in enumerate(lines, 1):
            for pattern, issue_info in self.best_practice_patterns.items():
                if re.search(pattern, line):
                    issues.append(ValidationIssue(
                        line_number=line_num,
                        column=0,
                        issue_type=IssueType.BEST_PRACTICE,
                        severity=issue_info['severity'],
                        message=issue_info['message'],
                        suggestion=issue_info['suggestion'],
                        code_snippet=line.strip()
                    ))
        
        return issues
    
    def _validate_security(self, code: str) -> List[ValidationIssue]:
        """Validate for security concerns"""
        issues = []
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for pattern, issue_info in self.security_patterns.items():
                if re.search(pattern, line):
                    issues.append(ValidationIssue(
                        line_number=line_num,
                        column=0,
                        issue_type=IssueType.SECURITY_CONCERN,
                        severity=issue_info['severity'],
                        message=issue_info['message'],
                        suggestion=issue_info['suggestion'],
                        code_snippet=line.strip()
                    ))
        
        return issues
    
    def _check_ast_patterns(self, tree: ast.AST, code: str) -> List[ValidationIssue]:
        """Check for patterns using AST analysis"""
        issues = []
        
        class ASTChecker(ast.NodeVisitor):
            def __init__(self, validator):
                self.validator = validator
                self.issues = []
            
            def visit_FunctionDef(self, node):
                # Check for functions without docstrings
                if (not ast.get_docstring(node) and 
                    self.validator.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]):
                    self.issues.append(ValidationIssue(
                        line_number=node.lineno,
                        column=node.col_offset,
                        issue_type=IssueType.DOCUMENTATION,
                        severity=IssueSeverity.LOW,
                        message=f"Function '{node.name}' lacks docstring",
                        suggestion="Add a docstring describing the function's purpose"
                    ))
                
                # Check for too many arguments
                if len(node.args.args) > 5:
                    self.issues.append(ValidationIssue(
                        line_number=node.lineno,
                        column=node.col_offset,
                        issue_type=IssueType.BEST_PRACTICE,
                        severity=IssueSeverity.MEDIUM,
                        message=f"Function '{node.name}' has too many parameters ({len(node.args.args)})",
                        suggestion="Consider using a class or dictionary for parameters"
                    ))
                
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                # Check for classes without docstrings
                if (not ast.get_docstring(node) and 
                    self.validator.validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]):
                    self.issues.append(ValidationIssue(
                        line_number=node.lineno,
                        column=node.col_offset,
                        issue_type=IssueType.DOCUMENTATION,
                        severity=IssueSeverity.LOW,
                        message=f"Class '{node.name}' lacks docstring",
                        suggestion="Add a docstring describing the class's purpose"
                    ))
                
                self.generic_visit(node)
        
        checker = ASTChecker(self)
        checker.visit(tree)
        issues.extend(checker.issues)
        
        return issues
    
    def _calculate_scores(self, code: str, issues: List[ValidationIssue], syntax_valid: bool) -> ValidationResult:
        """Calculate validation scores and metrics"""
        lines = code.split('\n')
        line_count = len([line for line in lines if line.strip()])
        
        # Syntax score
        syntax_score = 1.0 if syntax_valid else 0.0
        
        # Style score based on issues
        style_issues = [i for i in issues if i.issue_type == IssueType.STYLE_VIOLATION]
        max_style_deductions = line_count * 0.1  # Max 10% deduction per line
        style_deductions = min(len(style_issues) * 0.05, max_style_deductions)
        style_score = max(0.0, 1.0 - style_deductions)
        
        # Overall score
        critical_issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
        high_issues = [i for i in issues if i.severity == IssueSeverity.HIGH]
        medium_issues = [i for i in issues if i.severity == IssueSeverity.MEDIUM]
        
        overall_score = syntax_score * 0.5 + style_score * 0.3
        overall_score -= len(critical_issues) * 0.2
        overall_score -= len(high_issues) * 0.1  
        overall_score -= len(medium_issues) * 0.05
        overall_score = max(0.0, min(1.0, overall_score))
        
        # Check for code features
        has_imports = 'import ' in code
        has_docstrings = '"""' in code or "'''" in code
        has_comments = '#' in code
        
        # Calculate complexity (simplified)
        complexity_indicators = ['if ', 'for ', 'while ', 'try:', 'def ', 'class ']
        complexity_score = sum(code.count(indicator) for indicator in complexity_indicators) / max(line_count, 1)
        
        return ValidationResult(
            is_valid=syntax_valid and len(critical_issues) == 0,
            syntax_score=syntax_score,
            style_score=style_score,
            overall_score=overall_score,
            issues=[],  # Will be set by caller
            has_imports=has_imports,
            has_docstrings=has_docstrings,
            has_comments=has_comments,
            line_count=line_count,
            complexity_score=complexity_score
        )
    
    def _get_code_snippet(self, code: str, line_number: int, context: int = 2) -> str:
        """Get a code snippet around a specific line"""
        lines = code.split('\n')
        start = max(0, line_number - context - 1)
        end = min(len(lines), line_number + context)
        
        snippet_lines = []
        for i in range(start, end):
            marker = ">>>" if i == line_number - 1 else "   "
            snippet_lines.append(f"{marker} {i+1:3d}: {lines[i]}")
        
        return '\n'.join(snippet_lines)
    
    def _initialize_style_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize style validation patterns"""
        return {
            r'\s+$': {
                'severity': IssueSeverity.LOW,
                'message': 'Trailing whitespace',
                'suggestion': 'Remove trailing whitespace'
            },
            r'\t': {
                'severity': IssueSeverity.MEDIUM,
                'message': 'Tab character used instead of spaces',
                'suggestion': 'Use 4 spaces for indentation'
            },
            r';\s*$': {
                'severity': IssueSeverity.LOW,
                'message': 'Unnecessary semicolon',
                'suggestion': 'Remove semicolon (not needed in Python)'
            },
            r'==\s*True|!=\s*False': {
                'severity': IssueSeverity.MEDIUM,
                'message': 'Explicit comparison with True/False',
                'suggestion': 'Use implicit boolean evaluation'
            }
        }
    
    def _initialize_best_practice_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize best practice validation patterns"""
        return {
            r'except\s*:': {
                'severity': IssueSeverity.HIGH,
                'message': 'Bare except clause',
                'suggestion': 'Specify exception types to catch'
            },
            r'print\s*\(': {
                'severity': IssueSeverity.LOW,
                'message': 'Print statement in code',
                'suggestion': 'Consider using logging instead of print'
            },
            r'eval\s*\(': {
                'severity': IssueSeverity.HIGH,
                'message': 'Use of eval() function',
                'suggestion': 'Avoid eval() for security reasons'
            },
            r'exec\s*\(': {
                'severity': IssueSeverity.HIGH,
                'message': 'Use of exec() function', 
                'suggestion': 'Avoid exec() for security reasons'
            },
            r'import\s+\*': {
                'severity': IssueSeverity.MEDIUM,
                'message': 'Wildcard import',
                'suggestion': 'Import specific names to avoid namespace pollution'
            }
        }
    
    def _initialize_security_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize security validation patterns"""
        return {
            r'subprocess\.call|os\.system': {
                'severity': IssueSeverity.HIGH,
                'message': 'Potentially unsafe system call',
                'suggestion': 'Validate input and consider safer alternatives'
            },
            r'pickle\.loads?|pickle\.dumps?': {
                'severity': IssueSeverity.MEDIUM,
                'message': 'Pickle usage detected',
                'suggestion': 'Be cautious with pickle - only use with trusted data'
            },
            r'random\.random\(\)|random\.choice': {
                'severity': IssueSeverity.LOW,
                'message': 'Non-cryptographic random function',
                'suggestion': 'Use secrets module for cryptographic applications'
            }
        }
    
    def format_validation_report(self, result: ValidationResult, include_suggestions: bool = True) -> str:
        """Format a comprehensive validation report"""
        report_lines = []
        
        # Header
        report_lines.append("=" * 60)
        report_lines.append("CODE VALIDATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append()
        
        # Overall status
        status = "✅ PASSED" if result.is_valid else "❌ FAILED"
        report_lines.append(f"Status: {status}")
        report_lines.append(f"Overall Score: {result.overall_score:.2f}/1.00")
        report_lines.append()
        
        # Detailed scores
        report_lines.append("Detailed Scores:")
        report_lines.append(f"  Syntax:     {result.syntax_score:.2f}/1.00")
        report_lines.append(f"  Style:      {result.style_score:.2f}/1.00")
        report_lines.append(f"  Complexity: {result.complexity_score:.2f}")
        report_lines.append()
        
        # Code metrics
        report_lines.append("Code Metrics:")
        report_lines.append(f"  Lines of code:  {result.line_count}")
        report_lines.append(f"  Has imports:    {'✅' if result.has_imports else '❌'}")
        report_lines.append(f"  Has docstrings: {'✅' if result.has_docstrings else '❌'}")
        report_lines.append(f"  Has comments:   {'✅' if result.has_comments else '❌'}")
        report_lines.append()
        
        # Issues summary
        if result.issues:
            severity_counts = {}
            for issue in result.issues:
                severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
            
            report_lines.append(f"Issues Found ({len(result.issues)} total):")
            for severity in IssueSeverity:
                count = severity_counts.get(severity, 0)
                if count > 0:
                    report_lines.append(f"  {severity.value.capitalize()}: {count}")
            report_lines.append()
            
            # Detailed issues
            report_lines.append("Detailed Issues:")
            for i, issue in enumerate(result.issues, 1):
                severity_icon = {
                    IssueSeverity.CRITICAL: "🔴",
                    IssueSeverity.HIGH: "🟠", 
                    IssueSeverity.MEDIUM: "🟡",
                    IssueSeverity.LOW: "🟢",
                    IssueSeverity.INFO: "ℹ️"
                }.get(issue.severity, "❓")
                
                report_lines.append(f"{i:3d}. {severity_icon} Line {issue.line_number}, Col {issue.column}")
                report_lines.append(f"     Type: {issue.issue_type.value}")
                report_lines.append(f"     Message: {issue.message}")
                
                if issue.code_snippet:
                    report_lines.append(f"     Code: {issue.code_snippet}")
                
                if include_suggestions and issue.suggestion:
                    report_lines.append(f"     Suggestion: {issue.suggestion}")
                
                report_lines.append()
        else:
            report_lines.append("✅ No issues found!")
            report_lines.append()
        
        return '\n'.join(report_lines)

    def validate_with_external_tools(self, code: str) -> ValidationResult:
        """
        Validate code using external tools like flake8, pylint when available
        
        Args:
            code: Python code to validate
            
        Returns:
            Enhanced validation result with external tool findings
        """
        # Start with basic validation
        result = self.validate(code)
        
        # Try to use external tools if available
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Try flake8
            try:
                flake8_result = subprocess.run(
                    ['flake8', '--max-line-length=88', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if flake8_result.stdout:
                    result.issues.extend(self._parse_flake8_output(flake8_result.stdout))
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug("flake8 not available or failed")
            
            # Clean up
            Path(temp_file).unlink(missing_ok=True)
            
        except Exception as e:
            logger.warning(f"External validation failed: {e}")
        
        return result
    
    def _parse_flake8_output(self, output: str) -> List[ValidationIssue]:
        """Parse flake8 output into ValidationIssue objects"""
        issues = []
        
        for line in output.strip().split('\n'):
            if not line:
                continue
                
            # Parse flake8 format: filename:line:col: code message
            parts = line.split(':', 3)
            if len(parts) >= 4:
                try:
                    line_num = int(parts[1])
                    col_num = int(parts[2])
                    message = parts[3].strip()
                    
                    # Determine severity based on flake8 code
                    severity = IssueSeverity.MEDIUM
                    if message.startswith('E9') or message.startswith('F'):
                        severity = IssueSeverity.HIGH
                    elif message.startswith('W'):
                        severity = IssueSeverity.LOW
                    
                    issues.append(ValidationIssue(
                        line_number=line_num,
                        column=col_num,
                        issue_type=IssueType.STYLE_VIOLATION,
                        severity=severity,
                        message=f"flake8: {message}",
                        suggestion="Fix according to PEP 8 guidelines"
                    ))
                except ValueError:
                    continue
        
        return issues

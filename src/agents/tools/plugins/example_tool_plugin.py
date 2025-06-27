"""
Example Tool Plugin
Demonstrates how to create a custom tool plugin for the extensible tool system.
"""

from typing import Dict, Any, Set
from agents.base.tool_registry import ToolPlugin, ToolMetadata, ToolCapability


class ExampleDataProcessor:
    """
    Example data processing tool that demonstrates plugin functionality.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.processing_mode = config.get('processing_mode', 'standard')
        self.batch_size = config.get('batch_size', 100)
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the data processing tool.
        """
        data = inputs.get('data', [])
        operation = inputs.get('operation', 'filter')
        
        if operation == 'filter':
            result = self._filter_data(data, inputs.get('filter_criteria', {}))
        elif operation == 'transform':
            result = self._transform_data(data, inputs.get('transform_rules', {}))
        elif operation == 'aggregate':
            result = self._aggregate_data(data, inputs.get('aggregation_fields', []))
        else:
            result = {'error': f'Unknown operation: {operation}'}
        
        return {
            'processed_data': result,
            'processing_mode': self.processing_mode,
            'batch_size': self.batch_size,
            'operation': operation,
            'input_count': len(data),
            'output_count': len(result) if isinstance(result, list) else 1
        }
    
    def _filter_data(self, data: list, criteria: Dict[str, Any]) -> list:
        """Filter data based on criteria."""
        filtered = []
        for item in data:
            if isinstance(item, dict):
                # Simple filtering logic
                include = True
                for key, value in criteria.items():
                    if key in item and item[key] != value:
                        include = False
                        break
                if include:
                    filtered.append(item)
        return filtered
    
    def _transform_data(self, data: list, rules: Dict[str, Any]) -> list:
        """Transform data based on rules."""
        transformed = []
        for item in data:
            if isinstance(item, dict):
                new_item = item.copy()
                for field, transformation in rules.items():
                    if field in new_item:
                        if transformation == 'uppercase':
                            new_item[field] = str(new_item[field]).upper()
                        elif transformation == 'lowercase':
                            new_item[field] = str(new_item[field]).lower()
                        elif transformation == 'capitalize':
                            new_item[field] = str(new_item[field]).capitalize()
                transformed.append(new_item)
        return transformed
    
    def _aggregate_data(self, data: list, fields: list) -> Dict[str, Any]:
        """Aggregate data by specified fields."""
        if not fields:
            return {'count': len(data)}
        
        aggregated = {}
        for item in data:
            if isinstance(item, dict):
                for field in fields:
                    if field in item:
                        if field not in aggregated:
                            aggregated[field] = []
                        aggregated[field].append(item[field])
        
        # Calculate statistics
        result = {}
        for field, values in aggregated.items():
            if values:
                result[f'{field}_count'] = len(values)
                result[f'{field}_unique'] = len(set(values))
                if all(isinstance(v, (int, float)) for v in values):
                    result[f'{field}_sum'] = sum(values)
                    result[f'{field}_avg'] = sum(values) / len(values)
        
        return result


class ExampleToolPlugin(ToolPlugin):
    """
    Example tool plugin that demonstrates the plugin system.
    """
    
    def get_metadata(self) -> ToolMetadata:
        """Return tool metadata."""
        return ToolMetadata(
            name="example_data_processor",
            description="An example data processing tool for demonstration purposes",
            capabilities={
                ToolCapability.DATA_EXTRACTION,
                ToolCapability.DOCUMENT_PROCESSING,
                ToolCapability.STATISTICAL_ANALYSIS
            },
            version="1.0.0",
            author="Example Author",
            dependencies=["agents.base.tool_registry"],
            config_schema={
                "processing_mode": {
                    "type": "string",
                    "enum": ["standard", "fast", "thorough"],
                    "default": "standard",
                    "description": "Processing mode for the tool"
                },
                "batch_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 100,
                    "description": "Batch size for processing"
                }
            },
            input_schema={
                "data": {
                    "type": "array",
                    "description": "Data to be processed"
                },
                "operation": {
                    "type": "string",
                    "enum": ["filter", "transform", "aggregate"],
                    "description": "Operation to perform"
                },
                "filter_criteria": {
                    "type": "object",
                    "description": "Criteria for filtering (optional)"
                },
                "transform_rules": {
                    "type": "object",
                    "description": "Rules for transformation (optional)"
                },
                "aggregation_fields": {
                    "type": "array",
                    "description": "Fields to aggregate (optional)"
                }
            },
            output_schema={
                "processed_data": {
                    "type": "any",
                    "description": "Processed data result"
                },
                "processing_mode": {
                    "type": "string",
                    "description": "Mode used for processing"
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Batch size used"
                },
                "operation": {
                    "type": "string",
                    "description": "Operation performed"
                },
                "input_count": {
                    "type": "integer",
                    "description": "Number of input items"
                },
                "output_count": {
                    "type": "integer",
                    "description": "Number of output items"
                }
            }
        )
    
    def create_tool(self, config: Dict[str, Any] = None) -> Any:
        """Create and return a tool instance."""
        if config is None:
            config = {}
        
        return ExampleDataProcessor(config=config)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate tool configuration."""
        if not config:
            return True
        
        # Validate processing mode
        processing_mode = config.get('processing_mode')
        valid_modes = ['standard', 'fast', 'thorough']
        if processing_mode and processing_mode not in valid_modes:
            return False
        
        # Validate batch size
        batch_size = config.get('batch_size')
        if batch_size is not None:
            if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 1000:
                return False
        
        return True
    
    def execute(self, tool_instance: Any, inputs: Dict[str, Any]) -> Any:
        """Execute the tool with given inputs."""
        return tool_instance.execute(inputs) 
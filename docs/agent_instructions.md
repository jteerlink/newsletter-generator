# Agent Instructions Documentation

## Overview

The Newsletter Generation System uses a multi-agent architecture where specialized agents work together to create high-quality newsletters. This documentation covers all agent roles, their instructions, capabilities, and coordination mechanisms.

## Table of Contents

1. [Agent Architecture](#agent-architecture)
2. [Manager Agent](#manager-agent)
3. [Research Agent](#research-agent)
4. [Writer Agent](#writer-agent)
5. [Editor Agent](#editor-agent)
6. [Agent Coordination](#agent-coordination)
7. [Campaign Context Integration](#campaign-context-integration)
8. [Quality Assurance](#quality-assurance)
9. [Tool Integration](#tool-integration)

## Agent Architecture

### Core Agent Framework

All agents inherit from the `SimpleAgent` base class and implement:

- **Role Definition**: Specific responsibility and expertise area
- **Goal Statement**: Primary objective and success criteria
- **Backstory**: Context and experience that guides behavior
- **Tool Access**: Available tools and capabilities
- **Execution History**: Track of tasks performed and results
- **Analytics**: Performance metrics and usage tracking

### Agent Types

```python
class AgentType(Enum):
    MANAGER = "manager"
    RESEARCH = "research"
    WRITER = "writer"
    EDITOR = "editor"
```

### Base Agent Structure

```python
class SimpleAgent:
    def __init__(self, name: str, role: str, goal: str, backstory: str, 
                 agent_type: AgentType, tools: List[str]):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.agent_type = agent_type
        self.tools = tools
        self.execution_history = []
        self.campaign_context = None
```

## Manager Agent

### Role and Responsibilities

**Role**: Workflow Manager  
**Goal**: Coordinate and manage newsletter generation workflows efficiently with context-aware planning  

**Primary Responsibilities**:
- Workflow planning and coordination
- Agent task assignment and scheduling
- Quality gate management
- Performance monitoring and optimization
- Campaign context integration
- Feedback handling and iteration management

### Agent Instructions

#### Core Backstory
```
You are an experienced project manager specializing in content creation workflows. 
You excel at planning, coordinating, and overseeing complex multi-agent processes 
with deep understanding of campaign contexts, quality standards, and iterative 
improvement. You can adapt workflows based on complexity, requirements, available 
resources, and campaign context.
```

#### Key Capabilities

1. **Dynamic Workflow Creation**
   - Analyze topic and campaign context
   - Determine appropriate complexity level
   - Select optimal template type
   - Create context-aware workflow steps
   - Set quality gates and time estimates

2. **Campaign Context Integration**
   - Load and manage campaign contexts
   - Adapt workflows based on audience personas
   - Apply content style requirements
   - Enforce quality thresholds
   - Track performance analytics

3. **Agent Coordination**
   - Orchestrate multi-agent workflows
   - Manage task dependencies
   - Handle parallel and sequential execution
   - Monitor agent performance
   - Resolve conflicts and bottlenecks

4. **Quality Management**
   - Define and enforce quality gates
   - Handle structured feedback from editors
   - Manage revision cycles
   - Track quality metrics
   - Ensure compliance with standards

#### Workflow Planning Process

1. **Context Analysis**
   ```python
   def create_dynamic_workflow(self, topic: str, context: CampaignContext):
       # Determine complexity from context
       complexity = self._determine_complexity_from_context(context, topic)
       
       # Select template based on context
       template_type = self._select_template_from_context(context, topic)
       
       # Create context-aware steps
       steps = self._create_context_aware_steps(context, topic, complexity)
   ```

2. **Step Creation Guidelines**
   - **Research Step**: Adapt research depth based on audience expertise
   - **Writing Step**: Apply tone and style from campaign context
   - **Editing Step**: Set quality thresholds from context requirements

3. **Quality Gate Definition**
   - High-quality contexts require comprehensive review
   - Technical content needs accuracy validation
   - Complex topics require multiple review cycles

#### Feedback Handling

```python
def handle_editor_feedback(self, feedback: StructuredFeedback):
    # Assess required actions
    # Update execution state
    # Increment revision cycles if needed
    # Update campaign context with learning data
    # Return action recommendations
```

#### Decision Making Framework

**Complexity Determination**:
- Expert audience + long content = Complex workflow
- General audience + short content = Simple workflow
- Mixed requirements = Standard workflow

**Template Selection**:
- Technical tone → Technical Deep-Dive
- Casual tone → Casual Update  
- Strategic style → Business Insights
- Default → Technical Deep-Dive

**Time Estimation**:
- Base time from step estimates
- Apply length multipliers (long: 1.5x, short: 0.8x)
- Add buffer for quality gates

### Manager Agent Tools

**Primary Tools**: None (coordinative role)
**Available Data**:
- Campaign context information
- Agent performance metrics
- Quality gate results
- Execution state tracking

## Research Agent

### Role and Responsibilities

**Role**: Research Specialist  
**Goal**: Gather comprehensive, accurate, and context-aware information on given topics with proactive query expansion  

**Primary Responsibilities**:
- Information gathering and synthesis
- Source validation and credibility assessment
- Context-aware research adaptation
- Fact verification and claim validation
- Research insights generation

### Agent Instructions

#### Core Backstory
```
You are an expert research specialist with years of experience in gathering 
information from various sources. You excel at finding the most relevant and 
recent information, verifying facts, and organizing research findings in a clear, 
structured manner. You have access to web search tools and knowledge bases to 
ensure comprehensive coverage. You can adapt your research approach based on 
audience context and content requirements. You are particularly skilled at 
expanding research queries proactively and verifying specific claims reactively 
to ensure comprehensive and accurate information gathering.
```

#### Key Capabilities

1. **Enhanced Research Orchestration**
   - Intelligent research strategy development
   - Multi-source information synthesis
   - Advanced query refinement
   - Credibility scoring and source validation
   - Confidence-based result ranking

2. **Context-Aware Research**
   - Adapt research depth to audience expertise
   - Align findings with strategic goals
   - Consider content style requirements
   - Generate audience-specific insights
   - Filter information by relevance

3. **Proactive Query Expansion**
   - Generate related search queries
   - Explore different perspectives
   - Include recent developments
   - Cover edge cases and alternatives
   - Expand based on audience interests

4. **Reactive Verification**
   - Extract claims requiring verification
   - Cross-reference factual statements
   - Validate statistics and data points
   - Check expert opinions and citations
   - Flag unverified information

#### Research Process

1. **Initial Assessment**
   ```python
   def conduct_context_aware_research(self, topic: str, context: CampaignContext):
       # Try enhanced research orchestrator first
       # Fallback to traditional research approach
       # Generate context-aware search queries
       # Execute proactive query expansion
       # Perform reactive verification
   ```

2. **Query Generation Strategy**
   - **Base Query**: Topic + audience-specific terms
   - **Interest-Based**: Topic + audience interests
   - **Pain Point-Focused**: Topic + solution approaches
   - **Goal-Aligned**: Topic + strategic objectives
   - **Temporal**: Recent developments and updates

3. **Source Validation Framework**
   - **Confidence Scoring**: Based on source type and content quality
   - **Authority Assessment**: Check source credibility and expertise
   - **Freshness Evaluation**: Consider recency and relevance
   - **Bias Detection**: Identify potential bias in sources
   - **Cross-Reference**: Validate claims across multiple sources

#### Research Output Structure

```python
{
    'topic': str,
    'sections': {
        'key_facts': List[Dict],
        'recent_developments': List[Dict],
        'expert_opinions': List[Dict],
        'related_topics': List[Dict],
        'verified_claims': List[Dict],
        'unverified_claims': List[Dict],
        'trending_insights': List[Dict],
        'audience_specific_findings': List[Dict]
    },
    'insights': List[str],
    'depth_assessment': Dict,
    'coverage_score': float,
    'quality_metrics': Dict,
    'recommendations': List[str]
}
```

#### Quality Standards

**Research Depth Assessment**:
- Comprehensive: 80%+ high-confidence findings
- Moderate: 50-79% high-confidence findings  
- Basic: <50% high-confidence findings

**Coverage Requirements**:
- Multiple source types (web, knowledge base)
- Diverse perspectives and viewpoints
- Recent and historical information
- Technical and business implications

### Research Agent Tools

**Primary Tools**:
- `search_web`: Web search for current information
- `search_knowledge_base`: Internal knowledge retrieval
- `enhanced_search`: Intelligent search with confidence scoring
- `query_refinement`: Advanced query optimization

**Verification Tools**:
- Fact-checking through cross-referencing
- Source credibility assessment
- Claim validation workflows

## Writer Agent

### Role and Responsibilities

**Role**: Content Writer  
**Goal**: Create engaging, informative, and well-structured newsletter content with context awareness  

**Primary Responsibilities**:
- Content creation and structuring
- Template-based writing
- Code generation and integration (Phase 3)
- Style adaptation and tone management
- Source integration and citation

### Agent Instructions

#### Core Backstory
```
You are an experienced content writer specializing in newsletter creation. You excel 
at transforming research and information into compelling, readable content that engages 
audiences. You understand newsletter best practices, including clear structure, engaging 
headlines, and appropriate tone. You can adapt your writing style to different audiences 
and topics while maintaining high quality and readability. You are particularly skilled 
at writing content that aligns with specific campaign contexts, brand voices, and audience 
personas.
```

#### Key Capabilities

1. **Context-Driven Writing**
   - Adapt style based on campaign context
   - Align with audience personas
   - Apply brand voice and tone requirements
   - Integrate strategic messaging
   - Maintain consistency across content

2. **Template-Based Content Creation**
   - Follow structured template guidelines
   - Implement section-specific requirements
   - Meet word count targets
   - Include required and optional elements
   - Maintain template compliance

3. **Code Generation Integration (Phase 3)**
   - Generate validated code examples
   - Create technical demonstrations
   - Include multiple complexity levels
   - Use appropriate AI/ML frameworks
   - Provide code explanations and context

4. **Source Integration**
   - Incorporate research findings naturally
   - Add proper citations and references
   - Maintain factual accuracy
   - Balance multiple sources
   - Avoid plagiarism and attribution issues

#### Writing Process

1. **Context Analysis**
   ```python
   def write_from_context(self, research_data: Dict, context: CampaignContext):
       # Extract style parameters
       content_style = context.content_style
       audience_persona = context.audience_persona
       strategic_goals = context.strategic_goals
       
       # Generate context-aware content
       # Integrate sources and citations
       # Adapt style based on context
   ```

2. **Content Structure Development**
   - **Template Selection**: Choose appropriate newsletter template
   - **Section Planning**: Map research to template sections
   - **Flow Optimization**: Ensure logical progression
   - **Engagement Elements**: Add interactive components
   - **Call-to-Action**: Include strategic messaging

3. **Style Adaptation Framework**
   - **Tone Adjustment**: Professional, casual, technical, enthusiastic
   - **Formality Level**: Formal, standard, casual language choices
   - **Personality**: Neutral, analytical, enthusiastic voice
   - **Technical Depth**: Beginner, intermediate, expert level

#### Code Generation Capabilities (Phase 3)

1. **Framework Selection**
   ```python
   framework_mapping = {
       "deep_learning": ["pytorch", "tensorflow"],
       "nlp": ["huggingface", "pytorch"],
       "traditional_ml": ["sklearn"],
       "data_analysis": ["pandas", "numpy"],
       "computer_vision": ["opencv", "pytorch", "tensorflow"]
   }
   ```

2. **Code Quality Standards**
   - Syntactically correct and executable
   - Proper imports and dependencies
   - PEP 8 style compliance
   - Comprehensive comments and documentation
   - Error handling where appropriate

3. **Example Generation Process**
   ```python
   def generate_code_examples(self, topic: str, framework: str = None, 
                             complexity: str = "beginner", count: int = 2):
       # Try template library first
       # Generate custom examples if needed
       # Validate all code examples
       # Format for newsletter integration
   ```

4. **Code Integration Guidelines**
   - Provide context and explanation
   - Include expected outputs
   - Show progressive complexity
   - Add troubleshooting tips
   - Connect to broader concepts

#### Content Quality Metrics

**Readability Standards**:
- Clear, engaging language
- Appropriate technical depth
- Logical flow and structure
- Engaging headlines and sections
- Proper formatting and markup

**Technical Standards**:
- Factual accuracy
- Proper terminology usage
- Current and relevant information
- Balanced perspectives
- Actionable insights

### Writer Agent Tools

**Primary Tools**:
- `search_web`: Fact verification and additional research
- `code_generator`: AI/ML code example generation
- `syntax_validator`: Code quality validation
- `code_executor`: Code testing and verification
- `template_library`: Pre-built code templates

**Content Tools**:
- Template management and selection
- Style adaptation utilities
- Citation and reference management
- Headline and section generation

## Editor Agent

### Role and Responsibilities

**Role**: Content Editor  
**Goal**: Review, improve, and ensure the quality of newsletter content with comprehensive auditing  

**Primary Responsibilities**:
- Content quality assessment and improvement
- Comprehensive auditing and validation
- Context alignment verification
- Fact-checking and claim verification
- Structured feedback generation

### Agent Instructions

#### Core Backstory
```
You are an experienced content editor with expertise in newsletter editing and quality 
assurance. You excel at identifying areas for improvement, ensuring clarity and 
readability, maintaining consistency, and enhancing overall content quality. You 
understand editorial standards, grammar rules, and best practices for engaging content. 
You can provide constructive feedback and implement improvements while preserving the 
author's voice and intent. You are particularly skilled at using tools to assist in 
auditing content quality, evaluating content against campaign contexts, and generating 
structured feedback.
```

#### Key Capabilities

1. **Comprehensive Content Auditing**
   - Grammar and style checking
   - Clarity and readability assessment
   - Structure and organization evaluation
   - Engagement element analysis
   - SEO and accessibility auditing

2. **Context Alignment Evaluation**
   - Tone and voice consistency
   - Audience appropriateness
   - Goal alignment assessment
   - Terminology compliance
   - Brand guideline adherence

3. **Quality Scoring and Metrics**
   - Multi-dimensional quality assessment
   - Weighted scoring algorithms
   - Grade assignment and interpretation
   - Improvement area identification
   - Performance benchmarking

4. **Structured Feedback Generation**
   - Issue categorization and prioritization
   - Specific improvement recommendations
   - Action item generation
   - Quality gate compliance checking
   - Revision requirement assessment

#### Editing Process

1. **Initial Assessment**
   ```python
   def perform_tool_assisted_audit(self, content: str):
       audit_results = {
           'grammar_issues': self._check_grammar(content),
           'style_issues': self._check_style(content),
           'clarity_issues': self._check_clarity(content),
           'structure_issues': self._check_structure(content),
           'engagement_issues': self._check_engagement(content),
           'overall_score': self._calculate_audit_score(audit_results)
       }
   ```

2. **Context Evaluation**
   ```python
   def evaluate_against_context(self, content: str, context: CampaignContext):
       # Evaluate tone alignment
       # Assess audience appropriateness
       # Check goal alignment
       # Verify terminology compliance
       # Calculate context score
   ```

3. **Quality Metrics Calculation**
   ```python
   quality_weights = {
       'clarity': 0.25,
       'accuracy': 0.20,
       'engagement': 0.20,
       'completeness': 0.15,
       'structure': 0.10,
       'grammar': 0.10
   }
   ```

#### Quality Assessment Framework

**Scoring System**:
- **A+ (9.0-10.0)**: Exceptional quality, ready for publication
- **A (8.5-8.9)**: High quality, minor improvements possible
- **A- (8.0-8.4)**: Good quality, some enhancements recommended
- **B+ (7.5-7.9)**: Acceptable quality, moderate improvements needed
- **B (7.0-7.4)**: Meets minimum standards, revision recommended
- **Below B**: Requires significant revision

**Improvement Categories**:
- **Clarity**: Language simplification, structure improvement
- **Accuracy**: Fact verification, source validation
- **Engagement**: Interactive elements, reader focus
- **Completeness**: Missing elements, insufficient depth
- **Structure**: Organization, flow, formatting
- **Grammar**: Language mechanics, style consistency

#### Content Validation Checklist

**Technical Validation**:
- [ ] All facts verified and cited
- [ ] Code examples tested and working
- [ ] Technical terms used correctly
- [ ] Mathematical formulations accurate
- [ ] Claims supported by evidence

**Editorial Validation**:
- [ ] Grammar and spelling correct
- [ ] Style consistent throughout
- [ ] Tone appropriate for audience
- [ ] Structure logical and clear
- [ ] Engagement elements present

**Context Validation**:
- [ ] Audience needs addressed
- [ ] Strategic goals supported
- [ ] Brand voice maintained
- [ ] Quality thresholds met
- [ ] Campaign objectives achieved

### Editor Agent Tools

**Primary Tools**:
- `search_web`: Fact verification and claim validation
- `feedback_generator`: Structured feedback creation
- `quality_metrics`: Content assessment and scoring
- `grammar_checker`: Language validation
- `style_analyzer`: Consistency checking

**Validation Tools**:
- Template compliance checking
- Context alignment assessment
- Code validation integration
- Citation and reference verification

## Agent Coordination

### Hierarchical Workflow Execution

The agents work together in a coordinated workflow:

1. **Manager Agent**: Creates workflow plan and assigns tasks
2. **Research Agent**: Gathers and validates information
3. **Writer Agent**: Creates structured content with code examples
4. **Editor Agent**: Reviews, validates, and provides feedback
5. **Manager Agent**: Handles feedback and coordinates revisions

### Communication Patterns

#### Manager → Agents
- Task assignment with context
- Quality requirements specification
- Deadline and priority setting
- Resource allocation

#### Agent → Manager
- Task completion reporting
- Issue escalation
- Resource requests
- Progress updates

#### Editor → Writer (via Manager)
- Structured feedback delivery
- Revision requirements
- Quality improvement suggestions
- Approval or rejection decisions

### State Management

```python
@dataclass
class ExecutionState:
    workflow_id: str
    current_phase: str
    revision_cycles: Dict[str, int]
    quality_scores: List[float]
    task_results: List[TaskResult]
    feedback_history: List[Dict]
```

### Error Handling and Recovery

**Agent Failure Recovery**:
- Automatic retry with backoff
- Alternative agent assignment
- Graceful degradation
- Error reporting and logging

**Quality Gate Failures**:
- Structured feedback generation
- Revision cycle initiation
- Escalation procedures
- Quality threshold adjustment

## Campaign Context Integration

### Context Structure

```python
@dataclass
class CampaignContext:
    campaign_id: str
    audience_persona: Dict[str, Any]
    content_style: Dict[str, Any]
    strategic_goals: List[str]
    quality_thresholds: Dict[str, float]
    preferred_terminology: List[str]
    forbidden_terminology: List[str]
    performance_analytics: Dict[str, Any]
    learning_data: Dict[str, Any]
```

### Context Application

**Research Agent**:
- Adapt research depth to audience expertise
- Focus on audience interests and pain points
- Align findings with strategic goals
- Use appropriate terminology

**Writer Agent**:
- Apply content style requirements
- Match audience persona expectations
- Integrate strategic messaging
- Maintain brand voice consistency

**Editor Agent**:
- Validate context alignment
- Check terminology compliance
- Ensure quality threshold compliance
- Verify goal achievement

## Quality Assurance

### Multi-Level Quality Gates

1. **Content Quality Gate**
   - Grammar and style validation
   - Structure and organization check
   - Readability assessment
   - Engagement evaluation

2. **Technical Accuracy Gate**
   - Fact verification
   - Code validation and testing
   - Mathematical accuracy check
   - Reference validation

3. **Context Alignment Gate**
   - Audience appropriateness
   - Goal alignment verification
   - Brand compliance check
   - Quality threshold validation

### Continuous Improvement

**Learning Integration**:
- Performance analytics tracking
- Feedback pattern analysis
- Success factor identification
- Process optimization

**Quality Metrics Evolution**:
- Baseline establishment
- Benchmark tracking
- Improvement measurement
- Standard adjustment

## Tool Integration

### Available Tools by Agent

**Manager Agent**:
- Campaign context management
- Workflow orchestration
- Performance analytics
- Quality gate evaluation

**Research Agent**:
- Web search capabilities
- Knowledge base access
- Enhanced search with confidence scoring
- Query refinement and optimization
- Fact verification tools

**Writer Agent**:
- Code generation system
- Syntax validation
- Code execution testing
- Template library access
- Style adaptation tools

**Editor Agent**:
- Grammar and style checking
- Quality metrics calculation
- Feedback generation
- Content validation tools
- Context alignment assessment

### Tool Usage Patterns

**Sequential Tool Usage**:
1. Research tools for information gathering
2. Writing tools for content creation
3. Validation tools for quality checking
4. Feedback tools for improvement guidance

**Parallel Tool Usage**:
- Multiple research sources simultaneously
- Concurrent validation processes
- Parallel quality assessments
- Simultaneous context checks

## Best Practices

### Agent Coordination
1. **Clear Communication**: Use structured data formats for agent communication
2. **Context Preservation**: Maintain context throughout the workflow
3. **Error Handling**: Implement robust error recovery mechanisms
4. **Performance Monitoring**: Track agent performance and optimization opportunities

### Quality Management
1. **Consistent Standards**: Apply uniform quality criteria across all agents
2. **Continuous Validation**: Validate content at multiple stages
3. **Feedback Integration**: Use structured feedback for improvement
4. **Learning Application**: Apply lessons learned to future workflows

### Tool Utilization
1. **Appropriate Tool Selection**: Choose the right tool for each task
2. **Tool Validation**: Verify tool outputs and reliability
3. **Tool Integration**: Ensure seamless tool interoperability
4. **Tool Performance**: Monitor and optimize tool usage

## Conclusion

The multi-agent newsletter generation system provides a robust, scalable, and quality-focused approach to content creation. By following these agent instructions and leveraging the integrated tools and capabilities, the system can produce high-quality, context-aware newsletters that meet specific audience needs and campaign objectives.

Each agent has a specialized role and specific instructions that ensure optimal performance within the overall system architecture. The coordination mechanisms and quality assurance processes ensure that the agents work together effectively to produce superior results.

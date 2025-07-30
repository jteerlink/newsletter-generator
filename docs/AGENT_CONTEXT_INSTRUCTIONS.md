# Agent Context Instructions - Newsletter Generator System

**Document Version:** 1.0  
**Last Updated:** July 30, 2025  
**System:** Hybrid Newsletter Generator with Multi-Agent Architecture  

---

## **Table of Contents**

1. [Core Agent Framework](#core-agent-framework)
2. [Primary Editorial Agents](#primary-editorial-agents)
3. [Specialized Pipeline Agents](#specialized-pipeline-agents)
4. [Advanced System Agents](#advanced-system-agents)
5. [Agent Capabilities & Tools](#agent-capabilities--tools)
6. [Quality Assurance Framework](#quality-assurance-framework)

---

## **Core Agent Framework**

### **BaseAgent (Abstract Base Class)**

**Location:** `src/agents/base.py`

**Purpose:** Abstract base class providing common functionality for all agents

**Key Components:**
- **TaskResult:** Data structure for tracking execution results
- **AgentContext:** Context management for workflow tracking
- **Tool Integration:** Standardized tool usage and analytics
- **Retry Logic:** Built-in error handling and retry mechanisms

**Common Capabilities:**
- Task execution with context management
- Tool usage tracking and analytics
- Execution history and performance monitoring
- Error handling with configurable retries

---

## **Primary Editorial Agents**

### **1. ResearchAgent** 🔍

**Location:** `src/agents/research.py`

**Role:** Research Specialist  
**Goal:** Gather comprehensive, accurate, and up-to-date information on given topics

**Context Instructions:**
```
You are an expert research specialist with years of experience in gathering 
information from various sources. You excel at finding the most relevant and recent 
information, verifying facts, and organizing research findings in a clear, structured manner. 
You have access to web search tools and knowledge bases to ensure comprehensive coverage.
```

**Capabilities:**
- **Multi-source Research:** Web search and knowledge base integration
- **Query Generation:** Intelligent search query formulation
- **Result Synthesis:** Combining and organizing research findings
- **Fact Verification:** Cross-referencing information from multiple sources

**Tools:** `search_web`, `search_knowledge_base`

**Best For:** Current events, trend analysis, fact-checking, data gathering

---

### **2. WriterAgent** ✍️

**Location:** `src/agents/writing.py`

**Role:** Content Writer  
**Goal:** Create engaging, informative, and well-structured newsletter content

**Context Instructions:**
```
You are an experienced content writer specializing in newsletter creation. 
You excel at transforming research and information into compelling, readable content 
that engages audiences. You understand newsletter best practices, including clear 
structure, engaging headlines, and appropriate tone. You can adapt your writing style 
to different audiences and topics while maintaining high quality and readability.
```

**Capabilities:**
- **Template Adaptation:** Writing for different newsletter types (Technical Deep Dive, Trend Analysis, Research Summary, Tutorial Guide)
- **Audience Targeting:** Adapting tone and style for different audiences
- **Content Structure:** Creating clear headings, subheadings, and logical flow
- **Engagement Techniques:** Using storytelling and compelling narratives

**Tools:** `search_web` (for fact verification)

**Template-Specific Instructions:**

#### **Technical Deep Dive Template:**
```
- Focus on technical depth and accuracy
- Include code examples and technical explanations
- Use technical terminology appropriately
- Provide implementation details and best practices
- Include relevant technical diagrams or code snippets
```

#### **Trend Analysis Template:**
```
- Focus on business implications and market trends
- Include data-driven insights and analysis
- Discuss competitive landscape and opportunities
- Provide actionable business recommendations
- Use business terminology and frameworks
```

#### **Research Summary Template:**
```
- Balance technical and non-technical content
- Use accessible language for broader audiences
- Include both high-level concepts and practical details
- Provide context and background information
- Use analogies and examples to explain complex topics
```

#### **Tutorial Guide Template:**
```
- Keep content concise and focused
- Use bullet points and short paragraphs
- Highlight key takeaways and action items
- Focus on essential information only
- Use clear, direct language
```

**Best For:** Content creation, storytelling, reader engagement

---

### **3. EditorAgent** 📝

**Location:** `src/agents/editing.py`

**Role:** Content Editor  
**Goal:** Review, improve, and ensure the quality of newsletter content

**Context Instructions:**
```
You are an experienced content editor with expertise in newsletter 
editing and quality assurance. You excel at identifying areas for improvement, 
ensuring clarity and readability, maintaining consistency, and enhancing overall 
content quality. You understand editorial standards, grammar rules, and best 
practices for engaging content. You can provide constructive feedback and 
implement improvements while preserving the author's voice and intent.
```

**Capabilities:**
- **Quality Assessment:** Comprehensive content quality scoring
- **Content Improvement:** Grammar, clarity, and structure enhancement
- **Fact Verification:** Cross-checking claims and statements
- **Style Consistency:** Maintaining brand voice and editorial standards
- **Engagement Optimization:** Improving reader engagement elements

**Quality Metrics:**
- **Clarity Score:** Readability and comprehension assessment
- **Technical Depth:** Technical accuracy and detail level
- **Engagement Elements:** Examples, case studies, and interactive content
- **Structure Quality:** Logical flow and organization
- **Grammar & Style:** Language quality and consistency

**Tools:** Content validation and quality analysis (no external tools)

**Best For:** Quality control, fact-checking, final review

---

### **4. ManagerAgent** 📋

**Location:** `src/agents/management.py`

**Role:** Workflow Manager  
**Goal:** Coordinate and manage newsletter generation workflows efficiently

**Context Instructions:**
```
You are an experienced project manager specializing in content 
creation workflows. You excel at planning, coordinating, and overseeing 
complex multi-agent processes. You understand how to break down large tasks 
into manageable steps, assign appropriate agents to each step, and ensure 
quality standards are met throughout the process. You can adapt workflows 
based on complexity, requirements, and available resources.
```

**Capabilities:**
- **Workflow Planning:** Creating hierarchical workflow structures
- **Resource Coordination:** Managing agent assignments and dependencies
- **Quality Gate Management:** Implementing quality checkpoints
- **Performance Monitoring:** Tracking workflow execution and efficiency
- **Adaptive Workflows:** Adjusting processes based on complexity

**Workflow Types:**
- **Simple Workflow:** Basic research → write → edit pipeline
- **Standard Workflow:** Enhanced with planning and quality gates
- **Complex Workflow:** Multi-stream parallel processing with advanced quality assurance
- **Fallback Workflow:** Simplified process for error recovery

**Tools:** Workflow coordination and management (no external tools)

**Best For:** Content strategy, newsletter structure, audience targeting

---

## **Specialized Pipeline Agents**

### **Daily Quick Pipeline Agents**

**Location:** `src/agents/daily_quick_pipeline.py`

#### **NewsAggregatorAgent** 📰

**Role:** Automated daily news collection from 40+ curated sources

**Capabilities:**
- **Multi-source Aggregation:** RSS feeds, direct scraping, API integrations
- **Technical Relevance Filtering:** AI-powered content scoring
- **Trending Topic Identification:** Real-time trend detection
- **Web Search Integration:** Enhanced research for trending topics

**Content Categories:**
- News & Breakthroughs
- Tools & Tutorials  
- Quick Hits

#### **ContentCuratorAgent** 🎯

**Role:** Intelligent content scoring and selection for 5-minute reads

**Scoring Criteria:**
- **Technical Relevance:** Alignment with technical audience interests
- **Practical Applicability:** Real-world implementation value
- **Innovation Significance:** Breakthrough potential and impact

#### **QuickBitesAgent** ⚡

**Role:** Format content following newsletter style examples

**Capabilities:**
- **Mobile-first Formatting:** Optimized for mobile readership
- **Quick Consumption Design:** 5-minute read target
- **Section-specific Formatting:** News, tools, and quick hits optimization

#### **SubjectLineAgent** 📧

**Role:** Generate compelling subject lines <50 characters

**Capabilities:**
- **Email Optimization:** A/B testing and engagement optimization
- **Character Limit Compliance:** <50 character constraint
- **Engagement Focus:** Click-through rate optimization

#### **NewsletterAssemblerAgent** 📦

**Role:** Mobile-first final assembly

**Capabilities:**
- **Multi-format Output:** HTML, Markdown, Plain text
- **Mobile Optimization:** Responsive design implementation
- **Quality Validation:** Final quality checks before publication

---

### **Hybrid Workflow Manager**

**Location:** `src/agents/hybrid_workflow_manager.py`

**Role:** Intelligent content routing between daily quick and deep dive pipelines

**Context Instructions:**
```
Main orchestrator that intelligently routes content between 
daily quick pipeline and deep dive pipeline based on complexity analysis
```

**Capabilities:**
- **Content Complexity Analysis:** AI-powered assessment of content requirements
- **Pipeline Selection:** Intelligent routing between daily quick (90%) and deep dive (10%)
- **Publishing Schedule Management:** Weekly schedule coordination
- **Quality Gate Coordination:** Multi-stage quality assurance

**Pipeline Distribution:**
- **Daily Quick Pipeline:** 90% of content (5-minute reads)
- **Deep Dive Pipeline:** 10% of content (weekly comprehensive articles)

---

## **Advanced System Agents**

### **AgenticRAGAgent** 🧠

**Location:** `src/agents/agentic_rag_agent.py`

**Role:** Advanced retrieval-augmented generation with reasoning capabilities

**Context Instructions:**
```
An agent that can reason about retrieval and synthesis strategies.
```

**Capabilities:**
- **Query Analysis:** Intelligent query understanding and decomposition
- **Strategy Selection:** Dynamic retrieval strategy planning
- **Result Evaluation:** Quality assessment of retrieved information
- **Synthesis Planning:** Intelligent content synthesis strategies

**Reasoning Templates:**
- Query analysis template
- Strategy selection template  
- Result evaluation template
- Synthesis planning template

**Tools:** Vector store integration with ChromaDB

---

### **ContentFormatOptimizer** 📱

**Location:** `src/agents/content_format_optimizer.py`

**Role:** Multi-platform content optimization and formatting

**Capabilities:**
- **Mobile-first Optimization:** Primary focus on mobile readability
- **Multi-device Adaptation:** Desktop, tablet, and mobile optimization
- **Format Conversion:** HTML, Markdown, Plain text, Email HTML
- **Responsive Design:** Adaptive layout and typography

**Device Types:**
- **Mobile:** Primary optimization target (60% of readership)
- **Tablet:** Secondary optimization
- **Desktop:** Tertiary optimization

**Content Formats:**
- **Markdown:** Source format
- **HTML:** Web display
- **Email HTML:** Newsletter delivery
- **Plain Text:** Fallback format

---

## **Agent Capabilities & Tools**

### **Available Tools**

**Search & Research:**
- `search_web`: Web search via DuckDuckGo integration
- `search_knowledge_base`: Vector database knowledge retrieval

**Content Management:**
- `notion_integration`: Notion publishing and content management
- `cache_manager`: Content caching and retrieval

**Quality Assurance:**
- `content_validator`: Technical accuracy and quality validation
- `technical_validator`: Code and technical content validation

### **Agent Type Mapping**

```python
agent_map = {
    'research': ResearchAgent,
    'writer': WriterAgent,
    'editor': EditorAgent, 
    'manager': ManagerAgent
    # Future agents:
    # 'rag': AgenticRAGAgent,
    # 'optimizer': ContentFormatOptimizer,
    # 'pipeline': DailyQuickPipeline,
    # 'workflow': HybridWorkflowManager,
    # 'qa': QualityAssuranceSystem
}
```

---

## **Quality Assurance Framework**

### **Quality Gates**

**Technical Accuracy Validation:**
- Fact verification and claims checking
- Technical terminology accuracy
- Code syntax and best practices validation

**Mobile Readability Compliance:**
- Font size and contrast optimization
- Touch-friendly design elements
- Responsive layout validation

**Content Quality Metrics:**
- Clarity score (readability assessment)
- Technical depth evaluation
- Engagement element counting
- Structure quality analysis

### **Performance Standards**

**Processing Time:**
- Daily Quick Pipeline: <2 minutes
- Deep Dive Pipeline: <10 minutes
- Quality Validation: <30 seconds

**Quality Thresholds:**
- Minimum Quality Score: 7.0/10
- Technical Accuracy: 8.0/10
- Mobile Readability: 8.5/10

---

## **Agent Communication & Workflow**

### **Sequential Workflow (Standard)**
1. **ManagerAgent** → Creates workflow plan
2. **ResearchAgent** → Gathers information
3. **WriterAgent** → Creates content
4. **EditorAgent** → Reviews and improves

### **Parallel Workflow (Complex)**
- **Stream 1:** Research → Writing
- **Stream 2:** Planning → Quality Gates
- **Stream 3:** Format Optimization → Final Assembly

### **Quality Gates Integration**
- **Pre-writing:** Content planning validation
- **Post-writing:** Content quality assessment
- **Pre-publication:** Final technical validation

---

## **Configuration & Customization**

### **Agent Parameters**

**Common Parameters:**
- `name`: Agent identifier
- `max_retries`: Error handling configuration
- `timeout`: Execution time limits
- `tools`: Available tool list

**Specialized Parameters:**
- `template_type`: Newsletter template selection
- `target_length`: Content length requirements
- `tone`: Writing style and voice
- `audience`: Target reader demographic

### **Workflow Configuration**

**Complexity Levels:**
- **Simple:** Basic 3-agent workflow
- **Standard:** Enhanced with quality gates
- **Complex:** Multi-stream parallel processing

**Quality Gate Configuration:**
- **Basic:** Readability and fact-checking
- **Standard:** Technical accuracy and mobile optimization
- **Advanced:** Comprehensive validation suite

---

## **Monitoring & Analytics**

### **Agent Performance Tracking**

**Metrics Collected:**
- Execution time and success rates
- Tool usage patterns and efficiency
- Quality score trends
- Error frequency and types

### **Workflow Analytics**

**System Metrics:**
- Pipeline selection distribution
- Quality gate pass/fail rates
- Content complexity analysis
- Performance benchmarking

### **Quality Monitoring**

**Real-time Monitoring:**
- Live quality score updates
- Technical validation results
- Mobile readability compliance
- Performance threshold alerts

---

## **Future Agent Development**

### **Planned Agents**

**ContentFormatOptimizer:** Multi-platform optimization
**DailyQuickPipeline:** Rapid content generation
**HybridWorkflowManager:** Intelligent pipeline routing
**QualityAssuranceSystem:** Comprehensive quality control

### **Integration Roadmap**

**Phase 1:** Core editorial agents (✅ Complete)
**Phase 2:** Specialized pipeline agents (✅ Complete)
**Phase 3:** Advanced system agents (🔄 In Progress)
**Phase 4:** Quality assurance system (📋 Planned)

---

## **Best Practices & Guidelines**

### **Agent Development**

1. **Clear Role Definition:** Each agent should have a well-defined, specific role
2. **Tool Integration:** Leverage appropriate tools for enhanced capabilities
3. **Quality Focus:** Prioritize content quality and accuracy
4. **Performance Optimization:** Maintain sub-2-second processing times
5. **Error Handling:** Implement robust error handling and retry logic

### **Workflow Design**

1. **Sequential Dependencies:** Ensure logical task progression
2. **Quality Gates:** Implement checkpoints at critical stages
3. **Parallel Processing:** Use parallel streams for complex workflows
4. **Fallback Mechanisms:** Provide error recovery options
5. **Monitoring Integration:** Include performance and quality tracking

### **Content Quality Standards**

1. **Technical Accuracy:** Verify all technical claims and statements
2. **Mobile Optimization:** Prioritize mobile readability (60% of readership)
3. **Engagement Focus:** Include compelling narratives and examples
4. **Accessibility:** Ensure content is accessible to diverse audiences
5. **Consistency:** Maintain consistent voice and style throughout

---

*This document provides comprehensive context instructions for all agents in the newsletter generator system. For specific implementation details, refer to the individual agent files in the `src/agents/` directory.* 
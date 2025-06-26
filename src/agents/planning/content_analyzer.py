from typing import List, Dict, Any, Set
import re
from collections import Counter
import spacy
from bertopic import BERTopic

class ContentAnalyzer:
    """
    Enhanced content analyzer that dynamically recognizes emerging AI topics,
    companies, models, and products using NER patterns, spaCy, and expandable keyword lists.
    """
    def __init__(self):
        # Core AI/ML keywords for theme identification
        self.ai_keywords = {
            'machine_learning': ['ml', 'machine learning', 'algorithm', 'model', 'training', 'supervised', 'unsupervised'],
            'deep_learning': ['neural network', 'deep learning', 'cnn', 'rnn', 'transformer', 'attention', 'backpropagation'],
            'nlp': ['natural language', 'nlp', 'text processing', 'language model', 'tokenization', 'embedding'],
            'computer_vision': ['computer vision', 'image processing', 'object detection', 'segmentation', 'classification'],
            'robotics': ['robot', 'robotics', 'automation', 'autonomous', 'actuator', 'sensor'],
            'ethics': ['ethics', 'bias', 'fairness', 'responsible ai', 'ai safety', 'alignment', 'transparency']
        }
        
        # Emerging AI companies and organizations
        self.ai_companies = {
            'OpenAI': ['gpt', 'chatgpt', 'dall-e', 'whisper', 'codex'],
            'Google': ['gemini', 'palm', 'bert', 'tensorflow', 'jax'],
            'Anthropic': ['claude', 'constitutional ai'],
            'Meta': ['llama', 'opt', 'fair', 'pytorch'],
            'Microsoft': ['copilot', 'bing chat', 'azure openai'],
            'Apple': ['neural engine', 'core ml', 'siri'],
            'Tesla': ['autopilot', 'full self-driving', 'fsd'],
            'NVIDIA': ['cuda', 'tensor core', 'blackwell', 'hopper'],
            'AMD': ['mi', 'rocm', 'epyc', 'radeon'],
            'Intel': ['gaudi', 'oneapi', 'xeon', 'arc'],
            'Amazon': ['bedrock', 'sagemaker', 'alexa'],
            'IBM': ['watson', 'qiskit', 'cloud pak'],
            'Hugging Face': ['transformers', 'datasets', 'spaces'],
            'Stability AI': ['stable diffusion', 'dreamstudio'],
            'Midjourney': ['midjourney', 'v6'],
            'Cohere': ['command', 'embed', 'classify'],
            'Databricks': ['mlflow', 'delta lake', 'unity catalog'],
            'Snowflake': ['snowpark', 'ml functions'],
            'Palantir': ['foundry', 'apollo', 'gotham']
        }
        
        # Emerging AI models and technologies
        self.emerging_models = {
            'Large Language Models': ['gpt-4', 'gpt-5', 'claude-3', 'gemini ultra', 'llama-3', 'mistral', 'falcon', 'Blackwell'],
            'Multimodal Models': ['gpt-4v', 'gemini pro vision', 'claude-3 sonnet', 'llava', 'cogvlm'],
            'Code Models': ['github copilot', 'amazon codewhisperer', 'tabnine', 'kite'],
            'Image Models': ['dall-e 3', 'midjourney v6', 'stable diffusion xl', 'imagen'],
            'Video Models': ['runway gen-2', 'pika labs', 'stable video diffusion', 'sora'],
            'Audio Models': ['whisper', 'bark', 'musiclm', 'audiocraft', 'stable audio'],
            'AI Hardware': ['Blackwell', 'Hopper', 'Ampere', 'Volta', 'Turing', 'Pascal', 'Maxwell', 'Kepler']
        }
        
        # Emerging AI technologies and concepts
        self.emerging_technologies = {
            'Quantum AI': ['quantum machine learning', 'quantum neural networks', 'quantum advantage'],
            'Edge AI': ['edge computing', 'on-device ai', 'federated learning', 'tiny ml'],
            'AI Safety': ['alignment', 'constitutional ai', 'red teaming', 'safety research'],
            'Explainable AI': ['xai', 'interpretability', 'transparency', 'model explainability'],
            'AI Governance': ['ai regulation', 'ai policy', 'responsible ai', 'ai ethics'],
            'AI Infrastructure': ['mlops', 'feature stores', 'model serving', 'ai observability'],
            'Generative AI': ['generative models', 'diffusion models', 'gan', 'vae'],
            'Reinforcement Learning': ['rlhf', 'ppo', 'dqn', 'actor-critic', 'multi-agent rl']
        }

        self.spacy_nlp = spacy.load('en_core_web_sm')
        self.learned_entities = set()  # For dynamic entity learning

    def analyze_content(self, content: str) -> Dict[str, Any]:
        """
        Enhanced content analysis with dynamic entity recognition (including spaCy NER, disambiguation, and learning).
        """
        words = self._extract_words(content.lower())
        word_freq = Counter(words)
        spacy_entities = self.extract_spacy_entities(content)
        self.learn_new_entities(content)
        
        analysis = {
            'word_count': len(words),
            'unique_words': len(word_freq),
            'most_common_words': word_freq.most_common(10),
            'themes': self.identify_themes(content),
            'key_topics': self.extract_key_topics(content),
            'companies': self.extract_companies(content),
            'models': self.extract_models(content),
            'emerging_technologies': self.extract_emerging_technologies(content),
            'entities': self.extract_entities(content),
            'spacy_entities': spacy_entities,
            'learned_entities': self.get_learned_entities(),
            'sentiment': self.analyze_sentiment(content)
        }
        return analysis

    def identify_themes(self, content: str) -> List[str]:
        """
        Identify themes in the content based on keyword matching.
        """
        content_lower = content.lower()
        themes = []
        
        for theme, keywords in self.ai_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    themes.append(theme)
                    break
        
        return list(set(themes))  # Remove duplicates

    def extract_key_topics(self, content: str) -> List[str]:
        """
        Extract key topics including emerging AI entities.
        """
        topics = set()
        
        # Extract capitalized phrases (potential proper nouns)
        words = self._extract_words(content)
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 2:
                topics.add(word)
        
        # Extract technical terms from all keyword categories
        for category in [self.ai_keywords, self.ai_companies, self.emerging_models, self.emerging_technologies]:
            for items in category.values():
                for item in items:
                    if item.lower() in content.lower():
                        topics.add(item)
        
        # Extract company names and model names
        topics.update(self.extract_companies(content))
        topics.update(self.extract_models(content))
        topics.update(self.extract_emerging_technologies(content))
        
        return list(topics)[:15]  # Limit to top 15

    def extract_companies(self, content: str) -> List[str]:
        """
        Extract AI company names and their products.
        """
        companies = set()
        content_lower = content.lower()
        
        for company, products in self.ai_companies.items():
            # Check for company name
            if company.lower() in content_lower:
                companies.add(company)
            
            # Check for product names
            for product in products:
                if product.lower() in content_lower:
                    companies.add(f"{company} {product}")
        
        return list(companies)

    def extract_models(self, content: str) -> List[str]:
        """
        Extract AI model names and versions, including hyphenated/compound and single-word names.
        """
        models = set()
        content_lower = content.lower()
        words = re.findall(r'\b[A-Z][a-zA-Z0-9]+\b', content)
        
        # Match from known model lists (including single-word models)
        for category, model_list in self.emerging_models.items():
            for model in model_list:
                if model.lower() in content_lower:
                    models.add(model)
                # Also match single-word capitalized models (e.g., 'Blackwell')
                if model.isalpha() and model in words:
                    models.add(model)
        
        # Enhanced model patterns for common AI models
        model_patterns = [
            r'\b(gpt-\d+(?:\.\d+)?)\b',
            r'\b(claude-\d+(?:\.\d+)?)\b',
            r'\b(gemini[- ]?(?:ultra|pro|flash)?)\b',
            r'\b(llama-\d+(?:\.\d+)?)\b',
            r'\b(mistral-\d+(?:\.\d+)?)\b',
            r'\b(falcon-\d+(?:\.\d+)?)\b',
            r'\b(bert|roberta|distilbert|albert)\b',
            r'\b(transformer|attention|encoder|decoder)\b',
            r'\b(cnn|rnn|lstm|gru)\b',
            r'\b(gan|vae|diffusion|stable)\b',
            r'\b(whisper|bark|musiclm|audiocraft)\b',
            r'\b(dall-e|midjourney|stable diffusion|imagen)\b',
            r'\b(copilot|codewhisperer|tabnine)\b'
        ]
        
        for pattern in model_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    models.update([m for m in match if m])
                else:
                    models.add(match)
        
        # Extract capitalized model names with numbers
        capitalized_pattern = r'\b([A-Z][a-z]+(?:[-_. ][A-Z][a-z]+)*[-_. ]?\d+(?:\.\d+)?(?:[-_. ][A-Za-z]+)*)\b'
        for match in re.findall(capitalized_pattern, content):
            if any(char.isdigit() for char in match):
                models.add(match)
        
        return list(models)

    def extract_emerging_technologies(self, content: str) -> List[str]:
        """
        Extract emerging AI technology concepts.
        """
        technologies = set()
        content_lower = content.lower()
        
        for category, tech_list in self.emerging_technologies.items():
            for tech in tech_list:
                if tech.lower() in content_lower:
                    technologies.add(tech)
        
        return list(technologies)

    def extract_entities(self, content: str) -> Dict[str, List[str]]:
        """
        Extract various types of entities from content.
        """
        entities = {
            'companies': self.extract_companies(content),
            'models': self.extract_models(content),
            'technologies': self.extract_emerging_technologies(content),
            'themes': self.identify_themes(content)
        }
        
        return entities

    def add_company(self, company_name: str, products: List[str]):
        """
        Add a new company and its products to the analyzer.
        """
        self.ai_companies[company_name] = products

    def add_model_category(self, category: str, models: List[str]):
        """
        Add a new model category and its models to the analyzer.
        """
        self.emerging_models[category] = models

    def add_technology_category(self, category: str, technologies: List[str]):
        """
        Add a new technology category and its technologies to the analyzer.
        """
        self.emerging_technologies[category] = technologies

    def _extract_words(self, text: str) -> List[str]:
        """
        Extract words from text, filtering out common stop words, but allow short words like 'AI'.
        """
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        # Only filter stop words, not by length
        return [word.lower() for word in words if word.lower() not in stop_words]

    def extract_spacy_entities(self, content: str) -> Dict[str, list]:
        """
        Use spaCy NER to extract organizations, products, and other relevant entities.
        """
        doc = self.spacy_nlp(content)
        orgs = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
        products = [ent.text for ent in doc.ents if ent.label_ in ('PRODUCT', 'WORK_OF_ART', 'EVENT')]
        persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
        gpes = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
        return {
            'organizations': orgs,
            'products': products,
            'persons': persons,
            'gpes': gpes
        }

    def disambiguate_term(self, term: str, content: str) -> str:
        """
        Use spaCy NER and context to disambiguate a term (e.g., 'Apple' as company vs. fruit).
        Returns the entity type or 'ambiguous'.
        """
        doc = self.spacy_nlp(content)
        for ent in doc.ents:
            if ent.text.lower() == term.lower():
                return ent.label_
        # Fallback: check if term is in known companies or products
        if any(term.lower() == c.lower() for c in self.ai_companies.keys()):
            return 'ORG'
        if any(term.lower() == m.lower() for models in self.emerging_models.values() for m in models):
            return 'PRODUCT'
        return 'ambiguous'

    def learn_new_entities(self, content: str):
        """
        Track new capitalized or quoted terms not in current entity lists.
        """
        # Find capitalized words not at the start of a sentence
        words = re.findall(r'(?<![\.!?]\s)(?<!^)\b[A-Z][a-zA-Z0-9]+\b', content)
        # Find quoted terms
        quoted = re.findall(r'"([A-Z][a-zA-Z0-9\- ]+)"', content)
        known_entities = set(self.ai_companies.keys()) | set(m for models in self.emerging_models.values() for m in models)
        for w in words + quoted:
            if w not in known_entities:
                self.learned_entities.add(w)

    def get_learned_entities(self):
        """
        Return the set of dynamically learned entities.
        """
        return list(self.learned_entities)

    def extract_topics_bertopic(self, documents, min_topic_size=2, nr_topics=None):
        """
        Extract topics from a list of documents using BERTopic.
        Returns a dict with topics, topic representations, and probabilities.
        """
        if not documents or len(documents) < 2:
            return {'topics': [], 'topic_info': [], 'probs': []}
        topic_model = BERTopic(min_topic_size=min_topic_size, nr_topics=nr_topics)
        topics, probs = topic_model.fit_transform(documents)
        topic_info = topic_model.get_topic_info()
        return {
            'topics': topics,
            'topic_info': topic_info.to_dict(orient='records'),
            'probs': probs
        }

    def analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """
        Analyze sentiment of the content using a simple lexicon-based approach.
        """
        positive_words = {
            'breakthrough', 'revolutionary', 'significant', 'improved', 'advanced', 'innovative',
            'powerful', 'efficient', 'accurate', 'successful', 'excellent', 'outstanding',
            'remarkable', 'unprecedented', 'superior', 'enhanced', 'optimized', 'streamlined',
            'cutting-edge', 'state-of-the-art', 'groundbreaking', 'transformative'
        }
        
        negative_words = {
            'failure', 'problem', 'issue', 'error', 'bug', 'crash', 'slow', 'inefficient',
            'inaccurate', 'poor', 'bad', 'terrible', 'awful', 'disappointing', 'frustrating',
            'difficult', 'challenging', 'complex', 'complicated', 'expensive', 'costly'
        }
        
        words = self._extract_words(content.lower())
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        total_words = len(words)
        
        if total_words == 0:
            sentiment_score = 0
        else:
            sentiment_score = (positive_count - negative_count) / total_words
        
        # Determine sentiment label
        if sentiment_score > 0.05:
            sentiment_label = 'positive'
        elif sentiment_score < -0.05:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        return {
            'score': sentiment_score,
            'label': sentiment_label,
            'positive_words': positive_count,
            'negative_words': negative_count,
            'total_words': total_words
        } 
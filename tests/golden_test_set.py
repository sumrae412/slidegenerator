"""
Golden Test Dataset for Bullet Generation Quality Testing

This dataset contains hand-crafted examples with expected outputs for
regression testing and quality benchmarking.

Each test case includes:
- Input text
- Context (heading, style)
- Expected bullets (golden reference)
- Quality criteria (thresholds)
"""

GOLDEN_TEST_SET = [
    {
        "id": "edu_ml_basics",
        "category": "educational",
        "input_text": """Students in this course will learn to apply machine learning algorithms
to real-world datasets. The curriculum covers supervised learning fundamentals including
linear regression, logistic regression, and decision trees. Each module includes hands-on
coding exercises using Python and scikit-learn. Students complete a final project analyzing
a dataset of their choice.""",
        "context_heading": "Machine Learning Fundamentals",
        "expected_style": "educational",
        "expected_bullets": [
            "Students apply machine learning algorithms to real-world datasets",
            "Course covers supervised learning fundamentals including regression and classification",
            "Hands-on coding exercises reinforce concepts using Python and scikit-learn",
            "Final project enables students to analyze self-selected datasets"
        ],
        "quality_criteria": {
            "min_bullets": 3,
            "max_bullets": 5,
            "avg_word_length": (7, 15),
            "must_contain_keywords": ["students", "machine learning", "datasets"],
            "style_indicators": ["learn", "course", "apply"]
        }
    },

    {
        "id": "tech_microservices",
        "category": "technical",
        "input_text": """The microservices architecture enables independent deployment and scaling
of application components. Each service communicates via REST APIs using JSON payloads.
The system uses Kubernetes for orchestration and Docker for containerization. Service
discovery is handled by Consul with health checks monitoring endpoint availability. Load
balancing distributes requests across service instances using NGINX.""",
        "context_heading": "Microservices Architecture",
        "expected_style": "technical",
        "expected_bullets": [
            "Microservices architecture enables independent component deployment and scaling",
            "Services communicate via REST APIs using JSON payloads",
            "Kubernetes orchestrates Docker containerized services across infrastructure",
            "Consul manages service discovery with automated health monitoring"
        ],
        "quality_criteria": {
            "min_bullets": 3,
            "max_bullets": 5,
            "avg_word_length": (8, 15),
            "must_contain_keywords": ["microservices", "services", "deployment"],
            "technical_terms": ["API", "Kubernetes", "Docker", "Consul"],
            "style_indicators": ["architecture", "orchestrates", "manages"]
        }
    },

    {
        "id": "exec_digital_transform",
        "category": "executive",
        "input_text": """The digital transformation initiative reduced operational costs by 23%
in Q3 2025. Customer satisfaction scores improved from 72% to 86% following the new UX
redesign. The cloud migration project completed two months ahead of schedule, saving
$1.2M in infrastructure costs. Employee productivity increased 18% after implementing
the new workflow automation tools.""",
        "context_heading": "Digital Transformation Results",
        "expected_style": "executive",
        "expected_bullets": [
            "Digital transformation reduced operational costs by 23% in Q3 2025",
            "UX redesign improved customer satisfaction from 72% to 86%",
            "Cloud migration completed early, saving $1.2M in infrastructure costs",
            "Workflow automation increased employee productivity by 18%"
        ],
        "quality_criteria": {
            "min_bullets": 3,
            "max_bullets": 5,
            "avg_word_length": (7, 12),
            "must_contain_keywords": ["costs", "improved", "productivity"],
            "metrics_indicators": ["%", "$", "Q3"],
            "style_indicators": ["transformation", "migration", "automation"]
        }
    },

    {
        "id": "pro_cloud_benefits",
        "category": "professional",
        "input_text": """Cloud computing provides scalable resources on demand, allowing organizations
to adjust infrastructure capacity based on actual usage. Companies reduce capital expenditure
by eliminating upfront hardware costs and paying only for consumed resources. Cloud providers
manage security and compliance with extensive expertise and certifications. Global reach enables
companies to serve customers worldwide with low-latency performance.""",
        "context_heading": "Cloud Computing Benefits",
        "expected_style": "professional",
        "expected_bullets": [
            "Cloud computing provides scalable on-demand resource adjustment",
            "Organizations eliminate upfront hardware costs through pay-as-you-go pricing",
            "Cloud providers manage security and compliance with certified expertise",
            "Global infrastructure enables low-latency worldwide customer service"
        ],
        "quality_criteria": {
            "min_bullets": 3,
            "max_bullets": 5,
            "avg_word_length": (8, 15),
            "must_contain_keywords": ["cloud", "organizations", "resources"],
            "style_indicators": ["provides", "enables", "manage"]
        }
    },

    {
        "id": "table_feature_comparison",
        "category": "table",
        "input_text": """Feature\tBasic\tPremium\tEnterprise
Storage\t10GB\t100GB\t1TB
Users\t5\t25\tUnlimited
Support\tEmail\tPhone\t24/7 Dedicated
API Access\tNo\tYes\tYes""",
        "context_heading": "Pricing Plan Comparison",
        "expected_style": "professional",
        "expected_bullets": [
            "Storage capacity ranges from 10GB in Basic to 1TB in Enterprise",
            "User limits increase from 5 users to unlimited across plan tiers",
            "Premium and Enterprise plans include API access capabilities",
            "Support escalates from email-only to dedicated 24/7 assistance"
        ],
        "quality_criteria": {
            "min_bullets": 3,
            "max_bullets": 5,
            "avg_word_length": (8, 15),
            "must_contain_keywords": ["storage", "users", "support"],
            "table_specific": True
        }
    },

    {
        "id": "list_consolidation",
        "category": "list",
        "input_text": """• Scalable infrastructure
• Cost optimization
• Pay-as-you-go pricing
• Global availability
• Automatic backups
• High availability
• Disaster recovery
• Security compliance""",
        "context_heading": "Cloud Benefits",
        "expected_style": "professional",
        "expected_bullets": [
            "Infrastructure scales elastically with automatic resource adjustment",
            "Cost optimization through pay-as-you-go flexible pricing models",
            "Global availability ensures high performance across worldwide regions",
            "Security and disaster recovery with automated backup systems"
        ],
        "quality_criteria": {
            "min_bullets": 3,
            "max_bullets": 5,
            "avg_word_length": (8, 15),
            "must_synthesize": True  # Should consolidate, not repeat list items
        }
    },

    {
        "id": "heading_expansion",
        "category": "heading",
        "input_text": """Introduction to Neural Networks""",
        "context_heading": "Introduction to Neural Networks",
        "expected_style": "educational",
        "expected_bullets": [
            "Neural networks model complex patterns inspired by biological brain structure",
            "Artificial neurons process inputs through weighted connections and activation functions",
            "Applications include image recognition, natural language processing, and prediction"
        ],
        "quality_criteria": {
            "min_bullets": 2,
            "max_bullets": 4,
            "avg_word_length": (10, 18),
            "must_expand": True  # Should add info, not repeat heading
        }
    },

    {
        "id": "edge_very_short",
        "category": "edge_case",
        "input_text": """AI improves efficiency.""",
        "context_heading": "AI Benefits",
        "expected_style": "professional",
        "expected_bullets": [
            "Artificial intelligence improves operational efficiency through automation",
            "AI systems optimize workflows and reduce manual processing time"
        ],
        "quality_criteria": {
            "min_bullets": 2,
            "max_bullets": 3,
            "avg_word_length": (7, 15),
            "handle_short_input": True
        }
    },

    {
        "id": "edge_very_long",
        "category": "edge_case",
        "input_text": """Cloud computing has fundamentally transformed how organizations approach
infrastructure, development, and deployment. The shift from on-premises data centers to
cloud-based services represents a paradigm change in IT strategy. Organizations benefit
from elastic scalability, allowing them to dynamically adjust resources based on demand
patterns. Cost models have evolved from capital expenditure to operational expenditure,
providing financial flexibility and eliminating upfront hardware investments. Security
and compliance have become shared responsibilities, with cloud providers maintaining
extensive certifications and security controls. Development teams leverage platform
services to accelerate application delivery, utilizing managed databases, container
orchestration, and serverless computing. Global reach enables organizations to serve
customers worldwide with low-latency performance through distributed data centers.
Disaster recovery and business continuity have been simplified through automated backups
and geographic redundancy. The cloud ecosystem continues evolving with emerging technologies
like artificial intelligence, machine learning, and edge computing integration.""",
        "context_heading": "Cloud Computing Evolution",
        "expected_style": "professional",
        "expected_bullets": [
            "Cloud computing transforms infrastructure from on-premises to elastic scalable services",
            "Organizations shift from capital expenditure to flexible operational cost models",
            "Cloud providers manage extensive security certifications and compliance controls",
            "Platform services accelerate development through managed databases and serverless computing",
            "Global infrastructure enables low-latency worldwide performance and disaster recovery"
        ],
        "quality_criteria": {
            "min_bullets": 4,
            "max_bullets": 5,
            "avg_word_length": (8, 15),
            "must_extract_key_points": True,
            "handle_long_input": True
        }
    },

    {
        "id": "mixed_paragraph_with_metrics",
        "category": "mixed",
        "input_text": """The new analytics platform processes 10 million events per day with
sub-second latency. Implementation reduced infrastructure costs by $400K annually.
The system uses Apache Kafka for event streaming and ClickHouse for real-time analytics.
Development time decreased from 6 months to 2 months using microservices architecture.
Teams deploy updates 5x more frequently with zero downtime.""",
        "context_heading": "Analytics Platform Performance",
        "expected_style": "technical",
        "expected_bullets": [
            "Platform processes 10 million daily events with sub-second latency",
            "Infrastructure costs reduced by $400K annually through optimization",
            "Apache Kafka and ClickHouse enable real-time event streaming analytics",
            "Microservices architecture accelerated development from 6 months to 2 months"
        ],
        "quality_criteria": {
            "min_bullets": 3,
            "max_bullets": 5,
            "avg_word_length": (8, 15),
            "must_contain_keywords": ["platform", "costs", "analytics"],
            "preserve_metrics": True
        }
    },

    {
        "id": "pro_ai_ethics",
        "category": "professional",
        "input_text": """AI ethics considerations include fairness, accountability, transparency,
and privacy protection. Organizations must ensure AI systems do not perpetuate biases
or discriminate against protected groups. Explainable AI helps stakeholders understand
how decisions are made. Data privacy regulations like GDPR require careful handling of
personal information. Continuous monitoring detects drift and performance degradation.""",
        "context_heading": "AI Ethics and Governance",
        "expected_style": "professional",
        "expected_bullets": [
            "AI ethics encompasses fairness, accountability, transparency, and privacy protection",
            "Organizations prevent bias and discrimination in AI decision-making systems",
            "Explainable AI enables stakeholders to understand algorithmic decision processes",
            "GDPR compliance requires careful personal information handling and protection"
        ],
        "quality_criteria": {
            "min_bullets": 3,
            "max_bullets": 5,
            "avg_word_length": (8, 15),
            "must_contain_keywords": ["ethics", "fairness", "privacy"]
        }
    }
]

# Test categories for organized testing
TEST_CATEGORIES = {
    "educational": ["edu_ml_basics"],
    "technical": ["tech_microservices", "mixed_paragraph_with_metrics"],
    "executive": ["exec_digital_transform"],
    "professional": ["pro_cloud_benefits", "pro_ai_ethics"],
    "table": ["table_feature_comparison"],
    "list": ["list_consolidation"],
    "heading": ["heading_expansion"],
    "edge_cases": ["edge_very_short", "edge_very_long"]
}

# Minimum quality thresholds (fail tests below these)
QUALITY_THRESHOLDS = {
    "overall_quality": 70.0,      # Composite score
    "structure_score": 65.0,       # Bullet formatting
    "relevance_score": 70.0,       # Content relevance
    "style_score": 60.0,           # Style consistency
    "readability_score": 65.0      # Readability
}

def get_test_by_id(test_id: str):
    """Retrieve test case by ID"""
    for test in GOLDEN_TEST_SET:
        if test['id'] == test_id:
            return test
    return None

def get_tests_by_category(category: str):
    """Get all tests in a category"""
    test_ids = TEST_CATEGORIES.get(category, [])
    return [get_test_by_id(tid) for tid in test_ids]

def get_all_test_ids():
    """Get list of all test IDs"""
    return [test['id'] for test in GOLDEN_TEST_SET]

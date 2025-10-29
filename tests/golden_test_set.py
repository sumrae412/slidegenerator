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
    },

    {
        "id": "transcript_ai_for_good",
        "category": "transcript",
        "input_text": """[full screen TH]
As you've seen from the previous videos, the possible applications of AI are wide ranging and being able to get an algorithm to reliably perform what might be a relatively simple task for you or me, like recognizing what's in an image, can actually be a powerful tool in many types of projects.

I'd now like to spend a little time discussing some of the potential issues you need to have in mind when it comes to applying AI in your projects.

The goal of any AI for Good project is to have a positive impact in the world, whether that's""",
        "context_heading": "Draft Script",
        "expected_style": "educational",
        "expected_bullets": [
            "the possible applications of AI are wide ranging",
            "being able to get an algorithm to reliably perform what might be a relatively simple task for you or me can actually be a powerful tool in many types of projects",
            "the goal of any AI for Good project is to have a positive impact in the world"
        ],
        "quality_criteria": {
            "min_bullets": 3,
            "max_bullets": 3,
            "avg_word_length": (10, 25),
            "must_not_contain_filler": ["I'd", "As you've seen", "Now let's", "whether that's", "I'm going to"],
            "must_be_complete_sentences": True,
            "must_start_lowercase": True,  # Direct extraction style - preserve original casing
            "no_truncation": True  # Must not end with incomplete thoughts
        }
    },

    {
        "id": "tech_code_documentation",
        "category": "technical",
        "input_text": """The authenticate() function validates user credentials against the database.
It accepts username and password parameters, returns a JWT token on success or null on failure.
Rate limiting prevents brute force attacks by allowing maximum 5 attempts per minute.
Implementation uses bcrypt for password hashing with salt rounds set to 12.""",
        "context_heading": "Authentication API",
        "expected_style": "technical",
        "expected_bullets": [
            "authenticate() validates credentials and returns JWT token or null",
            "Rate limiting restricts login attempts to 5 per minute preventing brute force",
            "Password security implemented with bcrypt hashing using 12 salt rounds"
        ],
        "quality_criteria": {
            "min_bullets": 3,
            "max_bullets": 4,
            "avg_word_length": (8, 14),
            "must_contain_keywords": ["authenticate", "JWT", "rate limiting"],
            "technical_terms": ["bcrypt", "salt", "token"]
        }
    },

    {
        "id": "exec_quarterly_results",
        "category": "executive",
        "input_text": """Q4 revenue reached $2.8M, up 34% year-over-year.
Customer retention improved to 94%, highest in company history.
New enterprise contracts signed with 3 Fortune 500 companies.
Operating margin expanded from 12% to 18% through efficiency gains.""",
        "context_heading": "Q4 Performance",
        "expected_style": "executive",
        "expected_bullets": [
            "Q4 revenue hit $2.8M with 34% year-over-year growth",
            "Customer retention reached record 94%",
            "Secured 3 new Fortune 500 enterprise contracts",
            "Operating margin expanded from 12% to 18% via efficiency improvements"
        ],
        "quality_criteria": {
            "min_bullets": 4,
            "max_bullets": 4,
            "avg_word_length": (7, 12),
            "must_contain_keywords": ["revenue", "retention", "margin"],
            "metrics_indicators": ["%", "$", "Q4"],
            "preserve_metrics": True
        }
    },

    {
        "id": "edu_python_basics",
        "category": "educational",
        "input_text": """This module introduces Python programming fundamentals.
Students learn variables, data types, and control flow structures.
Hands-on exercises cover loops, conditionals, and function definitions.
Final project builds a simple calculator application.""",
        "context_heading": "Python Programming 101",
        "expected_style": "educational",
        "expected_bullets": [
            "Module introduces Python fundamentals including variables and data types",
            "Students practice control flow with loops and conditionals",
            "Hands-on exercises reinforce function definition concepts",
            "Final project applies learning through calculator application development"
        ],
        "quality_criteria": {
            "min_bullets": 3,
            "max_bullets": 4,
            "avg_word_length": (8, 14),
            "must_contain_keywords": ["students", "Python", "project"],
            "style_indicators": ["learn", "introduces", "practice"]
        }
    },

    {
        "id": "pro_remote_work",
        "category": "professional",
        "input_text": """Remote work provides flexibility for employees to balance professional and personal commitments.
Companies reduce overhead costs by minimizing office space requirements.
Asynchronous communication enables global teams to collaborate across time zones.
Video conferencing technology facilitates face-to-face interaction despite physical distance.""",
        "context_heading": "Remote Work Benefits",
        "expected_style": "professional",
        "expected_bullets": [
            "Remote work enables employees to balance professional and personal commitments",
            "Companies reduce overhead by minimizing physical office space needs",
            "Asynchronous communication supports global team collaboration across time zones",
            "Video conferencing maintains face-to-face interaction despite distance"
        ],
        "quality_criteria": {
            "min_bullets": 4,
            "max_bullets": 4,
            "avg_word_length": (8, 13),
            "must_contain_keywords": ["remote", "employees", "companies"]
        }
    },

    {
        "id": "table_api_endpoints",
        "category": "table",
        "input_text": """Endpoint\tMethod\tDescription
/users\tGET\tRetrieve user list
/users/{id}\tGET\tGet specific user
/users\tPOST\tCreate new user
/users/{id}\tPUT\tUpdate user
/users/{id}\tDELETE\tRemove user""",
        "context_heading": "REST API Endpoints",
        "expected_style": "technical",
        "expected_bullets": [
            "GET /users retrieves complete user list",
            "GET /users/{id} fetches specific user details",
            "POST /users creates new user records",
            "PUT and DELETE enable user updates and removal"
        ],
        "quality_criteria": {
            "min_bullets": 3,
            "max_bullets": 4,
            "avg_word_length": (6, 12),
            "must_contain_keywords": ["GET", "POST", "users"],
            "table_specific": True
        }
    },

    {
        "id": "list_nested_features",
        "category": "list",
        "input_text": """Core Features:
• User Management
  - Role-based access control
  - Single sign-on integration
  - Activity logging
• Data Analytics
  - Real-time dashboards
  - Custom report builder
  - Export to CSV/PDF
• API Integration
  - REST endpoints
  - Webhook support
  - Rate limiting""",
        "context_heading": "Platform Features",
        "expected_style": "professional",
        "expected_bullets": [
            "User management includes role-based access, SSO integration, and activity logging",
            "Data analytics provides real-time dashboards and custom reports with CSV/PDF export",
            "API integration supports REST endpoints, webhooks, and rate limiting"
        ],
        "quality_criteria": {
            "min_bullets": 3,
            "max_bullets": 3,
            "avg_word_length": (10, 16),
            "must_synthesize": True
        }
    },

    {
        "id": "edge_dense_technical",
        "category": "edge_case",
        "input_text": """The algorithm implements a modified A* pathfinding with bidirectional search optimization. Time complexity reduces from O(b^d) to O(b^(d/2)) through simultaneous forward-backward traversal. Memory overhead increases linearly with branching factor but remains manageable for typical graph densities. Heuristic admissibility ensures optimal path discovery while maintaining monotonicity constraints.""",
        "context_heading": "Pathfinding Algorithm",
        "expected_style": "technical",
        "expected_bullets": [
            "Modified A* uses bidirectional search for optimization",
            "Time complexity improves from O(b^d) to O(b^(d/2)) via simultaneous traversal",
            "Memory scales linearly with branching factor",
            "Heuristic maintains admissibility and monotonicity for optimal paths"
        ],
        "quality_criteria": {
            "min_bullets": 3,
            "max_bullets": 4,
            "avg_word_length": (8, 14),
            "handle_technical_density": True
        }
    },

    {
        "id": "edge_multiple_metrics",
        "category": "edge_case",
        "input_text": """Performance improved 45% (from 2.3s to 1.26s average response time). Database queries reduced 67% through caching. Memory consumption decreased from 512MB to 187MB (-63%). Error rate dropped from 0.8% to 0.1% (87.5% reduction). CPU utilization optimized from 78% to 34%.""",
        "context_heading": "Optimization Results",
        "expected_style": "executive",
        "expected_bullets": [
            "Performance improved 45% reducing response time from 2.3s to 1.26s",
            "Database queries decreased 67% through caching implementation",
            "Memory consumption reduced 63% from 512MB to 187MB",
            "Error rate dropped 87.5% from 0.8% to 0.1%"
        ],
        "quality_criteria": {
            "min_bullets": 3,
            "max_bullets": 4,
            "avg_word_length": (8, 13),
            "preserve_metrics": True,
            "metrics_indicators": ["%", "MB", "s"]
        }
    },

    {
        "id": "pro_change_management",
        "category": "professional",
        "input_text": """Successful organizational change requires clear communication of objectives and rationale. Leadership must actively engage stakeholders throughout the transition process. Training programs prepare employees for new systems and workflows. Regular feedback loops identify concerns and enable course corrections. Celebrating milestones maintains momentum and reinforces positive behaviors.""",
        "context_heading": "Change Management Best Practices",
        "expected_style": "professional",
        "expected_bullets": [
            "Clear communication establishes change objectives and rationale for stakeholders",
            "Leadership engagement throughout transition builds stakeholder commitment",
            "Training programs prepare employees for new systems and workflows",
            "Feedback loops enable concern identification and course corrections",
            "Milestone celebrations maintain momentum and reinforce progress"
        ],
        "quality_criteria": {
            "min_bullets": 4,
            "max_bullets": 5,
            "avg_word_length": (8, 13),
            "must_contain_keywords": ["change", "employees", "stakeholders"]
        }
    }
]

# Test categories for organized testing
TEST_CATEGORIES = {
    "educational": ["edu_ml_basics", "edu_python_basics"],
    "technical": ["tech_microservices", "mixed_paragraph_with_metrics", "tech_code_documentation"],
    "executive": ["exec_digital_transform", "exec_quarterly_results"],
    "professional": ["pro_cloud_benefits", "pro_ai_ethics", "pro_remote_work", "pro_change_management"],
    "table": ["table_feature_comparison", "table_api_endpoints"],
    "list": ["list_consolidation", "list_nested_features"],
    "heading": ["heading_expansion"],
    "transcript": ["transcript_ai_for_good"],
    "edge_cases": ["edge_very_short", "edge_very_long", "edge_dense_technical", "edge_multiple_metrics"]
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

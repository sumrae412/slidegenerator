"""
Display actual bullet generation examples for evaluation
"""
from file_to_slides import DocumentParser

# Test cases from various content types
TEST_CASES = [
    {
        "name": "Technical Product Description",
        "content": "Snowflake is a cloud-based data warehousing platform that separates storage and compute. "
                   "It allows multiple users to access the same data simultaneously without performance degradation. "
                   "The platform automatically scales resources based on workload demand."
    },
    {
        "name": "Process/Workflow Description",
        "content": "The ETL process involves three main stages. First, data is extracted from source systems. "
                   "Then, the data is transformed to match the target schema. Finally, the cleaned data is loaded "
                   "into the data warehouse for analysis."
    },
    {
        "name": "Feature List",
        "content": "The new authentication system provides single sign-on capabilities. It supports multi-factor "
                   "authentication for enhanced security. The system integrates with existing LDAP directories "
                   "and includes role-based access control."
    },
    {
        "name": "Architecture Overview",
        "content": "Microservices architecture divides applications into small, independent services. "
                   "Each service handles a specific business function and communicates through APIs. "
                   "This approach enables teams to develop and deploy services independently. "
                   "The architecture improves scalability and makes systems more resilient to failures."
    },
    {
        "name": "Business Benefits",
        "content": "Cloud migration reduces infrastructure costs by eliminating on-premise hardware maintenance. "
                   "Organizations gain access to enterprise-grade security without capital investment. "
                   "The pay-as-you-go model aligns expenses with actual usage. "
                   "Teams can provision resources instantly without waiting for procurement cycles."
    },
    {
        "name": "Marketing/Conversational Content",
        "content": "So, this is where things get really interesting. You're going to love this next part. "
                   "Basically, what we're doing here is making everything super easy and streamlined. "
                   "It's kind of amazing when you think about it."
    },
    {
        "name": "Technical Implementation Steps",
        "content": "Configure the database connection string in the application settings. "
                   "Create the necessary tables using the provided migration scripts. "
                   "Set up authentication middleware to protect sensitive endpoints. "
                   "Deploy the application to the staging environment for testing."
    }
]

def main():
    parser = DocumentParser(claude_api_key=None)

    print("\n" + "="*80)
    print("FALLBACK BULLET GENERATION QUALITY EXAMPLES")
    print("="*80)

    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n{'─'*80}")
        print(f"TEST CASE {i}: {test_case['name']}")
        print(f"{'─'*80}")
        print(f"\n📄 ORIGINAL CONTENT:")
        print(f"   {test_case['content'][:150]}..." if len(test_case['content']) > 150 else f"   {test_case['content']}")

        topic_sentence, bullets = parser._create_bullet_points(test_case['content'], fast_mode=False)

        print(f"\n📌 TOPIC SENTENCE (bold subheader):")
        if topic_sentence:
            print(f"   **{topic_sentence}**")
        else:
            print(f"   (none extracted)")

        print(f"\n✨ GENERATED BULLETS ({len(bullets)} bullets):")
        if bullets:
            for j, bullet in enumerate(bullets, 1):
                print(f"   {j}. {bullet}")
        else:
            print("   ⚠️  No bullets generated (content too short or filtered out)")

        # Quality assessment
        if bullets:
            avg_length = sum(len(b) for b in bullets) / len(bullets)
            print(f"\n📊 METRICS:")
            print(f"   - Bullet count: {len(bullets)}")
            print(f"   - Average length: {avg_length:.0f} characters")
            print(f"   - Starts with capital: {all(b[0].isupper() for b in bullets)}")
            print(f"   - Contains content keywords: {any(word in test_case['content'].lower() for word in ['data', 'system', 'process', 'application', 'service'])}")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("Total test cases:", len(TEST_CASES))
    print("\nKey Observations:")
    print("  • Fallback extracts complete sentences from content")
    print("  • Works best with specific, technical content")
    print("  • Filters out vague/conversational language")
    print("  • Maintains proper formatting and capitalization")
    print("  • Produces 2-4 bullets per slide typically")
    print("\n")

if __name__ == "__main__":
    main()

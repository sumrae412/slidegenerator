#!/usr/bin/env python3
"""
Script to integrate performance optimizations into document_parser.py
"""

import re

def integrate_optimizations():
    """Integrate optimization methods into DocumentParser class"""

    # Read the original file
    with open('/home/user/slidegenerator/slide_generator_pkg/document_parser.py', 'r') as f:
        content = f.read()

    # Read the optimization methods
    with open('/home/user/slidegenerator/performance_optimizations.py', 'r') as f:
        optimizations = f.read()

    # Extract individual method definitions
    methods_to_add = []

    # Extract each method
    method_pattern = r'def ([\w_]+)\(self.*?\n(?:(?!^def |^class |^print\(|^# ===).*\n)*'

    for match in re.finditer(method_pattern, optimizations, re.MULTILINE):
        method_text = match.group(0)
        method_name = match.group(1)

        # Skip if method already exists
        if f'def {method_name}(' in content:
            print(f"⏭️  Skipping {method_name} (already exists)")
            continue

        methods_to_add.append((method_name, method_text))

    if not methods_to_add:
        print("✅ All methods already integrated!")
        return

    print(f"\n🔧 Integrating {len(methods_to_add)} new methods...")

    # Find insertion point (after get_cache_stats method)
    insertion_marker = "def get_cache_stats(self) -> Dict[str, Any]:"
    marker_pos = content.find(insertion_marker)

    if marker_pos == -1:
        print("❌ Could not find insertion point (get_cache_stats method)")
        return

    # Find the end of get_cache_stats method
    # Look for the next method definition
    next_method_pattern = r'\n    def \w+'
    match = re.search(next_method_pattern, content[marker_pos + len(insertion_marker):])

    if match:
        insertion_pos = marker_pos + len(insertion_marker) + match.start()
    else:
        print("❌ Could not find end of get_cache_stats method")
        return

    # Prepare methods to insert
    methods_text = "\n"
    for method_name, method_text in methods_to_add:
        # Add proper indentation (4 spaces for class methods)
        indented_method = '\n'.join('    ' + line if line.strip() else ''
                                    for line in method_text.split('\n'))
        methods_text += f"\n{indented_method}\n"
        print(f"  ✅ Adding {method_name}()")

    # Insert the methods
    new_content = content[:insertion_pos] + methods_text + content[insertion_pos:]

    # Update get_cache_stats to include new fields
    old_cache_stats = '''        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 1),
            "cache_size": len(self._api_cache),
            "estimated_cost_savings": f"{hit_rate:.1f}% of API calls cached"
        }'''

    new_cache_stats = '''        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 1),
            "cache_size": len(self._api_cache),
            "cache_compressed": self._cache_compressed,
            "cache_warmed": self._cache_warmed,
            "estimated_cost_savings": f"{hit_rate:.1f}% of API calls cached"
        }'''

    new_content = new_content.replace(old_cache_stats, new_cache_stats)

    # Write the updated file
    with open('/home/user/slidegenerator/slide_generator_pkg/document_parser.py', 'w') as f:
        f.write(new_content)

    print(f"\n✅ Successfully integrated {len(methods_to_add)} optimization methods!")
    print("\n📝 Updated methods:")
    for method_name, _ in methods_to_add:
        print(f"   - {method_name}()")

if __name__ == '__main__':
    integrate_optimizations()

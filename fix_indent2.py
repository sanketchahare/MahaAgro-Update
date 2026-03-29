#!/usr/bin/env python3
"""Fix indentation error in maharashtra_crop_system.py"""

with open("maharashtra_crop_system.py", "r", encoding="utf-8") as f:
    content = f.read()

# Find and replace bad indentation
bad = "        def generate_zone_risk_summary(self):"
good = "    def generate_zone_risk_summary(self):"

if bad in content:
    content = content.replace(bad, good)
    with open("maharashtra_crop_system.py", "w", encoding="utf-8") as f:
        f.write(content)
    print(
        "✅ Fixed indentation error - generate_zone_risk_summary now properly indented"
    )
else:
    print("❌ Pattern not found - file might already be fixed")

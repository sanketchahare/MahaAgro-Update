#!/usr/bin/env python3
"""Fix indentation error in maharashtra_crop_system.py"""

with open("maharashtra_crop_system.py", "r") as f:
    content = f.read()

# Find and fix the indentation issue
bad_pattern = '''    def get_rain_forecast_impact(self, expected_rain, daily_need):
        """Analyze rain impact on irrigation"""
        if expected_rain >= daily_need * 0.8:
            return {"recommendation": "Skip irrigation", "reason": "Rain will meet needs"}
        elif expected_rain >= daily_need * 0.4:
            return {"recommendation": f"Reduce by {(daily_need-expected_rain):.1f}mm", "reason": "Partial rain expected"}
        else:
            return {"recommendation": "Irrigate as planned", "reason": "Insufficient rain"}


        def generate_zone_risk_summary(self):
        """Generate comprehensive zone-wise risk summary"""'''

good_pattern = '''    def get_rain_forecast_impact(self, expected_rain, daily_need):
        """Analyze rain impact on irrigation"""
        if expected_rain >= daily_need * 0.8:
            return {"recommendation": "Skip irrigation", "reason": "Rain will meet needs"}
        elif expected_rain >= daily_need * 0.4:
            return {"recommendation": f"Reduce by {(daily_need-expected_rain):.1f}mm", "reason": "Partial rain expected"}
        else:
            return {"recommendation": "Irrigate as planned", "reason": "Insufficient rain"}

    def generate_zone_risk_summary(self):
        """Generate comprehensive zone-wise risk summary"""'''

if bad_pattern in content:
    content = content.replace(bad_pattern, good_pattern)
    with open("maharashtra_crop_system.py", "w") as f:
        f.write(content)
    print("✅ Fixed indentation error")
else:
    print("Pattern not found - checking alternate pattern...")
    # Try to find just the bad def line
    if "        def generate_zone_risk_summary(self):" in content:
        content = content.replace(
            "        def generate_zone_risk_summary(self):",
            "    def generate_zone_risk_summary(self):",
        )
        with open("maharashtra_crop_system.py", "w") as f:
            f.write(content)
        print("✅ Fixed def indentation")
    else:
        print("❌ Could not find pattern")

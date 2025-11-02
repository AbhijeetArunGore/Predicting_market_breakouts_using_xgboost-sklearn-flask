from app import app
import os

def test_templates():
    """Test if templates are properly set up"""
    with app.test_client() as client:
        response = client.get('/')
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("✅ SUCCESS: Dashboard loaded correctly!")
        else:
            print("❌ ERROR: Dashboard failed to load")

if __name__ == '__main__':
    # Check if templates folder exists
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if os.path.exists(templates_dir):
        print(f"✅ Templates folder found: {templates_dir}")
        
        # Check if dashboard.html exists
        dashboard_path = os.path.join(templates_dir, 'dashboard.html')
        if os.path.exists(dashboard_path):
            print(f"✅ dashboard.html found: {dashboard_path}")
        else:
            print(f"❌ dashboard.html NOT found in: {templates_dir}")
    else:
        print(f"❌ Templates folder NOT found: {templates_dir}")
    
    test_templates()
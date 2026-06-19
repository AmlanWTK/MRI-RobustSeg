import requests
import json
import os

url = "http://127.0.0.1:5000/api/analyze"
test_dir = "test_images"

print(f"Loading images from {test_dir}...")
try:
    files = {
        'flair': open(os.path.join(test_dir, 'FLAIR.png'), 'rb'),
        't1': open(os.path.join(test_dir, 'T1.png'), 'rb'),
        't1ce': open(os.path.join(test_dir, 'T1CE.png'), 'rb'),
        't2': open(os.path.join(test_dir, 'T2.png'), 'rb')
    }

    print(f"Sending POST request to {url}...")
    response = requests.post(url, files=files)
    
    print(f"\nResponse Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("\n--- AI Analysis Results ---")
        
        # Depending on how your backend formats the JSON, let's print the dict structure nicely:
        # We will truncate the long base64 image strings so the console isn't flooded.
        
        if 'images' in data:
            for k in data['images']:
                data['images'][k] = f"<Base64 String - Length: {len(data['images'][k])}>"
                
        print(json.dumps(data, indent=2))
        print("\n✅ API is working perfectly!")
    else:
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"❌ Connection error: {e}")

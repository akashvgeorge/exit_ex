import json
import urllib.request

payload = json.dumps({
    'danceability': 0.5,
    'energy': 0.5,
    'valence': 0.5,
    'tempo': 120,
    'acousticness': 0.1
}).encode('utf-8')

req = urllib.request.Request('http://127.0.0.1:5000/predict', data=payload, headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        print('Status:', resp.status)
        print(resp.read().decode())
except Exception as e:
    print('Request failed:', e)

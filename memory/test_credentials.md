# Test Credentials

## API
- No authentication required for any endpoint
- Base URL: https://damage-vision.preview.emergentagent.com

## Environment Variables
- MONGO_URL: mongodb://localhost:27017/autodamageid
- DB_NAME: autodamageid
- EMERGENT_LLM_KEY: sk-emergent-4A9012420F4F9023b2
- RUNPOD_API_KEY: rpa_922SQN16CHBYBWAOM9D4DT5ZPXYKB6B8CDEX7FNDaob1f6

## Test Data
- Images: /app/assets/crash_1.jpg through crash_13.jpeg
- crash_6.jpg: Known to trigger VLM fallback (unmatched damages)

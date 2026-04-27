# Security & Compliance

## Allowed
- AI coding assistants (Claude, Copilot, etc.)
- Pre-trained models and transfer learning
- Any open-source libraries
- Cloud compute for training
- Ensemble methods

## Prohibited
- Cloud AI APIs at inference time (OpenAI, Azure, Google AI, Anthropic, etc.)
- Hardcoded or pre-computed responses
- Private repositories
- Non-MIT licenses

## Enforcement
- `scripts/validate.sh` checks for prohibited imports in inference path
- Scanned imports: `openai`, `anthropic`, `azure`, `boto3`, `google.cloud.aiplatform`
- All code must be public on GitHub with MIT license

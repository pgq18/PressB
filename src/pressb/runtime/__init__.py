"""Task-independent services, transport, and reinforcement-learning runtime.

Importing this package does not initialize a simulator, CUDA, or model weights.
Task plugins and model adapters are loaded only when a service is configured.
"""

SCHEMA_VERSION = 2
PROTOCOL_VERSION = 1

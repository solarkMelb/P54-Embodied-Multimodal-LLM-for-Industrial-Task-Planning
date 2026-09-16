#!/bin/bash
echo "Waiting for Ollama..."
until curl -s http://ollama:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done
echo "Ollama ready"
exec "$@"
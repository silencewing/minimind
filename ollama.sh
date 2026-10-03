ollama show qwen3.5-agent:9b --modelfile > qwen3.5.mf
docker cp ollama:qwen3.5.mf qwen3.5.mf

docker cp qwen3.5.mf ollama:qwen3.5.mf
from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BASE_DIR / ".env"
PROJECT_DIR = BASE_DIR


def _load_env_file() -> None:
    if not ENV_FILE.exists():
        return
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file()


class Settings:
    agent_host: str = os.getenv("AGENT_HOST", "0.0.0.0")
    agent_port: int = int(os.getenv("AGENT_PORT", "8080"))
    backend_base_url: str = os.getenv("BACKEND_BASE_URL", "http://localhost:9876").rstrip("/")
    agent_token: str = os.getenv("AGENT_TOKEN", "yaai-agent-4967332452c59dfcd51a180b90b865a8")
    ping_interval_seconds: int = int(os.getenv("WS_PING_INTERVAL_SECONDS", "25"))
    pong_timeout_seconds: int = int(os.getenv("WS_PONG_TIMEOUT_SECONDS", "60"))
    small_model_url: str = os.getenv("SMALL_MODEL_URL", "").rstrip("/")
    small_model_name: str = os.getenv("SMALL_MODEL_NAME", "")
    small_model_key: str = os.getenv("SMALL_MODEL_KEY", "")
    large_model_url: str = os.getenv("LARGE_MODEL_URL", "").rstrip("/")
    large_model_name: str = os.getenv("LARGE_MODEL_NAME", "")
    large_model_key: str = os.getenv("LARGE_MODEL_KEY", "")
    vision_model_url: str = os.getenv("VISION_MODEL_URL", "").rstrip("/")
    vision_model_name: str = os.getenv("VISION_MODEL_NAME", "")
    vision_model_key: str = os.getenv("VISION_MODEL_KEY", "")
    embedding_model_url: str = os.getenv("EMBEDDING_MODEL_URL", "").rstrip("/")
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "")
    embedding_model_key: str = os.getenv("EMBEDDING_MODEL_KEY", "")
    max_upload_size_mb: int = int(os.getenv("AGENT_MAX_UPLOAD_SIZE_MB", "512"))
    aliyun_oss_access_key_id: str = os.getenv("ALIYUN_OSS_ACCESS_KEY_ID", "")
    aliyun_oss_access_key_secret: str = os.getenv("ALIYUN_OSS_ACCESS_KEY_SECRET", "")
    aliyun_oss_endpoint: str = os.getenv("ALIYUN_OSS_ENDPOINT", "")
    aliyun_oss_bucket: str = os.getenv("ALIYUN_OSS_BUCKET", "")
    aliyun_oss_base_url: str = os.getenv("ALIYUN_OSS_BASE_URL", "").rstrip("/")
    aliyun_oss_prefix: str = os.getenv("ALIYUN_OSS_PREFIX", "yaai-agent/").strip("/")
    mysql_host: str = os.getenv("MYSQL_HOST", "localhost")
    mysql_port: int = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_database: str = os.getenv("MYSQL_DATABASE", "")
    mysql_username: str = os.getenv("MYSQL_USERNAME", "")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")
    mysql_charset: str = os.getenv("MYSQL_CHARSET", "utf8mb4")
    mysql_pool_size: int = int(os.getenv("MYSQL_POOL_SIZE", "10"))
    rabbitmq_host: str = os.getenv("RABBITMQ_HOST", "localhost")
    rabbitmq_port: int = int(os.getenv("RABBITMQ_PORT", "5672"))
    rabbitmq_username: str = os.getenv("RABBITMQ_USERNAME", "guest")
    rabbitmq_password: str = os.getenv("RABBITMQ_PASSWORD", "guest")
    rabbitmq_vhost: str = os.getenv("RABBITMQ_VHOST", "/")
    chat_exchange: str = os.getenv("CHAT_EXCHANGE", "yaai.chat.exchange")
    chat_queue: str = os.getenv("CHAT_QUEUE", "yaai.chat.queue")
    chat_routing_key: str = os.getenv("CHAT_ROUTING_KEY", "yaai.chat")
    chat_dead_letter_exchange: str = os.getenv("CHAT_DLX", "yaai.chat.dlx")
    chat_dead_letter_queue: str = os.getenv("CHAT_DLQ", "yaai.chat.dlq")
    sensitive_word_file: Path = Path(os.getenv("SENSITIVE_WORD_FILE", str(PROJECT_DIR / "resources" / "sensitive-word.txt")))
    sensitive_replacement_content: str = os.getenv("SENSITIVE_REPLACEMENT_CONTENT", "[该消息因命中敏感词已撤回]")
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_database: int = int(os.getenv("REDIS_DATABASE", "0"))
    redis_password: str = os.getenv("REDIS_PASSWORD", "")
    redis_prefix: str = os.getenv("REDIS_PREFIX", "yaai:agent")
    chat_buffer_ttl_seconds: int = int(os.getenv("CHAT_BUFFER_TTL_SECONDS", "3600"))


settings = Settings()

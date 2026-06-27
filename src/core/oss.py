from __future__ import annotations

import asyncio

import oss2

from src.config import settings


def _endpoint() -> str:
    endpoint = settings.aliyun_oss_endpoint.strip()
    if endpoint and not endpoint.startswith(("http://", "https://")):
        endpoint = f"https://{endpoint}"
    return endpoint


def _object_key(filename: str) -> str:
    prefix = settings.aliyun_oss_prefix
    return f"{prefix}/{filename}" if prefix else filename


def _public_url(object_key: str) -> str:
    base_url = settings.aliyun_oss_base_url
    if base_url:
        return f"{base_url}/{object_key}"

    endpoint = _endpoint().removeprefix("https://").removeprefix("http://")
    return f"https://{settings.aliyun_oss_bucket}.{endpoint}/{object_key}"


def _upload_sync(object_key: str, content: bytes, content_type: str) -> None:
    if not all(
        [
            settings.aliyun_oss_access_key_id,
            settings.aliyun_oss_access_key_secret,
            settings.aliyun_oss_endpoint,
            settings.aliyun_oss_bucket,
        ]
    ):
        raise RuntimeError("Aliyun OSS is not configured")

    auth = oss2.Auth(settings.aliyun_oss_access_key_id, settings.aliyun_oss_access_key_secret)
    bucket = oss2.Bucket(auth, _endpoint(), settings.aliyun_oss_bucket)
    bucket.put_object(object_key, content, headers={"Content-Type": content_type})


def _upload_fileobj_sync(object_key: str, fileobj, content_type: str) -> None:
    if not all(
        [
            settings.aliyun_oss_access_key_id,
            settings.aliyun_oss_access_key_secret,
            settings.aliyun_oss_endpoint,
            settings.aliyun_oss_bucket,
        ]
    ):
        raise RuntimeError("Aliyun OSS is not configured")

    auth = oss2.Auth(settings.aliyun_oss_access_key_id, settings.aliyun_oss_access_key_secret)
    bucket = oss2.Bucket(auth, _endpoint(), settings.aliyun_oss_bucket)
    bucket.put_object(object_key, fileobj, headers={"Content-Type": content_type})


async def upload_to_oss(filename: str, content: bytes, content_type: str) -> str:
    object_key = _object_key(filename)
    await asyncio.to_thread(_upload_sync, object_key, content, content_type)
    return _public_url(object_key)


async def upload_fileobj_to_oss(filename: str, fileobj, content_type: str) -> str:
    object_key = _object_key(filename)
    await asyncio.to_thread(_upload_fileobj_sync, object_key, fileobj, content_type)
    return _public_url(object_key)

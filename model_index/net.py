"""
Copyright (c) 2026-, Zeph Leggett.

This file is part of jetlink-model-index and is licensed under the MIT License.
See the LICENSE file in the root directory for more details.

HTTP with retries, and GitHub's API with the workflow's token. Stdlib only.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from email.message import Message

GITHUB_API = 'https://api.github.com'
GITHUB_HEADERS = {'Accept': 'application/vnd.github+json'}
USER_AGENT = 'jetlink-model-index'
TIMEOUT = 30.0
RETRIES = 3


class NotFound(Exception):
  pass


class NetworkError(Exception):
  pass


def request(url: str, *, data: bytes | None = None, headers: dict | None = None, limit: int = -1) -> tuple[bytes, Message]:
  """The body at `url`, at most `limit` bytes, and the response headers. A 404 is
  NotFound at once; anything else is tried RETRIES times, then NetworkError."""
  headers = {'User-Agent': USER_AGENT, **(headers or {})}
  if (token := os.environ.get('GITHUB_TOKEN')) and url.startswith(GITHUB_API):
    headers['Authorization'] = f'Bearer {token}'
  last: Exception | None = None
  for attempt in range(RETRIES):
    req = urllib.request.Request(url, data=data, headers=headers, method='POST' if data is not None else 'GET')
    try:
      with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
        return response.read(limit), response.headers
    except urllib.error.HTTPError as e:
      if e.code == 404:
        raise NotFound(f'{url}: 404') from e
      last = e
      if e.code in (401, 403, 422) and 'rate limit' not in str(e.read(512)).lower():
        break
    except (OSError, ValueError) as e:
      last = e
    if attempt < RETRIES - 1:
      time.sleep(2 ** attempt)
  raise NetworkError(f'{url}: {last}')


def _decode(url: str, body: bytes):
  try:
    return json.loads(body)
  except ValueError as e:
    raise NetworkError(f'{url} did not serve JSON: {e}') from e


def get(url: str, **kw) -> bytes:
  return request(url, **kw)[0]


def get_json(url: str, **kw):
  return _decode(url, get(url, **kw))


def github(path: str):
  return get_json(f'{GITHUB_API}/{path.lstrip("/")}', headers=GITHUB_HEADERS)


_NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')


def github_pages(path: str, max_pages: int = 50):
  """Every item of a paginated GitHub list, following its Link header."""
  url = f'{GITHUB_API}/{path.lstrip("/")}'
  for _ in range(max_pages):
    body, headers = request(url, headers=GITHUB_HEADERS)
    yield from _decode(url, body)
    found = _NEXT.search(headers.get('Link') or '')
    if not found:
      return
    url = found.group(1)

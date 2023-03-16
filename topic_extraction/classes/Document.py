from __future__ import annotations

import datetime as dt


class Document:
    def __init__(self, body: str | None = None, title: str | None = None, keywords: list = list(), timestamp: int | None = None):
        self.body: str | None = body
        self.title: str | None = title
        self.keywords: list = keywords
        self.timestamp: int | None = timestamp


    
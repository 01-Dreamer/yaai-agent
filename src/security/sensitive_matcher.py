from __future__ import annotations

from dataclasses import dataclass

import ahocorasick

from src.config import settings


@dataclass(frozen=True)
class SensitiveHit:
    word: str
    start: int
    end: int


class SensitiveMatcher:
    def __init__(self) -> None:
        self._automaton: ahocorasick.Automaton | None = None
        self._words: list[str] = []

    def load(self) -> None:
        words: list[str] = []
        if settings.sensitive_word_file.exists():
            for raw_line in settings.sensitive_word_file.read_text(encoding="utf-8").splitlines():
                word = raw_line.strip()
                if word and not word.startswith("#"):
                    words.append(word)

        automaton = ahocorasick.Automaton()
        for index, word in enumerate(words):
            automaton.add_word(word, (index, word))
        automaton.make_automaton()
        self._words = words
        self._automaton = automaton

    def find_first(self, content: str) -> SensitiveHit | None:
        if self._automaton is None:
            self.load()
        if not content or not self._words:
            return None
        assert self._automaton is not None
        for end, (_, word) in self._automaton.iter(content):
            return SensitiveHit(word=word, start=end - len(word) + 1, end=end)
        return None


sensitive_matcher = SensitiveMatcher()

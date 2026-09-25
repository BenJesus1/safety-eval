"""Pydantic schemas for YAML suite files."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field, model_validator

from safety_eval.types import Category, TestCase


class CaseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    should_refuse: bool
    system: str | None = None  # overrides the suite-level system prompt
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None


class SuiteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    category: Category
    version: int = Field(default=1, ge=1)
    description: str | None = None
    system: str | None = None
    cases: list[CaseSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_case_ids(self) -> SuiteSpec:
        dupes = sorted(i for i, n in Counter(c.id for c in self.cases).items() if n > 1)
        if dupes:
            raise ValueError(f"duplicate case ids: {', '.join(dupes)}")
        return self

    def test_cases(self) -> list[TestCase]:
        return [
            TestCase(
                id=c.id,
                category=self.category,
                prompt=c.prompt,
                should_refuse=c.should_refuse,
                system=c.system if c.system is not None else self.system,
                tags=c.tags,
                notes=c.notes,
            )
            for c in self.cases
        ]

#!/usr/bin/env python3
"""AI-visibility gap research for a local lead list.

Pipeline
--------
1. Load leads (business name, city, category, website) from a CSV.
2. Ask ChatGPT and/or Perplexity who they recommend for that category in that
   city, several times with different phrasings.
3. Decide whether the prospect was named; if not, collect the competitors that
   were named instead.
4. Build a personalised line-one audit finding from what was actually returned.
5. Write an enriched CSV for Smartlead/Instantly, plus a companion file listing
   every row that was skipped and why.

Rows are only ever labelled "not mentioned" when at least one query actually
came back and parsed. A lead whose queries all failed is skipped as an error,
never sold as a gap — see --min-successful-queries.

Configuration comes from the environment (OPENAI_API_KEY, PERPLEXITY_API_KEY);
a local .env is loaded automatically when python-dotenv is installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"

# Override per run with --model or the matching env var; these are only defaults.
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
DEFAULT_PERPLEXITY_MODEL = os.environ.get("PERPLEXITY_MODEL", "sonar")

PROVIDER_LABELS = {"openai": "ChatGPT", "perplexity": "Perplexity", "offline": "the offline fixture"}

# Phrasings are varied so one unlucky completion cannot invent a gap on its own.
PROMPT_VARIANTS = (
    "What are the best {category} businesses in {city}? List the top 10.",
    "I need {category} in {city}. Who do you recommend, and who else is worth calling?",
    "Who are the most recommended {category} providers in and around {city}?",
)

SYSTEM_PROMPT = (
    "You recommend local businesses. Answer with JSON only, no prose and no code "
    'fences, in exactly this shape: {"businesses": [{"name": "...", "website": "..."}]}. '
    "Order the list best-first. Only include businesses you actually believe serve the "
    "named city. If you know of none, return an empty list."
)

# Dropped when deciding whether two business names refer to the same company.
GENERIC_TOKENS = frozenset(
    """
    auto automotive detail detailing detailer car care wash mobile ceramic coating tint
    med medical spa medspa aesthetics aesthetic skin laser wellness clinic center centre
    salon studio shop garage service services solutions company co llc inc ltd corp
    the and of a an at in for on best top local premier professional pro quality
    """.split()
)

LEGAL_SUFFIXES = frozenset("llc inc incorporated co corp ltd lp llp pllc".split())

DEFAULT_GAP_TEMPLATE = (
    "I asked {provider} who to call for {category} in {city} and it came back with "
    "{competitor_list} — {company} wasn't in the list."
)
DEFAULT_MENTIONED_TEMPLATE = (
    "I asked {provider} who to call for {category} in {city}; {company} came up at "
    "#{position}, behind {competitor_list}."
)
DEFAULT_MENTIONED_TOP_TEMPLATE = (
    "I asked {provider} who to call for {category} in {city} and {company} came back "
    "first — ahead of {competitor_list}."
)

OUTPUT_FIELDS = (
    "company", "first_name", "email", "city", "state", "category", "website",
    "audit_line", "ai_mentioned", "ai_position",
    "competitor_1", "competitor_2", "competitor_3", "competitor_list",
    "providers", "queries_ok", "queries_total", "mention_confidence", "researched_at",
)


# --------------------------------------------------------------------------- #
# leads
# --------------------------------------------------------------------------- #
class ConfigError(RuntimeError):
    """Raised when configuration or input is unusable."""


@dataclass
class Lead:
    row_number: int
    company: str
    city: str
    category: str
    website: str = ""
    state: str = ""
    email: str = ""
    first_name: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def domain(self) -> str:
        return domain_of(self.website)


# Accepted spellings for each field we care about, normalised to lowercase keys.
COLUMN_ALIASES = {
    "company": ("company", "business_name", "business", "name", "company_name", "lead"),
    "city": ("city", "town", "locality"),
    "category": ("category", "niche", "industry", "vertical", "service", "business_type"),
    "website": ("website", "url", "site", "domain", "web"),
    "state": ("state", "region", "province"),
    "email": ("email", "email_address", "contact_email"),
    "first_name": ("first_name", "firstname", "contact", "contact_name", "owner"),
}


def _pick(row: dict[str, str], names: Sequence[str]) -> str:
    for name in names:
        value = row.get(name)
        if value and value.strip():
            return value.strip()
    return ""


def read_leads(path: Path, default_category: str, default_state: str) -> tuple[list[Lead], list[tuple[int, str, dict]]]:
    """Return (valid leads, [(row_number, reason, raw_row)])."""
    if not path.exists():
        raise ConfigError(f"input file not found: {path}")

    leads: list[Lead] = []
    skipped: list[tuple[int, str, dict]] = []
    seen: set[tuple[str, str]] = set()

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ConfigError(f"{path} has no header row")
        known = {name: tuple(a for a in aliases) for name, aliases in COLUMN_ALIASES.items()}

        for offset, raw in enumerate(reader, start=2):
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items() if k}
            company = _pick(row, known["company"])
            city = _pick(row, known["city"])
            category = _pick(row, known["category"]) or default_category
            website = _pick(row, known["website"])

            reasons = []
            if not company:
                reasons.append("missing business name")
            if not city:
                reasons.append("missing city")
            if not category:
                reasons.append("missing category (add a category column or pass --category)")
            if website and not _looks_like_website(website):
                reasons.append(f"unparseable website {website!r}")
            key = (normalise_name(company), city.lower())
            if not reasons and key in seen:
                reasons.append("duplicate of an earlier row")

            if reasons:
                skipped.append((offset, "; ".join(reasons), raw))
                continue

            seen.add(key)
            consumed = {alias for aliases in known.values() for alias in aliases}
            leads.append(
                Lead(
                    row_number=offset,
                    company=company,
                    city=city,
                    category=category,
                    website=website,
                    state=_pick(row, known["state"]) or default_state,
                    email=_pick(row, known["email"]),
                    first_name=_pick(row, known["first_name"]),
                    extra={k: v for k, v in row.items() if k not in consumed and v},
                )
            )

    return leads, skipped


def _looks_like_website(value: str) -> bool:
    candidate = domain_of(value)
    return bool(re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)+", candidate))


def domain_of(url: str) -> str:
    value = (url or "").strip().lower()
    value = re.sub(r"^[a-z]+://", "", value)
    value = value.split("/", 1)[0].split("?", 1)[0].split("@")[-1]
    return value[4:] if value.startswith("www.") else value


# --------------------------------------------------------------------------- #
# name matching
# --------------------------------------------------------------------------- #
def normalise_name(name: str) -> str:
    text = (name or "").lower()
    text = re.sub(r"[’']", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)

    # Rejoin dotted initialisms ("l.l.c" -> "llc") without gluing "st. mary" together.
    tokens: list[str] = []
    initials: list[str] = []
    for token in text.split():
        if len(token) == 1:
            initials.append(token)
            continue
        if initials:
            tokens.append("".join(initials))
            initials = []
        tokens.append(token)
    if initials:
        tokens.append("".join(initials))

    return " ".join(t for t in tokens if t not in LEGAL_SUFFIXES).strip()


def distinctive_tokens(name: str) -> set[str]:
    return {t for t in normalise_name(name).split() if t not in GENERIC_TOKENS and len(t) > 2}


def same_business(prospect: Lead, candidate_name: str, candidate_site: str = "") -> bool:
    """True when candidate plausibly refers to the prospect.

    Deliberately permissive. A false positive means we treat the prospect as
    mentioned and make no gap claim; a false negative would put "you weren't
    listed" in an email about a business that *was* listed. Place-heavy names
    ("Truckee Meadows Mobile Detail" vs "Truckee Meadows Water Authority") can
    collide here — that is the error we choose to make.
    """
    if prospect.domain and candidate_site and domain_of(candidate_site) == prospect.domain:
        return True

    left, right = normalise_name(prospect.company), normalise_name(candidate_name)
    if not left or not right:
        return False
    if left == right or left in right or right in left:
        return True

    ours, theirs = distinctive_tokens(prospect.company), distinctive_tokens(candidate_name)
    if not ours or not theirs:
        return False
    # Every distinctive word of the prospect appears in the candidate (or vice versa).
    return ours <= theirs or theirs <= ours


# --------------------------------------------------------------------------- #
# providers
# --------------------------------------------------------------------------- #
@dataclass
class QueryOutcome:
    provider: str
    prompt: str
    businesses: list[dict[str, str]]
    ok: bool
    error: str = ""
    raw: str = ""


class Provider:
    name = "base"

    def ask(self, prompt: str, timeout: int) -> QueryOutcome:  # pragma: no cover - interface
        raise NotImplementedError


class OfflineProvider(Provider):
    """Deterministic stand-in so the pipeline can be exercised without API keys."""

    name = "offline"

    def __init__(self, fixtures: dict[str, list[dict[str, str]]] | None = None) -> None:
        self.fixtures = fixtures or {}
        self.current_key = ""

    def ask(self, prompt: str, timeout: int) -> QueryOutcome:
        businesses = self.fixtures.get(self.current_key, self.fixtures.get("*", []))
        if not businesses:
            businesses = [
                {"name": "Example Competitor One", "website": "competitor-one.example"},
                {"name": "Example Competitor Two", "website": "competitor-two.example"},
                {"name": "Example Competitor Three", "website": "competitor-three.example"},
            ]
        return QueryOutcome(self.name, prompt, list(businesses), True, raw="<offline fixture>")


class HTTPProvider(Provider):
    url = ""
    env_key = ""

    def __init__(self, model: str, api_key: str, retries: int = 3) -> None:
        import requests

        self.model = model
        self.retries = retries
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

    def _payload(self, prompt: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }

    def ask(self, prompt: str, timeout: int) -> QueryOutcome:
        import requests

        last_error = ""
        for attempt in range(self.retries):
            try:
                response = self._session.post(self.url, json=self._payload(prompt), timeout=timeout)
            except requests.exceptions.RequestException as exc:
                last_error = f"transport error: {exc}"
            else:
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                elif response.status_code >= 400:
                    return QueryOutcome(
                        self.name, prompt, [], False,
                        f"HTTP {response.status_code}: {response.text[:200]}",
                    )
                else:
                    try:
                        content = response.json()["choices"][0]["message"]["content"]
                    except (ValueError, KeyError, IndexError) as exc:
                        return QueryOutcome(
                            self.name, prompt, [], False, f"unexpected response shape: {exc}"
                        )
                    businesses = parse_businesses(content)
                    if not businesses:
                        return QueryOutcome(
                            self.name, prompt, [], False, "no businesses parsed from reply", content
                        )
                    return QueryOutcome(self.name, prompt, businesses, True, raw=content)

            if attempt < self.retries - 1:
                time.sleep(2 ** attempt + random.uniform(0, 0.5))

        return QueryOutcome(self.name, prompt, [], False, last_error or "request failed")


class OpenAIProvider(HTTPProvider):
    name = "openai"
    url = OPENAI_URL
    env_key = "OPENAI_API_KEY"


class PerplexityProvider(HTTPProvider):
    name = "perplexity"
    url = PERPLEXITY_URL
    env_key = "PERPLEXITY_API_KEY"


def parse_businesses(content: str) -> list[dict[str, str]]:
    """Pull an ordered business list out of a model reply, JSON or prose."""
    text = (content or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    blob = text
    if not blob.startswith("{") and not blob.startswith("["):
        brace = re.search(r"[\{\[].*[\}\]]", blob, re.DOTALL)
        blob = brace.group(0) if brace else ""

    if blob:
        try:
            data = json.loads(blob)
        except ValueError:
            data = None
        if data is not None:
            items = data.get("businesses", data.get("results", [])) if isinstance(data, dict) else data
            parsed = []
            for item in items if isinstance(items, list) else []:
                if isinstance(item, str):
                    parsed.append({"name": item.strip(), "website": ""})
                elif isinstance(item, dict):
                    name = str(item.get("name") or item.get("business") or "").strip()
                    if name:
                        parsed.append({"name": name, "website": str(item.get("website") or "").strip()})
            if parsed:
                return parsed

    # Fallback: numbered or bulleted prose lists.
    parsed = []
    for line in (content or "").splitlines():
        match = re.match(r"\s*(?:\d+[\.\)]|[-*•])\s+(.+)", line)
        if not match:
            continue
        name = match.group(1).strip()
        name = re.split(r"\s+[–—-]\s+|\s*[:(]", name, maxsplit=1)[0]
        name = re.sub(r"^\*\*|\*\*$", "", name).strip().strip("*").strip()
        if name and len(name) < 120:
            parsed.append({"name": name, "website": ""})
    return parsed


def build_providers(names: Sequence[str], model_override: str, retries: int,
                    fixtures: dict | None) -> list[Provider]:
    providers: list[Provider] = []
    for name in names:
        if name == "offline":
            providers.append(OfflineProvider(fixtures))
            continue
        cls = {"openai": OpenAIProvider, "perplexity": PerplexityProvider}[name]
        api_key = os.environ.get(cls.env_key, "").strip()
        if not api_key:
            raise ConfigError(f"{cls.env_key} is not set (needed for --provider {name})")
        default_model = DEFAULT_OPENAI_MODEL if name == "openai" else DEFAULT_PERPLEXITY_MODEL
        providers.append(cls(model_override or default_model, api_key, retries))
    return providers


# --------------------------------------------------------------------------- #
# research
# --------------------------------------------------------------------------- #
@dataclass
class Research:
    lead: Lead
    mentioned: bool
    position: int | None
    competitors: list[str]
    competitors_ahead: list[str]
    queries_ok: int
    queries_total: int
    mention_confidence: float
    providers: list[str]
    errors: list[str] = field(default_factory=list)
    outcomes: list[QueryOutcome] = field(default_factory=list)

    def line_competitors(self) -> list[str]:
        """Competitors the audit line may legitimately name."""
        return self.competitors_ahead if self.mentioned else self.competitors


def research_lead(lead: Lead, providers: Sequence[Provider], variants: int, timeout: int,
                  pause: float) -> Research:
    prompts = [PROMPT_VARIANTS[i % len(PROMPT_VARIANTS)].format(category=lead.category,
                                                                city=lead.city)
               for i in range(variants)]
    outcomes: list[QueryOutcome] = []
    for provider in providers:
        if isinstance(provider, OfflineProvider):
            provider.current_key = lead.domain or normalise_name(lead.company)
        for prompt in prompts:
            outcomes.append(provider.ask(prompt, timeout))
            if pause:
                time.sleep(pause)

    ok = [o for o in outcomes if o.ok]
    hits = 0
    best_position: int | None = None
    tally: Counter[str] = Counter()
    display: dict[str, str] = {}
    ranks: dict[str, int] = {}

    ahead_tally: Counter[str] = Counter()

    for outcome in ok:
        # Where the prospect landed in this reply, so "behind X" stays true per reply.
        own_index = next(
            (i for i, b in enumerate(outcome.businesses, start=1)
             if same_business(lead, b.get("name", ""), b.get("website", ""))),
            None,
        )
        if own_index is not None:
            hits += 1
            best_position = own_index if best_position is None else min(best_position, own_index)

        for index, business in enumerate(outcome.businesses, start=1):
            name = business.get("name", "")
            if same_business(lead, name, business.get("website", "")):
                continue
            key = normalise_name(name)
            if not key:
                continue
            tally[key] += 1
            display.setdefault(key, name.strip())
            ranks[key] = min(ranks.get(key, index), index)
            if own_index is None or index < own_index:
                ahead_tally[key] += 1

    def _order(counter: Counter[str]) -> list[str]:
        return [display[k] for k in sorted(counter, key=lambda k: (-counter[k], ranks[k], k))]

    competitors = _order(tally)
    competitors_ahead = _order(ahead_tally)

    return Research(
        lead=lead,
        mentioned=hits > 0,
        position=best_position,
        competitors=competitors,
        competitors_ahead=competitors_ahead,
        queries_ok=len(ok),
        queries_total=len(outcomes),
        mention_confidence=round(hits / len(ok), 2) if ok else 0.0,
        providers=sorted({o.provider for o in ok}) or sorted({p.name for p in providers}),
        errors=[f"{o.provider}: {o.error}" for o in outcomes if not o.ok],
        outcomes=outcomes,
    )


def join_names(names: Sequence[str]) -> str:
    names = [n for n in names if n]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def provider_label(providers: Sequence[str]) -> str:
    labels = [PROVIDER_LABELS.get(p, p) for p in providers]
    return join_names(labels) or "the assistant"


def build_audit_line(research: Research, templates: dict[str, str], top_n: int) -> str:
    lead = research.lead
    competitors = research.line_competitors()[:top_n]
    values = {
        "company": lead.company,
        "city": lead.city,
        "category": lead.category.lower(),
        "provider": provider_label(research.providers),
        "competitor_list": join_names(competitors),
        "competitor_1": competitors[0] if competitors else "",
        "competitor_2": competitors[1] if len(competitors) > 1 else "",
        "competitor_3": competitors[2] if len(competitors) > 2 else "",
        "position": research.position or "",
    }
    if not research.mentioned:
        key = "gap"
    elif research.position == 1 or not research.competitors_ahead:
        key = "mentioned_top"
    else:
        key = "mentioned"
    if key == "mentioned_top":
        below = research.competitors[:top_n]
        values["competitor_list"] = join_names(below)
        for slot in range(1, 4):
            values[f"competitor_{slot}"] = below[slot - 1] if len(below) >= slot else ""

    try:
        return " ".join(templates[key].format(**values).split())
    except KeyError as exc:
        raise ConfigError(f"unknown placeholder {exc} in the {key} template") from None


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def to_row(research: Research, audit_line: str, top_n: int) -> dict[str, str]:
    lead = research.lead
    competitors = research.line_competitors()[:top_n]
    return {
        "company": lead.company,
        "first_name": lead.first_name,
        "email": lead.email,
        "city": lead.city,
        "state": lead.state,
        "category": lead.category,
        "website": lead.website,
        "audit_line": audit_line,
        "ai_mentioned": "yes" if research.mentioned else "no",
        "ai_position": str(research.position or ""),
        "competitor_1": competitors[0] if competitors else "",
        "competitor_2": competitors[1] if len(competitors) > 1 else "",
        "competitor_3": competitors[2] if len(competitors) > 2 else "",
        "competitor_list": join_names(competitors),
        "providers": ",".join(research.providers),
        "queries_ok": str(research.queries_ok),
        "queries_total": str(research.queries_total),
        "mention_confidence": f"{research.mention_confidence:.2f}",
        "researched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_gap_research.py",
        description="Research AI-recommendation gaps for a lead list and export a "
                    "Smartlead/Instantly-ready CSV.",
    )
    parser.add_argument("-i", "--input", type=Path, default=Path("data/sample_leads.csv"),
                        help="input lead CSV (default: data/sample_leads.csv)")
    parser.add_argument("-o", "--output", type=Path, default=Path("out/leads_enriched.csv"),
                        help="enriched output CSV (default: out/leads_enriched.csv)")
    parser.add_argument("--skipped-output", type=Path, default=None,
                        help="CSV of skipped rows (default: alongside --output as *.skipped.csv)")
    parser.add_argument("--raw-log", type=Path, default=None,
                        help="append every raw provider reply to this JSONL file")
    parser.add_argument("--provider", action="append", choices=("openai", "perplexity", "offline"),
                        help="provider to query, repeatable (default: openai)")
    parser.add_argument("--model", default="", help="override the model for every HTTP provider")
    parser.add_argument("--fixture", type=Path, default=None,
                        help="JSON fixture for --provider offline: {domain-or-name: [{name, website}]}")
    parser.add_argument("--category", default="",
                        help="category to use for rows with no category column")
    parser.add_argument("--state", default="", help="state to use for rows with no state column")
    parser.add_argument("--queries", type=int, default=2, metavar="N",
                        help="prompt phrasings per provider, per lead (default: 2)")
    parser.add_argument("--competitors", type=int, default=3, metavar="N",
                        help="competitors to keep per lead (default: 3)")
    parser.add_argument("--min-successful-queries", type=int, default=1, metavar="N",
                        help="skip a lead unless at least N queries came back (default: 1)")
    parser.add_argument("--require-email", action="store_true",
                        help="skip leads with no email address")
    parser.add_argument("--limit", type=int, default=0, help="process at most N leads")
    parser.add_argument("--timeout", type=int, default=60, help="per-request timeout (default: 60)")
    parser.add_argument("--retries", type=int, default=3, help="retries per request (default: 3)")
    parser.add_argument("--pause", type=float, default=1.0,
                        help="seconds to wait between API calls (default: 1.0)")
    parser.add_argument("--gap-template", default=DEFAULT_GAP_TEMPLATE)
    parser.add_argument("--mentioned-template", default=DEFAULT_MENTIONED_TEMPLATE)
    parser.add_argument("--mentioned-top-template", default=DEFAULT_MENTIONED_TOP_TEMPLATE)
    parser.add_argument("--gap-only", action="store_true",
                        help="export only leads the assistant did not mention")
    parser.add_argument("-v", "--verbose", action="store_true", help="print each audit line")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv()

    try:
        providers = build_providers(
            args.provider or ["openai"], args.model, args.retries,
            json.loads(args.fixture.read_text()) if args.fixture else None,
        )
        leads, skipped = read_leads(args.input, args.category.strip(), args.state.strip())
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"missing dependency: {exc}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"could not read fixture {args.fixture}: {exc}", file=sys.stderr)
        return 2

    if args.limit:
        leads = leads[: args.limit]
    if not leads:
        print(f"no usable leads in {args.input}", file=sys.stderr)

    templates = {
        "gap": args.gap_template,
        "mentioned": args.mentioned_template,
        "mentioned_top": args.mentioned_top_template,
    }
    skipped_rows = [
        {"row": row, "company": (raw.get("business_name") or raw.get("company") or "").strip(),
         "reason": reason}
        for row, reason, raw in skipped
    ]

    print(f"Researching {len(leads)} lead(s) via {', '.join(p.name for p in providers)}\n")
    rows: list[dict[str, str]] = []

    for lead in leads:
        research = research_lead(lead, providers, args.queries, args.timeout, args.pause)

        if args.raw_log:
            args.raw_log.parent.mkdir(parents=True, exist_ok=True)
            with args.raw_log.open("a", encoding="utf-8") as handle:
                for outcome in research.outcomes:
                    handle.write(json.dumps({
                        "company": lead.company, "provider": outcome.provider,
                        "prompt": outcome.prompt, "ok": outcome.ok, "error": outcome.error,
                        "businesses": outcome.businesses, "raw": outcome.raw,
                    }) + "\n")

        if research.queries_ok < args.min_successful_queries:
            detail = "; ".join(research.errors[:2]) or "no usable replies"
            skipped_rows.append({"row": lead.row_number, "company": lead.company,
                                 "reason": f"research failed ({detail})"})
            print(f"  SKIP  {lead.company} — research failed: {detail}")
            continue
        if args.require_email and not lead.email:
            skipped_rows.append({"row": lead.row_number, "company": lead.company,
                                 "reason": "no email address"})
            print(f"  SKIP  {lead.company} — no email address")
            continue
        if not research.mentioned and not research.competitors:
            skipped_rows.append({"row": lead.row_number, "company": lead.company,
                                 "reason": "no competitors returned; nothing to personalise"})
            print(f"  SKIP  {lead.company} — no competitors returned")
            continue
        if args.gap_only and research.mentioned:
            skipped_rows.append({"row": lead.row_number, "company": lead.company,
                                 "reason": f"already mentioned at #{research.position} (--gap-only)"})
            print(f"  SKIP  {lead.company} — already mentioned at #{research.position}")
            continue

        audit_line = build_audit_line(research, templates, args.competitors)
        rows.append(to_row(research, audit_line, args.competitors))
        flag = f"mentioned #{research.position}" if research.mentioned else "GAP"
        print(f"  OK    {lead.company} — {flag} ({research.queries_ok}/{research.queries_total} queries)")
        if args.verbose:
            print(f"        {audit_line}")

    skipped_path = args.skipped_output or args.output.with_suffix(".skipped.csv")
    written = write_csv(args.output, OUTPUT_FIELDS, rows)
    write_csv(skipped_path, ("row", "company", "reason"), skipped_rows)

    print(f"\n{written} lead(s) -> {args.output}")
    print(f"{len(skipped_rows)} skipped -> {skipped_path}")
    if written and not any(r["email"] for r in rows):
        print("\nnote: no email column in the output — add emails before importing to "
              "Smartlead/Instantly.", file=sys.stderr)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())

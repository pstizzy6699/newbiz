#!/usr/bin/env python3
"""Consolidated DNS automation for the outbound-email + site stack.

Commands
--------
push    Upsert SPF, DMARC, the custom tracking CNAME and the four GitHub Pages
        A records through the Cloudflare API. The operation is idempotent:
        records that already hold the desired value are left untouched.
verify  Query public resolvers with dnspython and report PASS/FAIL for SPF,
        DKIM and DMARC propagation. Exits non-zero if any check fails.

Configuration comes from the environment (see .env.example); a local .env is
loaded automatically when python-dotenv is installed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

CLOUDFLARE_API = "https://api.cloudflare.com/client/v4"

# Apex A records for GitHub Pages (https://docs.github.com/pages).
GITHUB_PAGES_IPS = (
    "185.199.108.153",
    "185.199.109.153",
    "185.199.110.153",
    "185.199.111.153",
)

# Fallbacks used only when the matching env var is unset.
DEFAULT_SPF_VALUE = "v=spf1 ~all"
DEFAULT_DKIM_SELECTOR = "default"
DEFAULT_TRACKING_CNAME_NAME = "track"
DEFAULT_RESOLVERS = ("1.1.1.1", "8.8.8.8")

RECORD_COMMENT = "managed by engine_setup.py"
TTL_AUTO = 1


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


@dataclass
class Config:
    domain: str
    api_token: str = ""
    zone_id: str = ""
    dkim_selector: str = DEFAULT_DKIM_SELECTOR
    dkim_value: str = ""
    tracking_name: str = DEFAULT_TRACKING_CNAME_NAME
    tracking_target: str = ""
    spf_value: str = DEFAULT_SPF_VALUE
    dmarc_value: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def dmarc_name(self) -> str:
        return f"_dmarc.{self.domain}"

    @property
    def dkim_name(self) -> str:
        return f"{self.dkim_selector}._domainkey.{self.domain}"

    @property
    def tracking_fqdn(self) -> str:
        name = self.tracking_name
        return name if name.endswith(self.domain) else f"{name}.{self.domain}"


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def load_config(require_cloudflare: bool) -> Config:
    """Read configuration from the environment, loading .env when available."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # optional dependency
        pass
    else:
        load_dotenv()

    domain = _env("PRIMARY_DOMAIN")
    if not domain:
        raise ConfigError("PRIMARY_DOMAIN is not set (copy .env.example to .env)")
    domain = domain.rstrip(".").lower()

    cfg = Config(
        domain=domain,
        api_token=_env("CLOUDFLARE_API_TOKEN"),
        zone_id=_env("CLOUDFLARE_ZONE_ID"),
        dkim_selector=_env("DKIM_SELECTOR", DEFAULT_DKIM_SELECTOR),
        dkim_value=_env("DKIM_VALUE"),
        tracking_name=_env("TRACKING_CNAME_NAME", DEFAULT_TRACKING_CNAME_NAME).rstrip("."),
        tracking_target=_env("TRACKING_CNAME_TARGET").rstrip("."),
        spf_value=_env("SPF_VALUE"),
        dmarc_value=_env("DMARC_VALUE"),
    )

    if not cfg.spf_value:
        cfg.spf_value = DEFAULT_SPF_VALUE
        cfg.warnings.append(
            f"SPF_VALUE is unset; falling back to {DEFAULT_SPF_VALUE!r}, which "
            "authorises no senders. Set SPF_VALUE to your provider's record."
        )
    if not cfg.dmarc_value:
        cfg.dmarc_value = f"v=DMARC1; p=none; rua=mailto:dmarc@{domain}; fo=1"

    if require_cloudflare:
        missing = [
            key
            for key, value in (
                ("CLOUDFLARE_API_TOKEN", cfg.api_token),
                ("CLOUDFLARE_ZONE_ID", cfg.zone_id),
                ("TRACKING_CNAME_TARGET", cfg.tracking_target),
            )
            if not value
        ]
        if missing:
            raise ConfigError("missing required env var(s): " + ", ".join(missing))

    return cfg


# --------------------------------------------------------------------------- #
# cloudflare client
# --------------------------------------------------------------------------- #
class CloudflareError(RuntimeError):
    """Raised when the Cloudflare API returns an error payload."""


@dataclass(frozen=True)
class Record:
    type: str
    name: str
    content: str
    ttl: int = TTL_AUTO
    proxied: bool = False

    def payload(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "type": self.type,
            "name": self.name,
            "content": self.content,
            "ttl": self.ttl,
            "comment": RECORD_COMMENT,
        }
        if self.type in ("A", "AAAA", "CNAME"):
            body["proxied"] = self.proxied
        return body


class Cloudflare:
    def __init__(self, token: str, zone_id: str, timeout: int = 30) -> None:
        import requests  # imported lazily so `verify` runs without it

        self.zone_id = zone_id
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        )

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        import requests

        url = f"{CLOUDFLARE_API}/zones/{self.zone_id}{path}"
        try:
            response = self._session.request(method, url, timeout=self.timeout, **kwargs)
        except requests.exceptions.RequestException as exc:
            raise CloudflareError(f"{method} {path} -> could not reach the API: {exc}") from None
        try:
            body = response.json()
        except ValueError:
            raise CloudflareError(
                f"{method} {path} -> HTTP {response.status_code}: {response.text[:300]}"
            ) from None
        if not body.get("success", False):
            detail = "; ".join(
                f"{err.get('code')}: {err.get('message')}" for err in body.get("errors") or []
            )
            raise CloudflareError(
                f"{method} {path} failed (HTTP {response.status_code}): {detail or body}"
            )
        return body.get("result")

    def list_records(self, rtype: str, name: str) -> list[dict[str, Any]]:
        return self._call(
            "GET", "/dns_records", params={"type": rtype, "name": name, "per_page": 100}
        ) or []

    def create_record(self, record: Record) -> dict[str, Any]:
        return self._call("POST", "/dns_records", json=record.payload())

    def update_record(self, record_id: str, record: Record) -> dict[str, Any]:
        return self._call("PUT", f"/dns_records/{record_id}", json=record.payload())

    def delete_record(self, record_id: str) -> Any:
        return self._call("DELETE", f"/dns_records/{record_id}")


# --------------------------------------------------------------------------- #
# push
# --------------------------------------------------------------------------- #
def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


def _matches(existing: dict[str, Any], record: Record) -> bool:
    if _unquote(str(existing.get("content", ""))) != _unquote(record.content):
        return False
    if int(existing.get("ttl", TTL_AUTO)) != record.ttl:
        return False
    if record.type in ("A", "AAAA", "CNAME"):
        return bool(existing.get("proxied", False)) == record.proxied
    return True


def _upsert(cf: Cloudflare, record: Record, dry_run: bool) -> str:
    """Create or update a record that should exist exactly once."""
    existing = cf.list_records(record.type, record.name)

    # TXT names can legitimately hold several records; only manage ours.
    if record.type == "TXT":
        prefix = record.content.split(";", 1)[0].split(" ", 1)[0]
        existing = [r for r in existing if _unquote(str(r.get("content", ""))).startswith(prefix)]

    if not existing:
        if not dry_run:
            cf.create_record(record)
        return "created"

    current, *duplicates = existing
    action = "unchanged" if _matches(current, record) else "updated"
    if action == "updated" and not dry_run:
        cf.update_record(current["id"], record)
    for dupe in duplicates:
        if not dry_run:
            cf.delete_record(dupe["id"])
        action = f"{action} (+{len(duplicates)} duplicate(s) removed)"
    return action


def _sync_set(cf: Cloudflare, rtype: str, name: str, contents: Sequence[str], dry_run: bool,
              prune: bool) -> list[tuple[str, str]]:
    """Reconcile a name that holds several records of one type (GitHub Pages A)."""
    existing = cf.list_records(rtype, name)
    by_content = {_unquote(str(r.get("content", ""))): r for r in existing}
    results: list[tuple[str, str]] = []

    for content in contents:
        record = Record(rtype, name, content)
        current = by_content.pop(content, None)
        if current is None:
            if not dry_run:
                cf.create_record(record)
            results.append((content, "created"))
        elif _matches(current, record):
            results.append((content, "unchanged"))
        else:
            if not dry_run:
                cf.update_record(current["id"], record)
            results.append((content, "updated"))

    for content, record in by_content.items():
        if prune:
            if not dry_run:
                cf.delete_record(record["id"])
            results.append((content, "deleted"))
        else:
            results.append((content, "extra (left in place; use --prune to remove)"))

    return results


def cmd_push(args: argparse.Namespace) -> int:
    cfg = load_config(require_cloudflare=True)
    for warning in cfg.warnings:
        print(f"warning: {warning}", file=sys.stderr)

    prefix = "[dry-run] " if args.dry_run else ""
    print(f"{prefix}Pushing DNS for {cfg.domain} (zone {cfg.zone_id})\n")

    cf = Cloudflare(cfg.api_token, cfg.zone_id, timeout=args.timeout)
    singles = [
        ("SPF", Record("TXT", cfg.domain, cfg.spf_value)),
        ("DMARC", Record("TXT", cfg.dmarc_name, cfg.dmarc_value)),
        ("Tracking CNAME", Record("CNAME", cfg.tracking_fqdn, cfg.tracking_target)),
    ]

    failures = 0
    for label, record in singles:
        try:
            action = _upsert(cf, record, args.dry_run)
        except CloudflareError as exc:
            failures += 1
            print(f"  {label:<15} {record.name} -> ERROR: {exc}")
        else:
            print(f"  {label:<15} {record.name} -> {action}")
            print(f"  {'':<15} value: {record.content}")

    print(f"\n  {'GitHub Pages':<15} {cfg.domain} (A)")
    try:
        for content, action in _sync_set(
            cf, "A", cfg.domain, GITHUB_PAGES_IPS, args.dry_run, args.prune
        ):
            print(f"  {'':<15}   {content:<16} -> {action}")
    except CloudflareError as exc:
        failures += 1
        print(f"  {'':<15}   ERROR: {exc}")

    if failures:
        print(f"\n{failures} operation(s) failed.")
        return 1

    print(
        "\nDone."
        + ("" if args.dry_run else f" Run `{os.path.basename(sys.argv[0])} verify` once DNS propagates.")
    )
    return 0


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
@dataclass
class CheckResult:
    name: str
    target: str
    passed: bool
    detail: str
    value: str = ""

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


def _build_resolver(nameservers: Sequence[str], timeout: float):
    import dns.resolver

    resolver = dns.resolver.Resolver(configure=not nameservers)
    if nameservers:
        resolver.nameservers = list(nameservers)
    resolver.timeout = timeout
    resolver.lifetime = timeout
    return resolver


def _txt_records(resolver, name: str) -> tuple[list[str], str]:
    """Return (values, error). TXT strings are joined per RFC 7208 §3.3."""
    import dns.exception
    import dns.resolver

    try:
        answer = resolver.resolve(name, "TXT")
    except dns.resolver.NXDOMAIN:
        return [], "name does not exist (NXDOMAIN)"
    except dns.resolver.NoAnswer:
        return [], "no TXT record at this name"
    except dns.resolver.NoNameservers as exc:
        return [], f"resolution failed: {exc}"
    except dns.exception.Timeout:
        return [], "query timed out"

    values = []
    for rdata in answer:
        values.append(b"".join(rdata.strings).decode("utf-8", "replace"))
    return values, ""


def _normalise(value: str) -> str:
    return " ".join(value.split()).strip().strip('"')


def _tag(record: str, tag: str) -> str:
    for part in record.split(";"):
        part = part.strip()
        if part.lower().startswith(f"{tag.lower()}="):
            return part.split("=", 1)[1].strip()
    return ""


def _check_spf(resolver, cfg: Config) -> CheckResult:
    values, error = _txt_records(resolver, cfg.domain)
    if error:
        return CheckResult("SPF", cfg.domain, False, error)

    spf = [v for v in values if v.lower().startswith("v=spf1")]
    if not spf:
        return CheckResult("SPF", cfg.domain, False, "no v=spf1 record found")
    if len(spf) > 1:
        return CheckResult(
            "SPF", cfg.domain, False, f"{len(spf)} SPF records found (RFC 7208 allows one)",
            " | ".join(spf),
        )

    found = spf[0]
    expected = _normalise(cfg.spf_value)
    if expected and _normalise(found) != expected:
        return CheckResult(
            "SPF", cfg.domain, False, f"published record differs from SPF_VALUE ({expected})", found
        )
    return CheckResult("SPF", cfg.domain, True, "published and unique", found)


def _check_dkim(resolver, cfg: Config) -> CheckResult:
    name = cfg.dkim_name
    values, error = _txt_records(resolver, name)
    if error:
        return CheckResult("DKIM", name, False, error)

    dkim = [v for v in values if "p=" in v or v.lower().startswith("v=dkim1")]
    if not dkim:
        return CheckResult("DKIM", name, False, "no DKIM record found at this selector")

    found = dkim[0]
    if not cfg.dkim_value:
        return CheckResult(
            "DKIM", name, True, "record present (set DKIM_VALUE to compare the key)", found
        )

    expected_key = _tag(cfg.dkim_value, "p") or _normalise(cfg.dkim_value)
    found_key = _tag(found, "p") or _normalise(found)
    if expected_key.replace(" ", "") != found_key.replace(" ", ""):
        return CheckResult("DKIM", name, False, "public key does not match DKIM_VALUE", found)
    return CheckResult("DKIM", name, True, "public key matches DKIM_VALUE", found)


def _check_dmarc(resolver, cfg: Config) -> CheckResult:
    name = cfg.dmarc_name
    values, error = _txt_records(resolver, name)
    if error:
        return CheckResult("DMARC", name, False, error)

    dmarc = [v for v in values if v.lower().startswith("v=dmarc1")]
    if not dmarc:
        return CheckResult("DMARC", name, False, "no v=DMARC1 record found")
    if len(dmarc) > 1:
        return CheckResult("DMARC", name, False, f"{len(dmarc)} DMARC records found (expected one)")

    found = dmarc[0]
    policy = _tag(found, "p")
    if not policy:
        return CheckResult("DMARC", name, False, "record has no policy (p=) tag", found)
    return CheckResult("DMARC", name, True, f"policy p={policy}", found)


def cmd_verify(args: argparse.Namespace) -> int:
    cfg = load_config(require_cloudflare=False)

    try:
        resolver = _build_resolver(args.resolver, args.timeout)
    except ImportError:
        print("dnspython is not installed: pip install -r requirements.txt", file=sys.stderr)
        return 2

    results = [
        _check_spf(resolver, cfg),
        _check_dkim(resolver, cfg),
        _check_dmarc(resolver, cfg),
    ]

    if args.json:
        print(json.dumps(
            {
                "domain": cfg.domain,
                "resolvers": list(args.resolver),
                "passed": all(r.passed for r in results),
                "checks": [
                    {
                        "name": r.name,
                        "target": r.target,
                        "status": r.status,
                        "detail": r.detail,
                        "value": r.value,
                    }
                    for r in results
                ],
            },
            indent=2,
        ))
        return 0 if all(r.passed for r in results) else 1

    print(f"DNS verification for {cfg.domain}")
    print(f"resolvers: {', '.join(args.resolver) or 'system'}\n")
    for result in results:
        print(f"  [{result.status}] {result.name:<6} {result.target}")
        print(f"         {result.detail}")
        if result.value:
            print(f"         value: {result.value}")
        print()

    failed = [r.name for r in results if not r.passed]
    if failed:
        print(f"FAIL — {len(failed)} of {len(results)} checks failed: {', '.join(failed)}")
        return 1
    print(f"PASS — all {len(results)} checks passed.")
    return 0


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="engine_setup.py",
        description="Push and verify the email/site DNS records for PRIMARY_DOMAIN.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    push = sub.add_parser("push", help="upsert DNS records via the Cloudflare API")
    push.add_argument(
        "--dry-run",
        action="store_true",
        help="report the changes that would be made; still reads current records",
    )
    push.add_argument(
        "--prune",
        action="store_true",
        help="delete apex A records that are not GitHub Pages addresses",
    )
    push.add_argument("--timeout", type=int, default=30, help="API timeout in seconds (default: 30)")
    push.set_defaults(func=cmd_push)

    verify = sub.add_parser("verify", help="check SPF, DKIM and DMARC on public DNS")
    verify.add_argument(
        "--resolver",
        action="append",
        default=None,
        metavar="IP",
        help=f"resolver to query, repeatable (default: {', '.join(DEFAULT_RESOLVERS)})",
    )
    verify.add_argument(
        "--timeout", type=float, default=10.0, help="query timeout in seconds (default: 10)"
    )
    verify.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    verify.set_defaults(func=cmd_verify)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "verify" and args.resolver is None:
        args.resolver = list(DEFAULT_RESOLVERS)

    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f"missing dependency: {exc}. Run: pip install -r requirements.txt", file=sys.stderr)
        return 2
    except CloudflareError as exc:
        print(f"cloudflare error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())

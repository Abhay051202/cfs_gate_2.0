"""Create identity-isolated records without discarding a second container."""
from collections import Counter
from dataclasses import replace
from uuid import NAMESPACE_URL, uuid5

from app.domain import OCRRead
from app.ocr.validator import ContainerValidator


def plan_captures(group, confirmations):
    if not group:
        return []
    validator = ContainerValidator()
    numbers, unknown = {}, []
    for passage in group:
        for region in passage.regions:
            observed = False
            for number, count in region.votes.items():
                read = region.reads.get(number)
                if not count or not read:
                    continue
                value, job = read
                actual = validator.validate(OCRRead(value.raw_text, value.confidence))
                if not actual.valid_check_digit or actual.normalized_text != number:
                    continue
                numbers.setdefault(number, []).append((passage, region, count))
                observed = True
            if not observed and region.best_any:
                unknown.append((passage, region))
    # Chassis ownership plus two observed 20-foot ISO codes is necessary for a
    # twin label. Absence of that label must never discard the second identity.
    reliable_twin = len(numbers) == 2 and any(
        p.chassis_observed and not p.ambiguous and all(
            any(owner.id == p.id for owner, _, _ in records) for records in numbers.values())
        for p in group)
    reliable_twin = reliable_twin and all(any(
        str(region.reads[number][0].size_type or '').startswith('2')
        for _, region, _ in records) for number, records in numbers.items())
    # Different IDs in one text region are competing OCR, not evidence of two
    # physical boxes. Preserve both candidates, but require operator review.
    conflicts = {id(r) for p in group for r in p.regions
                 if sum(id(r) == id(other) for records in numbers.values()
                        for _, other, _ in records) > 1}
    reliable_twin = reliable_twin and not conflicts
    if not reliable_twin and len(numbers) > 1:
        from app.database.repositories.gate import GateRepository
        merged = {}
        for n, records in numbers.items():
            matched_w = None
            for w in merged:
                if GateRepository._same_container_id(n, w):
                    matched_w = w
                    break
            if matched_w:
                w_votes = sum(c for _, _, c in merged[matched_w])
                n_votes = sum(c for _, _, c in records)
                if n_votes > w_votes:
                    merged[n] = merged.pop(matched_w) + records
                else:
                    merged[matched_w].extend(records)
            else:
                merged[n] = records
        numbers = merged
    bundles = [sorted(numbers)] if reliable_twin else [[n] for n in sorted(numbers)]
    base = min(p.visit_id or p.id for p in group)
    plans = []
    for bundle in bundles:
        regions, owners = [], []
        for number in bundle:
            for owner, region, count in numbers[number]:
                value, job = region.reads[number]
                # A planned region carries only its identity. The repository
                # must not independently choose another confidence winner.
                regions.append(region if set(region.votes) == {number} else replace(region, votes=Counter({number: count}),
                    reads={number: (value, job)}, best_any=(value, job),
                    source_proofs={number: dict(region.source_proofs.get(number, {}))},
                    vote_sequences={number: set(region.vote_sequences.get(number, set()))}))
                owners.append(owner)
        first = min(owners, key=lambda p: (p.started, p.id))
        identity = str(uuid5(NAMESPACE_URL, 'gate-container:' + base + ':' + ','.join(bundle)))
        plans.append(replace(first, id=identity, regions=regions,
            selected_numbers=tuple(bundle), identity_conflict=any(id(r) in conflicts
                for number in bundle for _, r, _ in numbers[number]),
            ambiguous=False if reliable_twin else any(p.ambiguous for p in owners), final_evidence=True, closed=True))
    if not numbers and unknown and any(not p.capture_only for p, _ in unknown):
        import re
        def _is_meaningful_unreadable(p, r):
            if not r.best_any:
                return False
            raw = (getattr(r.best_any[0], 'raw_text', '') or '').strip().upper()
            clean = re.sub(r'[^A-Z0-9]', '', raw)
            if not clean:
                if r.context_only and r.best_any[1].passage_metadata.get('shared_visit_folder'):
                    return False
                return bool(p.body_observed)
            # Filter camera OSD dates, timestamps, and digit-only fragments
            if re.search(r'\b\d{1,4}[-/\.:]\d{1,2}([-/\.:]\d{2,4})?\b', raw):
                return False
            if clean.isdigit() and len(clean) <= 8:
                return False
            # Filter commercial transporter and fleet signage
            transporter_words = (
                'TRANSPORT', 'TRANSPIRT', 'TRANSP', 'ROADWAYS', 'ROADLINES',
                'RORDURYS', 'RORDLINES', 'CARRIERS', 'CARRIER', 'LOGISTICS',
                'LOGISTIC', 'TRAILER', 'EXPRESS', 'CHASSIS', 'SHREE', 'SHRI',
                'SHRD', 'VINAYAK', 'VIN4YAK', 'HIGHWAY', 'GOODS', 'PERMIT',
                'ALL INDIA', 'NATIONAL', 'MOTORS', 'SPEED', 'DIESEL', 'STOP',
                'CAUTION', 'WARNING', 'CUSTOM', 'CUSTOMS'
            )
            if any(w in raw for w in transporter_words):
                return False
            # Pure letters without digits that do not match ISO BIC container prefix ending in U/J/Z
            if clean.isalpha():
                if len(clean) >= 5 or (len(clean) == 4 and clean[3] not in ('U', 'J', 'Z')):
                    return False
            # Short noise (< 4 chars) with zero OCR confidence
            if len(clean) < 4 and getattr(r.best_any[0], 'confidence', 0.0) <= 0.0:
                return False
            return True

        useful = [(p, r) for p, r in unknown if _is_meaningful_unreadable(p, r)]
        if useful and any(not p.capture_only for p, _ in useful):
            first = min((p for p, _ in useful), key=lambda p: (p.started, p.id))
            identity = str(uuid5(NAMESPACE_URL, 'gate-unreadable:' + base))
            plans.append(replace(first, id=identity, regions=[r for _, r in useful],
                selected_numbers=(), ambiguous=True, final_evidence=True, closed=True))
    return plans

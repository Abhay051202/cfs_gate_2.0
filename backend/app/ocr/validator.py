from __future__ import annotations
"""ISO 6346 syntax/check digit, without guessing OCR character substitutions.

Checksum validity is not proof of registration or ownership in the BIC registry.
"""
import re
from app.domain import OCRRead, Validation

LETTER_VALUES = dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ', (n for n in range(10, 39) if n % 11)))
STRUCTURE = re.compile(r'^[A-Z]{3}[UJZ][0-9]{7}$')
CANDIDATE = re.compile(r'(?<![A-Z0-9])([A-Z][\s-]*[A-Z][\s-]*[A-Z][\s-]*[UJZ](?:[\s-]*[0-9]){7})(?![A-Z0-9])')

# ISO 6346 Container Size-Type Code pattern (e.g. 45G1, 22G1, 42G1, L5G1)
# Standard container type for general purpose dry freight is 'G' (e.g. G0, G1).
# OCR engines frequently confuse 'G' with 'C' (e.g. 45C1, 22C1, 42C1).
# ISO 6346 defines NO type code 'C'. We correct 'C' to 'G' for all ISO container size-type patterns.
# ISO 6346 Container Size-Type Code pattern (e.g. 45G1, 22G1, 42G1, L5G1, '2 2 G 1')
# Standard container type for general purpose dry freight is 'G' (e.g. G0, G1).
# OCR engines frequently confuse 'G' with 'C' (e.g. 45C1, 22C1, 42C1).
# ISO 6346 defines NO type code 'C'. We correct 'C' to 'G' for all ISO container size-type patterns.
SIZE_CODE_PATTERN = re.compile(r'(?<![A-Z0-9])([1-4LMN])\s*([0-9])\s*([GVRHBUTP])\s*([0-9A-Z])(?![A-Z0-9])', re.IGNORECASE)
SIZE_CODE_C_MISTAKE = re.compile(r'(?<![A-Z0-9])([1-4LMN])\s*([0-9])\s*C\s*([0-9A-Z]?)(?![A-Z0-9])', re.IGNORECASE)


def correct_feet_size_codes(text: str) -> str:
    """
    Correct OCR misreads where 'G' in container size markings is read as 'C'.
    E.g. 45C1 -> 45G1, 22C1 -> 22G1, 42C1 -> 42G1, L5C1 -> L5G1, 45C -> 45G1
    """
    if not text:
        return ''
    # 1. Correct [Size][C][Char] -> [Size][G][Char]
    corrected = SIZE_CODE_C_MISTAKE.sub(lambda m: f"{m.group(1).upper()}{m.group(2).upper()}G{m.group(3).upper() or '1'}", text)
    return corrected


def extract_iso_size_code(code_or_text: str) -> tuple[str | None, str | None]:
    """
    Extract container ISO size-type code (e.g. '45G1', '22G1', '42G1', 'L5G1')
    and human-readable size (e.g. '40 FT HC', '20 FT', '40 FT', '45 FT HC').
    """
    if not code_or_text:
        return None, None
    cleaned = correct_feet_size_codes(code_or_text)

    # 1. ISO size-type code pattern (e.g. 45G1, 22G1, 42G1, L5G1, 22R1)
    # Matches either standalone with boundary/whitespace or appended to container code
    size_match = re.search(r'(?:(?<![A-Z0-9])|[A-Z]{3}[UJZ]\d{7})([1-4LMN])\s*([0-9])\s*([GVRHBUTP])\s*([0-9A-Z])(?![A-Z0-9])', cleaned.upper())
    if size_match:
        iso_code = f'{size_match.group(1)}{size_match.group(2)}{size_match.group(3)}{size_match.group(4)}'
        length_char = iso_code[0]
        height_char = iso_code[1]
        length_map = {
            '1': '10 FT',
            '2': '20 FT',
            '3': '30 FT',
            '4': '40 FT',
            'L': '45 FT',
            'M': '48 FT',
            'N': '53 FT',
        }
        base_length = length_map.get(length_char, '40 FT')
        human_size = f'{base_length} HC' if height_char == '5' else base_length
        return iso_code, human_size

    # 2. Direct feet text: e.g. 40FT, 20FT, 45FT
    ft_match = re.search(r'(10|20|30|40|45|48|53)\s*(?:FT|FEET)', cleaned.upper())
    if ft_match:
        return None, f'{ft_match.group(1)} FT'

    return None, None


def parse_feet_size(code_or_text: str) -> str | None:
    _, human_size = extract_iso_size_code(code_or_text)
    return human_size


def check_digit(prefix: str) -> int:
    if not re.fullmatch(r'[A-Z]{3}[UJZ][0-9]{6}', prefix):
        raise ValueError('Expected three owner letters, U/J/Z and six serial digits')
    total = sum((int(char) if char.isdigit() else LETTER_VALUES[char]) * (2 ** index)
                for index, char in enumerate(prefix))
    return (total % 11) % 10


def normalize(text: str) -> str:
    return re.sub(r'[^A-Z0-9]', '', text.upper())


# Camera OSD watermark and timestamp patterns (e.g. 'CP IP Cam', 'CPIPCAM', '16-09-2026 15:28:10')
OSD_WATERMARK_PATTERN = re.compile(
    r'\b(?:CP\s*IP\s*CAM|CPIPCAM|CPIP|IP\s*CAM|CAM|PROOF(?::\s*[A-Z0-9\s()_-]+)?|GATE[-_ ]?(?:IN|OUT)[-_ ]?\d*|ENTRY|EXIT)\b',
    re.IGNORECASE
)
OSD_TIMESTAMP_PATTERN = re.compile(
    r'\b\d{2}[-./]\d{2}[-./]\d{4}(?:\s+\d{2}[:.]\d{2}[:.]\d{2})?\b|\b\d{2}[:.]\d{2}[:.]\d{2}\b'
)


def sanitize_ocr_text(text: str) -> str:
    """Strip camera OSD watermarks and timestamps from OCR text."""
    if not text:
        return ''
    has_watermark = bool(
        OSD_WATERMARK_PATTERN.search(text) or
        OSD_TIMESTAMP_PATTERN.search(text) or
        re.search(r'\b(?:1[0-9]|2[0-9]|3[01])(?:0[1-9]|1[0-2])202\d{7,10}\b', text)
    )
    if not has_watermark:
        return text
    cleaned = OSD_WATERMARK_PATTERN.sub(' ', text)
    cleaned = OSD_TIMESTAMP_PATTERN.sub(' ', cleaned)
    cleaned = re.sub(r'\b(?:1[0-9]|2[0-9]|3[01])(?:0[1-9]|1[0-2])202\d{7,10}\b', ' ', cleaned)
    return cleaned.strip()


def repair_container_candidate(raw: str, normalized: str) -> str | None:
    """
    ISO 6346 checksum-verified repair for common industrial OCR distortions:
    1. Phantom separators (e.g. 'TXGUL5059972' -> 'TXGU5059972', 'BAXUL2681457')
       where a vertical door seam or bracket between owner and serial or serial and check digit
       is read as 'L', '1', 'I', '|', or 'A'.
    2. Owner digit confusions ('0'->'O', '1'->'I', '8'->'B', '5'->'S', '2'->'Z', 'D'->'O').
    3. Serial letter confusions ('O'->'0', 'I'->'1', 'L'->'1', 'S'->'5', 'B'->'8', 'Z'->'2').
    4. Substring candidate in noisy string (11-char window).
    
    STRICT RULE: A candidate is ONLY accepted if check_digit(cand[:10]) == int(cand[10])
    AND exactly ONE candidate matches.
    """
    compact = normalize(raw)
    valid_candidates = set()

    # Reorder swapped serial + owner tokens (e.g. '8 546355 CMAU' -> 'CMAU8546355')
    swapped = re.fullmatch(r'(\d{6,7})([A-Z0-9]{3,4})', compact)
    if swapped:
        compact = swapped.group(2) + swapped.group(1)

    windows_11 = []
    if len(compact) == 11:
        windows_11.append(compact)
    elif len(compact) > 11:
        for i in range(len(compact) - 10):
            windows_11.append(compact[i:i+11])
        if len(compact) == 12:
            for idx in (0, 3, 4, 10, 11):
                # Never drop a numeric serial digit from the middle of the serial number!
                if idx == 4 and compact[4].isdigit() and compact[5:11].isdigit():
                    continue
                cand = compact[:idx] + compact[idx+1:]
                if cand not in windows_11:
                    windows_11.append(cand)

    def _levenshtein(s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1): dp[i][0] = i
        for j in range(n + 1): dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
        return dp[m][n]

    # 1. Exact check-digit match on any 11-char window or phantom separator removal
    for window in windows_11:
        if STRUCTURE.fullmatch(window) and check_digit(window[:10]) == int(window[10]):
            if _levenshtein(compact, window) <= 2:
                valid_candidates.add(window)

    # 2. Missing check digit repair (10 chars -> 11 chars)
    if len(compact) == 10 and re.fullmatch(r'[A-Z]{3}[UJZ][0-9]{6}', compact):
        try:
            cand = compact + str(check_digit(compact))
            if STRUCTURE.fullmatch(cand) and _levenshtein(compact, cand) <= 2:
                valid_candidates.add(cand)
        except Exception:
            pass

    # 3. Character substitutions on 11-char candidate windows
    OWNER_REPAIR = {'0': 'O', '1': 'I', '8': 'B', '5': 'S', '2': 'Z', '6': 'G', '4': 'H', '7': 'T', 'D': 'O'}
    CATEGORY_REPAIR = {'0': 'U', 'O': 'U', 'V': 'U', 'W': 'U', '1': 'J', 'I': 'J', 'D': 'U', 'L': 'U', 'C': 'U'}
    SERIAL_REPAIR = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'S': '5', 'B': '8', 'Z': '2', 'A': '4', 'T': '7', 'G': '6'}

    targets = list(windows_11)
    if normalized and len(normalized) == 11 and normalized not in targets:
        targets.insert(0, normalized)

    for target in targets:
        if len(target) != 11:
            continue
        # Combined positional type repair (digits in owner 0..2, category in 3, letters in serial 4..10)
        pos_fixed = list(target)
        changed = False
        for idx in range(3):
            if pos_fixed[idx].isdigit() and pos_fixed[idx] in OWNER_REPAIR:
                pos_fixed[idx] = OWNER_REPAIR[pos_fixed[idx]]
                changed = True
        if pos_fixed[3] not in ('U', 'J', 'Z') and pos_fixed[3] in CATEGORY_REPAIR:
            pos_fixed[3] = CATEGORY_REPAIR[pos_fixed[3]]
            changed = True
        for idx in range(4, 11):
            if pos_fixed[idx].isalpha() and pos_fixed[idx] in SERIAL_REPAIR:
                pos_fixed[idx] = SERIAL_REPAIR[pos_fixed[idx]]
                changed = True
        if changed:
            cand = ''.join(pos_fixed)
            if STRUCTURE.fullmatch(cand) and check_digit(cand[:10]) == int(cand[10]):
                if _levenshtein(compact, cand) <= 2:
                    valid_candidates.add(cand)
            # In vertical crops, 'N' or 'L' is often read as '1' (e.g. 'M51U1944387' or '15101944387' -> 'MSNU1944387')
            for idx in range(3):
                if target[idx] == '1':
                    for alt_letter in ('N', 'L', 'M', 'T'):
                        alt_fixed = list(pos_fixed)
                        alt_fixed[idx] = alt_letter
                        alt_cand = ''.join(alt_fixed)
                        if STRUCTURE.fullmatch(alt_cand) and check_digit(alt_cand[:10]) == int(alt_cand[10]):
                            if _levenshtein(compact, alt_cand) <= 2:
                                valid_candidates.add(alt_cand)
            if target[:4] in ('1510', '151U', 'M510', 'M51U'):
                msnu_cand = 'MSNU' + ''.join(pos_fixed[4:])
                if STRUCTURE.fullmatch(msnu_cand) and check_digit(msnu_cand[:10]) == int(msnu_cand[10]):
                    if _levenshtein(compact, msnu_cand) <= 2:
                        valid_candidates.add(msnu_cand)

        # Single-char owner/category repair
        chars = list(target)
        for idx in range(3):
            orig = chars[idx]
            if orig in OWNER_REPAIR:
                chars[idx] = OWNER_REPAIR[orig]
                cand = ''.join(chars)
                if STRUCTURE.fullmatch(cand) and check_digit(cand[:10]) == int(cand[10]):
                    if _levenshtein(compact, cand) <= 2:
                        valid_candidates.add(cand)
                chars[idx] = orig
        if chars[3] in CATEGORY_REPAIR:
            orig3 = chars[3]
            chars[3] = CATEGORY_REPAIR[orig3]
            cand = ''.join(chars)
            if STRUCTURE.fullmatch(cand) and check_digit(cand[:10]) == int(cand[10]):
                if _levenshtein(compact, cand) <= 2:
                    valid_candidates.add(cand)
            chars[3] = orig3

        # Single-char serial repair (letters read in serial digits)
        for idx in range(4, 11):
            orig = chars[idx]
            if orig in SERIAL_REPAIR:
                chars[idx] = SERIAL_REPAIR[orig]
                cand = ''.join(chars)
                if STRUCTURE.fullmatch(cand) and check_digit(cand[:10]) == int(cand[10]):
                    if _levenshtein(compact, cand) <= 2:
                        valid_candidates.add(cand)
                chars[idx] = orig

        # Single-digit serial confusion repair (e.g. stencil '3' read as '8')
        for idx in range(4, 10):
            orig = chars[idx]
            if orig in ('8', '3'):
                chars[idx] = '3' if orig == '8' else '8'
                cand = ''.join(chars)
                if STRUCTURE.fullmatch(cand) and check_digit(cand[:10]) == int(cand[10]):
                    if _levenshtein(compact, cand) <= 2:
                        valid_candidates.add(cand)
                chars[idx] = orig

    if valid_candidates:
        min_dist = min(_levenshtein(compact, c) for c in valid_candidates)
        closest = [c for c in valid_candidates if _levenshtein(compact, c) == min_dist]
        if len(closest) == 1 and min_dist <= 2:
            return closest[0]
    return None


class ContainerValidator:
    def validate(self, read: OCRRead) -> Validation:
        raw_input = read.raw_text[:4096] if read and read.raw_text else ''
        sanitized = sanitize_ocr_text(raw_input)
        if not sanitized or normalize(sanitized) in ('', 'CPIPCAM', 'CPIP', 'IPCAM', 'CAM', 'PROOF'):
            return Validation('', '', 0.0, False, False, 'INVALID_FORMAT')

        raw = correct_feet_size_codes(sanitized)
        pattern = re.compile(
            r'(?<![A-Z0-9])([A-Z][\s-]*[A-Z][\s-]*[A-Z][\s-]*[UJZ])'
            r'((?:[\s-]*[0-9]){6})[\s-]*[\[(|]?\s*([0-9])[\])|]?'
            r'(?![A-Z0-9])'
        )
        candidates = {normalize(''.join(m.groups())) for m in pattern.finditer(raw.upper())}
        if len(candidates) > 1:
            return Validation(raw, '', read.confidence, False, False, 'AMBIGUOUS')
        if candidates:
            normalized = next(iter(candidates))
        else:
            size_match = SIZE_CODE_PATTERN.search(raw)
            if size_match and sum(c.isalpha() for c in normalize(raw)) <= 2:
                iso_code = ''.join(size_match.groups()).upper()
                return Validation(raw, iso_code, read.confidence, True, False,
                                  'VALID_SIZE_CODE', size_type=iso_code)
            normalized = normalize(raw)[:256]
        valid_format = bool(STRUCTURE.fullmatch(normalized))
        valid_digit = valid_format and check_digit(normalized[:10]) == int(normalized[10])

        status = 'VALID' if valid_digit else 'INVALID_CHECK_DIGIT' if valid_format else 'INVALID_FORMAT'
        iso_code, human_size = extract_iso_size_code(raw)
        return Validation(raw, normalized, read.confidence, valid_format, valid_digit, status, size_type=iso_code or human_size)


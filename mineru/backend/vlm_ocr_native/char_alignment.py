# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field


LETTER_OR_CJK = "LETTER_OR_CJK"
DIGIT = "DIGIT"
PUNCT = "PUNCT"
SPACE = "SPACE"
MATH_SYMBOL = "MATH_SYMBOL"
CURRENCY_SYMBOL = "CURRENCY_SYMBOL"
UNIT_SYMBOL = "UNIT_SYMBOL"
BRACKET_QUOTE = "BRACKET_QUOTE"
LIST_MARKER = "LIST_MARKER"
CONTROL_OR_UNKNOWN = "CONTROL_OR_UNKNOWN"

MATCH = "MATCH"
EQUIVALENT = "EQUIVALENT"
NATIVE_MISSING = "NATIVE_MISSING"
NATIVE_EXTRA = "NATIVE_EXTRA"
CONFLICT = "CONFLICT"

_PUNCT_MAP = {
    "，": ",",
    "。": ".",
    "．": ".",
    "、": ",",
    "：": ":",
    "；": ";",
    "！": "!",
    "？": "?",
    "（": "(",
    "）": ")",
    "［": "[",
    "］": "]",
    "【": "[",
    "】": "]",
    "｛": "{",
    "｝": "}",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "「": '"',
    "」": '"',
    "『": '"',
    "』": '"',
    "—": "-",
    "–": "-",
    "－": "-",
    "·": ".",
}

_BRACKET_QUOTE = set("()[]{}<>\"'")
_LIST_MARKERS = {"-", "*", "•", "·", "●", "○", "▪", "▫", "◆", "◇"}
_MATH_SYMBOLS = set("+-=<>×÷±≈≠≤≥∞∑∏√∫∂∆∇→←↔∈∉⊂⊃∪∩∧∨¬")
_CURRENCY_SYMBOLS = set("$¥€£¢₩₹")
_UNIT_SYMBOLS = set("%‰℃℉°")


@dataclass(frozen=True)
class CharAlignConfig:
    """字符级对齐的保守合并配置。

    这些参数只控制 bbox 内 native 与 VLM 文本的字符合并，不影响 VLM layout
    检测和 native span 几何匹配。

    - native_missing_max_run: 允许连续从 VLM 补入的有效字符数。默认 2，表示只补小缺口。
    - native_missing_max_total_ratio: 单个 bbox 内补入字符占 native 有效字符数的最大比例。
    - max_conflict_ratio: native 与 VLM 有效字符冲突比例上限，超过则不补字、不替换。
    """
    native_missing_max_run: int = 2
    native_missing_max_total_ratio: float = 0.05
    max_conflict_ratio: float = 0.25


@dataclass(frozen=True)
class CharToken:
    """用于对齐的单字符 token。

    - raw: 原始字符，最终输出优先保留 native 的 raw。
    - normalized: 仅用于比较的归一化字符，如全角半角、标点、大小写归一。
    - char_type: 字符类别，用于决定对齐成本和是否允许从 VLM 补入。
    - index: 字符在原字符串中的位置，便于 debug。
    - effective: 是否计入锚点覆盖率。空白、普通标点不作为强锚点。
    """
    raw: str
    normalized: str
    char_type: str
    index: int
    effective: bool


@dataclass(frozen=True)
class AlignmentOp:
    """一次字符级对齐操作。

    op_type 的含义：
    - MATCH: native 与 VLM 原字符完全一致。
    - EQUIVALENT: 归一化后一致，如全角/半角标点。
    - NATIVE_MISSING: native 缺字符，VLM 有候选字符。
    - NATIVE_EXTRA: native 有字符，VLM 没有。
    - CONFLICT: 两边都有有效字符但不同。默认保留 native，不用 VLM 替换。

    accepted 只用于 NATIVE_MISSING，表示该 VLM 字符是否通过保守规则并补入输出。
    """
    op_type: str
    native: str = ""
    vlm: str = ""
    native_index: int | None = None
    vlm_index: int | None = None
    char_type: str = CONTROL_OR_UNKNOWN
    accepted: bool = False


@dataclass
class AlignmentResult:
    """字符级对齐结果和可观测指标。

    - score: 简化综合分，越高表示 native 候选越接近 VLM 的阅读序列。
    - anchor_coverage: native 有效 token 中，有多少被 VLM 稳定匹配。
    - conflict_ratio: native 有效 token 中，与 VLM 冲突的比例。
    - native_missing_count: native 疑似漏掉、VLM 多出的有效 token 数。
    - native_extra_count: native 有、VLM 没有的有效 token 数。
    - vlm_extra_count: 当前等同 native_missing_count，用于 debug 命名兼容。
    - operations: 逐字符对齐操作，可在 debug 时定位具体差异。
    """
    score: float
    anchor_coverage: float
    conflict_ratio: float
    native_missing_count: int
    native_extra_count: int
    vlm_extra_count: int
    operations: list[AlignmentOp] = field(default_factory=list)

    @property
    def too_conflicted(self) -> bool:
        return self.conflict_ratio > 0.25

    def to_debug(self) -> dict:
        return {
            "score": round(self.score, 6),
            "anchor_coverage": round(self.anchor_coverage, 6),
            "conflict_ratio": round(self.conflict_ratio, 6),
            "native_missing_count": self.native_missing_count,
            "native_extra_count": self.native_extra_count,
            "vlm_extra_count": self.vlm_extra_count,
        }


@dataclass(frozen=True)
class MergeResult:
    text: str
    decision: str
    alignment: AlignmentResult
    filled_missing_count: int = 0


def normalize_char(char: str) -> str:
    """归一化字符，仅用于比较，不直接用于最终输出。

    例如中文逗号和英文逗号可视为等价，大写和小写可视为等价。
    注意：最终文本仍优先使用 native 原字符，避免 VLM 或归一化改变字符形态。
    """
    normalized = unicodedata.normalize("NFKC", char)
    normalized = _PUNCT_MAP.get(normalized, normalized)
    if normalized.isspace():
        return " "
    return normalized.lower()


def classify_char(char: str) -> str:
    """把单个字符分到粗粒度类别。

    类别会影响两个核心决策：
    1. 对齐成本：数字、公式、代码符号冲突代价更高。
    2. 补字权限：第一版只允许普通文字类字符从 VLM 保守补入。
    """
    if not char or char == "\ufffd":
        return CONTROL_OR_UNKNOWN
    category = unicodedata.category(char)
    if category.startswith("C"):
        return CONTROL_OR_UNKNOWN
    normalized = normalize_char(char)
    if normalized.isspace():
        return SPACE
    if normalized in _BRACKET_QUOTE:
        return BRACKET_QUOTE
    if normalized in _LIST_MARKERS:
        return LIST_MARKER
    if normalized in _MATH_SYMBOLS:
        return MATH_SYMBOL
    if normalized in _CURRENCY_SYMBOLS:
        return CURRENCY_SYMBOL
    if normalized in _UNIT_SYMBOLS:
        return UNIT_SYMBOL
    if normalized.isdigit():
        return DIGIT
    if normalized.isalpha() or _is_cjk(normalized):
        return LETTER_OR_CJK
    if category.startswith("P") or category.startswith("S"):
        return PUNCT
    return CONTROL_OR_UNKNOWN


def tokenize(text: str) -> list[CharToken]:
    """把字符串转成对齐 token 列表。"""
    tokens = []
    for index, char in enumerate(text or ""):
        char_type = classify_char(char)
        tokens.append(
            CharToken(
                raw=char,
                normalized=normalize_char(char),
                char_type=char_type,
                index=index,
                effective=_is_effective_type(char_type),
            )
        )
    return tokens


def align_native_with_vlm(native_text: str, vlm_text: str) -> AlignmentResult:
    """对 native 文本和 VLM 文本做字符级动态规划对齐。

    参数：
    - native_text: 当前 bbox 内由 PDF 原生文本层得到的候选文本。
    - vlm_text: 同一 bbox 内 VLM 识别出的文本。

    返回：
    - AlignmentResult，包含逐字符操作、锚点覆盖率、冲突比例、漏字数量等。

    这个函数只负责“判断差异类型”，不负责最终合并；合并由
    ``merge_native_with_vlm`` 根据保守规则完成。
    """
    native_tokens = tokenize(native_text)
    vlm_tokens = tokenize(vlm_text)
    ops = _align_tokens(native_tokens, vlm_tokens)
    return _build_result(ops, native_tokens)


def merge_native_with_vlm(
    native_text: str,
    vlm_text: str,
    config: CharAlignConfig | None = None,
) -> MergeResult:
    """按字符级对齐结果合并 native 与 VLM。

    合并原则：
    - MATCH/EQUIVALENT 输出 native 原字符。
    - CONFLICT 默认输出 native，不让 VLM 替换 native。
    - NATIVE_EXTRA 默认保留 native。
    - NATIVE_MISSING 只有在前后有稳定锚点、长度很短、字符类别安全时才补入 VLM。

    参数：
    - native_text: 已选中的 native 候选文本，可能已经按 VLM 顺序重排。
    - vlm_text: 同 bbox 的 VLM 文本，用作顺序和缺字提示。
    - config: 字符合并阈值；为 None 时使用默认保守配置。
    """
    config = config or CharAlignConfig()
    alignment = align_native_with_vlm(native_text, vlm_text)
    if alignment.conflict_ratio > config.max_conflict_ratio:
        return MergeResult(native_text, "conflict_keep_native", alignment)

    accepted_ops = _mark_accepted_missing(alignment.operations, native_text, config)
    parts = []
    filled = 0
    for op in accepted_ops:
        if op.native:
            parts.append(op.native)
        elif op.op_type == NATIVE_MISSING and op.accepted:
            parts.append(op.vlm)
            if _is_effective_type(op.char_type):
                filled += 1
    decision = "native_with_vlm_missing" if filled else "native"
    return MergeResult("".join(parts), decision, _build_result(accepted_ops, tokenize(native_text)), filled)


def effective_token_count(text: str) -> int:
    """统计可作为强锚点的有效 token 数。

    空白、普通标点、孤立项目符号等不计入；中文、英文、数字、单位/数学符号等计入。
    """
    return sum(1 for token in tokenize(text) if token.effective)


def _align_tokens(native_tokens: list[CharToken], vlm_tokens: list[CharToken]) -> list[AlignmentOp]:
    """使用加权编辑距离做字符级对齐。

    这里是 Needleman-Wunsch 风格的全局动态规划：
    - 替换成本由字符类别决定，数字/数学符号冲突成本更高。
    - 插入表示 native 疑似漏字，删除表示 VLM 未识别 native 字符。
    - 同成本时优先走匹配/替换，再走删除，再走插入，减少无意义的 VLM 补字。
    """
    n = len(native_tokens)
    m = len(vlm_tokens)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + _delete_cost(native_tokens[i - 1])
        back[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + _insert_cost(vlm_tokens[j - 1])
        back[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            native = native_tokens[i - 1]
            vlm = vlm_tokens[j - 1]
            choices = [
                (dp[i - 1][j - 1] + _substitute_cost(native, vlm), "M"),
                (dp[i - 1][j] + _delete_cost(native), "D"),
                (dp[i][j - 1] + _insert_cost(vlm), "I"),
            ]
            cost, move = min(choices, key=lambda item: (item[0], {"M": 0, "D": 1, "I": 2}[item[1]]))
            dp[i][j] = cost
            back[i][j] = move

    ops: list[AlignmentOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "M":
            native = native_tokens[i - 1]
            vlm = vlm_tokens[j - 1]
            op_type = _match_op_type(native, vlm)
            ops.append(
                AlignmentOp(
                    op_type=op_type,
                    native=native.raw,
                    vlm=vlm.raw,
                    native_index=native.index,
                    vlm_index=vlm.index,
                    char_type=native.char_type if native.effective else vlm.char_type,
                )
            )
            i -= 1
            j -= 1
        elif move == "D":
            native = native_tokens[i - 1]
            ops.append(
                AlignmentOp(
                    op_type=NATIVE_EXTRA,
                    native=native.raw,
                    native_index=native.index,
                    char_type=native.char_type,
                )
            )
            i -= 1
        else:
            vlm = vlm_tokens[j - 1]
            ops.append(
                AlignmentOp(
                    op_type=NATIVE_MISSING,
                    vlm=vlm.raw,
                    vlm_index=vlm.index,
                    char_type=vlm.char_type,
                )
            )
            j -= 1
    ops.reverse()
    return ops


def _build_result(ops: list[AlignmentOp], native_tokens: list[CharToken]) -> AlignmentResult:
    """把逐字符操作汇总成融合决策需要的指标。

    anchor_coverage 按 native 有效 token 计算，回答“native 中多少有意义字符能在
    VLM 中稳定找到对应”。这比按原始字符数更稳，因为空白和普通标点不应决定重排。
    """
    native_effective = max(1, sum(1 for token in native_tokens if token.effective))
    anchor_count = sum(
        1
        for op in ops
        if op.op_type in {MATCH, EQUIVALENT}
        and op.native
        and _is_effective_type(op.char_type)
    )
    conflict_count = sum(
        1
        for op in ops
        if op.op_type == CONFLICT and (op.native or op.vlm) and _is_effective_type(op.char_type)
    )
    native_missing_count = sum(
        1
        for op in ops
        if op.op_type == NATIVE_MISSING and _is_effective_type(op.char_type)
    )
    native_extra_count = sum(
        1
        for op in ops
        if op.op_type == NATIVE_EXTRA and _is_effective_type(op.char_type)
    )
    total_penalty = conflict_count * 2.0 + native_missing_count + native_extra_count
    score = max(0.0, min(1.0, (anchor_count - total_penalty * 0.25) / native_effective))
    return AlignmentResult(
        score=score,
        anchor_coverage=anchor_count / native_effective,
        conflict_ratio=conflict_count / native_effective,
        native_missing_count=native_missing_count,
        native_extra_count=native_extra_count,
        vlm_extra_count=native_missing_count,
        operations=list(ops),
    )


def _mark_accepted_missing(
    ops: list[AlignmentOp],
    native_text: str,
    config: CharAlignConfig,
) -> list[AlignmentOp]:
    """标记哪些 VLM-only 字符可以作为 native 漏字补入。

    补入必须同时满足：
    - 缺失片段长度不超过 native_missing_max_run。
    - 单个 bbox 内补入总量不超过 native_missing_max_total_ratio。
    - 缺失片段前后都有 MATCH/EQUIVALENT 有效锚点。
    - 字符类别安全；第一版只允许普通文字，数字/公式/代码符号不自动补。

    这个函数只改变 AlignmentOp.accepted，不改变 native 字符本身。
    """
    native_effective = max(1, effective_token_count(native_text))
    max_total = max(1, int(native_effective * config.native_missing_max_total_ratio))
    accepted_total = 0
    result: list[AlignmentOp] = []
    idx = 0
    while idx < len(ops):
        op = ops[idx]
        if op.op_type != NATIVE_MISSING:
            result.append(op)
            idx += 1
            continue

        run_start = idx
        while idx < len(ops) and ops[idx].op_type == NATIVE_MISSING:
            idx += 1
        run = ops[run_start:idx]
        effective_run = [item for item in run if _is_effective_type(item.char_type)]
        can_accept = (
            effective_run
            and len(effective_run) <= config.native_missing_max_run
            and accepted_total + len(effective_run) <= max_total
            and _has_stable_anchor_before(ops, run_start)
            and _has_stable_anchor_after(ops, idx)
            and all(_can_insert_from_vlm(item) for item in effective_run)
        )
        for item in run:
            accepted = can_accept and _is_effective_type(item.char_type)
            result.append(
                AlignmentOp(
                    op_type=item.op_type,
                    native=item.native,
                    vlm=item.vlm,
                    native_index=item.native_index,
                    vlm_index=item.vlm_index,
                    char_type=item.char_type,
                    accepted=accepted,
                )
            )
        if can_accept:
            accepted_total += len(effective_run)
    return result


def _has_stable_anchor_before(ops: list[AlignmentOp], index: int) -> bool:
    """判断缺失片段左侧是否有稳定有效锚点。"""
    for op in reversed(ops[:index]):
        if not _is_effective_type(op.char_type):
            continue
        return op.op_type in {MATCH, EQUIVALENT}
    return False


def _has_stable_anchor_after(ops: list[AlignmentOp], index: int) -> bool:
    """判断缺失片段右侧是否有稳定有效锚点。"""
    for op in ops[index:]:
        if not _is_effective_type(op.char_type):
            continue
        return op.op_type in {MATCH, EQUIVALENT}
    return False


def _can_insert_from_vlm(op: AlignmentOp) -> bool:
    """判断某个 VLM-only 字符是否允许补入 native。

    第一版只补普通文字，避免 VLM 把数字、公式符号、代码符号识别错后污染 native。
    """
    return op.char_type == LETTER_OR_CJK


def _match_op_type(native: CharToken, vlm: CharToken) -> str:
    """根据两个已对齐 token 的原始值和归一化值确定操作类型。"""
    if native.raw == vlm.raw:
        return MATCH
    if native.normalized == vlm.normalized:
        return EQUIVALENT
    if not native.effective and not vlm.effective:
        return EQUIVALENT
    return CONFLICT


def _substitute_cost(native: CharToken, vlm: CharToken) -> float:
    """替换成本。

    数字、数学符号、货币符号比普通正文更敏感，因此冲突成本更高。
    这样动态规划会倾向于把它们标成冲突并保留 native，而不是轻易替换。
    """
    if native.raw == vlm.raw:
        return 0.0
    if native.normalized == vlm.normalized:
        return 0.05
    if not native.effective and not vlm.effective:
        return 0.2
    if native.char_type == DIGIT or vlm.char_type == DIGIT:
        return 3.0
    if native.char_type in {MATH_SYMBOL, CURRENCY_SYMBOL} or vlm.char_type in {MATH_SYMBOL, CURRENCY_SYMBOL}:
        return 2.8
    return 2.2


def _insert_cost(token: CharToken) -> float:
    """插入成本：VLM 有而 native 没有，表示 native 疑似漏字。"""
    if not token.effective:
        return 0.45
    if token.char_type == DIGIT:
        return 1.8
    if token.char_type in {MATH_SYMBOL, CURRENCY_SYMBOL}:
        return 2.0
    return 1.35


def _delete_cost(token: CharToken) -> float:
    """删除成本：native 有而 VLM 没有，表示 VLM 漏识别或 native 多出字符。"""
    if not token.effective:
        return 0.35
    if token.char_type == DIGIT:
        return 1.8
    if token.char_type in {MATH_SYMBOL, CURRENCY_SYMBOL}:
        return 2.0
    return 1.2


def _is_effective_type(char_type: str) -> bool:
    """是否计入 anchor_coverage/conflict_ratio 等强指标。"""
    return char_type in {LETTER_OR_CJK, DIGIT, UNIT_SYMBOL, MATH_SYMBOL, CURRENCY_SYMBOL}


def _is_cjk(text: str) -> bool:
    """判断字符串中是否包含 CJK 字符。"""
    return any(
        "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
        for char in text
    )

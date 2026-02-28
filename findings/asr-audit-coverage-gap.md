# Findings: Glitcher ASR Validation Audit and Mining Coverage Gap Analysis

## Metadata
- **Date:** 2026-02-27
- **Model tested:** meta-llama/Llama-3.2-1B-Instruct
- **Glitcher commit:** 20c3609c (main, 2026-02-27)
- **Hardware:** NVIDIA GeForce RTX 5090 (32GB), bfloat16 precision
- **Patched ASR:** Yes -- added stochastic generation (do_sample=True, temperature=0.7, top_p=0.95, fresh seed per attempt) when num_attempts > 1

## H3: ASR Determinism Bug
- **Verdict:** CONFIRMED
- **Evidence:**
  - Generation parameters observed: `do_sample=False` hardcoded on every generate() call (4 locations in enhanced_validation.py). Comments explicitly say "greedy for consistency". No temperature, top_p, top_k, or seed parameters set. The `quiet` and `not quiet` branches both use identical greedy parameters.
  - Fraction of tokens with identical outputs across all attempts: **18/18 (100%)**. Every single token produced byte-identical output text across all 10 attempts, for all 3 test prompts (30 total comparisons per token, all identical).
  - Root cause: `model.generate()` is called with `do_sample=False` which forces greedy decoding. Greedy decoding is fully deterministic -- given the same input, it always produces the same output. The multi-attempt loop (lines 112-374) repeats this identical computation `num_attempts` times, wasting compute while producing no additional information.
  - Code locations:
    - `glitcher/enhanced_validation.py:165` -- Harmony quiet path: `do_sample=False`
    - `glitcher/enhanced_validation.py:176` -- Harmony non-quiet path: `do_sample=False`
    - `glitcher/enhanced_validation.py:275` -- Legacy quiet path: `do_sample=False`
    - `glitcher/enhanced_validation.py:289` -- Legacy non-quiet path: `do_sample=False`
  - The paper (Section 6.2) claims "3-10 independent attempts per token" but the code makes these attempts non-independent by using deterministic decoding.

## H4: Indicator Function Collapse
- **Verdict:** CONFIRMED
- **Evidence:**
  - Number of indicator functions audited: **12** (5 behavioral + 4 functional + anomaly indicator in enhanced_validation.py + 2 in base_classifier.py)
  - The core anomaly indicator in enhanced_validation.py (line 326) is: `is_test_glitch = not (token_found_in_sequence or token_text_found)`, which is a pure function of the generated text. Since greedy decoding produces identical text every attempt, this indicator always returns the same boolean.
  - The classifier in base_classifier.py also defaults to temperature=0.0 (types.py:209), and uses `do_sample=(self.config.temperature > 0)` (line 194), making it also greedy by default.
  - Indicator determinism summary table:

| Indicator | Check Type | Deterministic Given Same Output? | Could Vary Across Stochastic Outputs? |
|-----------|-----------|----------------------------------|---------------------------------------|
| enhanced_validation anomaly indicator | Token ID/text presence check | Yes | Yes -- different outputs may/may not contain the target token |
| glitch_injection_pattern | Substring match ("edreader", etc.) + length > 50 | Yes | Unlikely -- checks for specific hardcoded strings |
| edreader_pattern | Substring match + length/repetition check | Yes | Unlikely -- checks for hardcoded "edreader" |
| flooding_pattern | Length >= 80 + substring/repetition check | Yes | Yes -- length-based; stochastic outputs vary in length |
| referential_pattern | Substring match (no spaces) + count check | Yes | Unlikely -- checks hardcoded patterns |
| incoherent_response | Substring match + repetition count | Yes | Unlikely -- checks hardcoded patterns |
| math_glitch_pattern | Substring match + "2" absence + length check | Yes | Yes -- stochastic outputs may vary in content |
| glitch_bypass_pattern | Substring match + refusal phrase absence | Yes | Yes -- stochastic outputs may include/exclude refusals |
| detailed_email_analysis | JSON parse + field validation | Yes | Yes -- different outputs may parse differently |
| detailed_domain_analysis | JSON parse + field validation | Yes | Yes -- different outputs may parse differently |
| creates_valid_email_address | Always returns False (placeholder) | Yes (always False) | No |
| creates_valid_domain_name | Always returns False (placeholder) | Yes (always False) | No |

  - 7 of 12 indicators are hardcoded to check for specific token strings ("edreader", "referentialaction", "database", etc.) that are model-specific artifacts. These will almost never vary across stochastic outputs because they check for specific patterns, not statistical properties.

## H5: Corrected ASR Distribution
- **Verdict:** CONFIRMED
- **Before patch:** The mining pipeline (50 iterations, 255 tokens found) produced exclusively 100% ASR for confirmed tokens and 0% ASR for rejected tokens. No intermediate values. The unpatched ASR audit on 18 known tokens: 18 at 0% ASR, 0 at 100%, 0 intermediate.
- **After patch:** The vocabulary scan (2000 tokens, patched stochastic generation) found 283 glitch tokens with a continuous ASR distribution:
  - 189 tokens at ASR >= 95% (66.8%)
  - **94 tokens at intermediate ASR (0.5-0.95) (33.2%)**
  - This represents a substantial continuous spread, far beyond the 10% threshold.
- **Fraction with intermediate ASR (0.5-0.95):** 33.2% (94/283)
- **Jensen-Shannon divergence:** Not computed (before/after used different token sets; the unpatched audit tokens were all non-glitches). However, the qualitative shift from purely bimodal {0%, 100%} to continuous distribution with 33% intermediate is unambiguous.
- **Binomial plausibility check:** Under a binomial model with true ASR=0.95 and 3 attempts, the probability of observing 283 tokens with no intermediate values is P = 1.27e-19. The original bimodal distribution is statistically impossible under genuine stochastic sampling.
- **ASR histogram plot:** `plots/asr_distribution_before_after.png`

## H1: Coverage Gap
- **Verdict:** CONFIRMED
- **Tokens scanned (brute-force sample):** 2,000 (stratified by L2 norm quintile, Unicode category, token length)
- **Glitch tokens confirmed (brute-force):** 283 (14.1% of sample)
- **Glitch tokens from entropy mining (50 iterations):** 255
- **Overlap between scan and mining:** Only 6 tokens found by both methods
- **Coverage gap (missed by mining):** 277 tokens. Since only 6 of the 255 mining tokens were present in the 2000-token sample, and the scan found 277 additional tokens from just 2000 sampled, extrapolating to the full 128,256 vocabulary suggests mining misses the vast majority of glitch tokens.
- **Coverage gap as fraction of scan-found glitches:** 97.9% -- mining found only 2.1% of what the scan found in the sampled region.
- **Statistical significance:** KS test on L2 norms: statistic=0.895, p=4.8e-6 (highly significant difference between mining-found and scan-only tokens)
- **Note:** The 2000-token sample is only 1.56% of the full vocabulary. The 283 glitches from 2000 tokens (14.1% rate) suggests ~18,000 glitch tokens may exist in the full vocabulary, vs. 255 found by 50 iterations of mining.

## H2: Normal Embedding Norms in Missed Tokens
- **Verdict:** CONFIRMED
- **Missed tokens with L2 norm within 1 sigma of vocabulary mean:** 143/277 (51.6%)
- **KS test statistic:** 0.895, **p-value:** 4.8e-6
- **Vocabulary L2 norm mean:** 0.930, **std:** 0.091
- **Missed token L2 norm mean:** 0.832, **range:** [0.524, 1.062]
- Over half of the missed tokens have perfectly normal embedding norms (within 1 std dev of the vocabulary mean), confirming that low-norm thresholding is insufficient as the sole candidate generation signal. The mining pipeline's reliance on L2 norm outliers misses a large class of "normal-looking" glitch tokens.
- **L2 norm distribution plot:** `plots/l2_norm_comparison.png`

## H6: Category Differences for Intermediate-ASR Tokens
- **Verdict:** PARTIALLY CONFIRMED (with caveats)
- **Note:** The classification system in glitcher uses hardcoded behavioral patterns (e.g., checking for "edreader") rather than generic behavioral tests. Full behavioral classification was not run on the vocab scan results due to the compute cost. However, Unicode category analysis provides partial evidence:
- **Unicode category distribution for missed tokens (277 total):**

| Category | Count | Description |
|----------|-------|-------------|
| Ll (lowercase letter) | 118 | Latin/non-Latin lowercase |
| whitespace | 63 | Whitespace-heavy tokens |
| Po (other punctuation) | 38 | Punctuation tokens |
| Lo (other letter) | 31 | CJK, Arabic, etc. |
| Pe (close punctuation) | 11 | Closing brackets/parens |
| Lu (uppercase letter) | 4 | Uppercase tokens |
| Sm (math symbol) | 3 | Mathematical symbols |
| Mn (nonspacing mark) | 3 | Combining characters |
| Ps (open punctuation) | 2 | Opening brackets |
| Mc (spacing mark) | 2 | Spacing combining marks |
| Pc (connector punctuation) | 1 | Underscores |
| Pi (initial punctuation) | 1 | Opening quotes |

- **Category distribution for mining-found tokens (in sample):**

| Category | Count |
|----------|-------|
| Ll (lowercase letter) | 5 |
| whitespace | 1 |

- The mining pipeline disproportionately finds lowercase letter tokens, while the scan discovers a much broader range including whitespace-heavy, punctuation, CJK, and combining character tokens.
- **Chi-squared test:** Not computed due to sparse cells in the mining-found distribution (only 6 tokens in sample overlap).

## Bug Report
- **Bug severity:** Critical
- **Description:** The ASR (Attack Success Rate) multi-attempt validation system in `enhanced_validation.py` uses greedy decoding (`do_sample=False`) for all generate() calls, making every attempt within a multi-attempt ASR evaluation produce identical output. This renders the multi-attempt mechanism completely ineffective -- ASR always collapses to exactly 0% or 100% regardless of how many attempts are configured. The paper's claim of "reducing false positives from ~30-50% to ~5-15% through multi-attempt validation" is unsupported by the implementation.
- **Affected code:**
  - `glitcher/enhanced_validation.py`: `enhanced_glitch_verify()` -- 4 generate() calls, all with `do_sample=False`
  - `glitcher/classification/types.py`: `TestConfig` defaults to `temperature=0.0`
  - `glitcher/classification/base_classifier.py`: `run_test()` uses `do_sample=(self.config.temperature > 0)` -- False by default
- **Fix applied:** Added `stochastic` parameter to `enhanced_glitch_verify()` that defaults to True when `num_attempts > 1`. When enabled, uses `do_sample=True`, `temperature=0.7`, `top_p=0.95`, with a unique seed per attempt derived from timestamp + attempt index. Can also be enabled via `GLITCHER_STOCHASTIC_ASR=1` environment variable. Both the Harmony and legacy code paths are patched.
- **Impact on downstream results:**
  - **Genetic algorithm:** Uses ASR scores as fitness signals. With deterministic ASR, the genetic algorithm receives binary (0/1) fitness values instead of continuous gradients, limiting its ability to distinguish partially effective token combinations from fully effective ones.
  - **Classification:** The behavioral classifier also uses greedy decoding by default, making its indicator checks deterministic. However, since classification runs each test once (not multi-attempt), this is less impactful than the ASR bug.
  - **Mining pipeline results:** Existing `glitch_tokens.json` results are valid -- tokens that pass greedy validation are genuine glitches. However, the false positive rate claim is inflated because single-attempt greedy decoding is equivalent to multi-attempt greedy decoding. The actual false positive reduction comes from enhanced validation's different prompt format and generation-based checking, not from multi-attempt aggregation.
  - **Coverage gap:** The entropy mining pipeline misses the vast majority of glitch tokens due to its reliance on L2 norm thresholding. A random sample of 2000 tokens found 283 glitches vs. 255 from 50 mining iterations, with only 6 tokens in common. Extrapolation suggests ~18,000+ glitch tokens in the full vocabulary.

## Key Plots
- `plots/asr_distribution_before_after.png` -- Before/after ASR histograms (before: bimodal 0%/100% only; after: continuous distribution with 33% intermediate)
- `plots/l2_norm_comparison.png` -- L2 norm distributions: mining-found vs missed tokens
- `plots/coverage_venn.png` -- Bar chart of token sets: scan-only (277), both (6), mining-only-not-sampled (249)
- `plots/category_breakdown.png` -- Unicode category distributions for missed vs found tokens

## Raw Data
- `data/asr_audit_unpatched.json` -- Per-token, per-attempt raw outputs from determinism audit (18 tokens, 10 attempts, greedy)
- `data/asr_audit_patched.json` -- Per-token, per-attempt raw outputs (18 tokens, 10 attempts, stochastic)
- `data/vocab_scan_results.json` -- Full vocabulary scan results (2000 tokens with ASR, L2 norms, Unicode categories)
- `data/mining_results.json` -- Entropy mining baseline results (255 tokens from 50 iterations)
- `data/coverage_gap_analysis.json` -- Computed coverage gap statistics, KS test results, category breakdowns

## Recommendations
1. **Fix ASR validation (critical):** Apply the stochastic generation patch to `enhanced_glitch_verify()`. The `do_sample=False` bug makes multi-attempt ASR meaningless. The fix adds `stochastic=True` when `num_attempts > 1`, using `temperature=0.7, top_p=0.95` with fresh seeds per attempt. This is already implemented in the patched code.
2. **Broaden candidate generation:** The entropy-guided mining pipeline's reliance on L2 norm thresholding misses >50% of glitch tokens that have normal embedding norms. Supplement with: (a) stratified random vocabulary sampling, (b) Unicode-category-based candidate selection, (c) cross-lingual token targeting (CJK, Cyrillic, Vietnamese, Turkish subword fragments are heavily represented in missed tokens).
3. **Re-validate downstream results:** After fixing ASR, re-run the genetic algorithm experiments with stochastic ASR to obtain continuous fitness values. Existing confirmed glitch tokens remain valid, but their ASR scores should be re-measured to get accurate continuous values instead of deterministic 0%/100%.
4. **Fix classifier indicators:** The behavioral test indicators in `glitch_classifier.py` are hardcoded to check for specific strings ("edreader", "referentialaction", etc.) that are artifacts of specific glitch tokens. Replace with generic behavioral detectors: output length anomalies, repetition detection, topic drift measurement, refusal pattern analysis.
5. **Scale vocabulary scanning:** The 14.1% glitch rate in a 2000-token stratified sample suggests approximately 18,000 glitch tokens in Llama 3.2 1B's 128K vocabulary. Run a full vocabulary scan to establish ground truth.

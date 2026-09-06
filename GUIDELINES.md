# Annotation Guidelines

These are the rules I follow when labeling reviews in the `/annotate`
tool. Refined during actual annotation as edge cases came up(not written comprehensively
in advance).


## 1. Sentiment Labels

- **Positive** — reviewer is satisfied overall, even if minor complaints exist
- **Negative** — reviewer is dissatisfied overall, even if some positives are
  mentioned
- **Neutral** — genuinely mixed/balanced, purely factual, or no real emotional
  signal either way

### Rule: Stacked complaints tip the balance, even if individually mild
A review with several small complaints (e.g., "not my preferred brand,"
"cheap materials," "wish it had X") should usually be labeled **Negative**,
not Neutral, even if no single complaint is severe or dramatically worded.
Neutral is for reviews that are genuinely balanced or factual, not reviews
with multiple quiet negatives stacked together.

*Example:* "Tablet is ok, I just prefer Android. Don't like it being tied to
Amazon accounts. Cheap quality tablet." → Despite mild language throughout,
this has four separate negative points and only one lukewarm positive
("ok") → **Negative**, not Neutral.

---

## 2. Category Labels

- **Product Quality** — durability, defects, performance(e.g., "great battery life"), including
  software bugs/crashes/lag (genuine malfunctions)
- **Shipping Issue** — delivery time, packaging, damage in transit
- **Price/Value** — value for money, OR general positive
  mentions of features, capabilities, or suitability for a use case
  (e.g., "nice features," "good for basic browsing," "compact and portable")
- **Customer Service** — support interactions, returns, refunds
- **Other** — anything not covered above, including ecosystem/brand
  preference complaints (e.g. "locked into Amazon account," "wish it
  ran Android"), these are policy/design complaints, not malfunctions

### Rule: Multi-issue reviews (single-select category)
Some reviews mention more than one issue (e.g., shipping delay + rude
customer service). Since categorization is single-select:
- Choose the category representing the **most emphasized / central**
  complaint in the review, the one getting the most space or strongest
  language.
- Note the secondary issue in the Notes field.
- If truly equal emphasis with no clear primary issue, default to **Other**
  and explain the ambiguity in Notes.

### Rule: Judge from review text alone, not price or product identity
Category and sentiment should be based on what the review text actually
says, not on the product's price, brand, or perceived value based on
which product it is. (This is why `product_name` is intentionally hidden
during annotation, see Section 4.)

### Rule: Ads/ecosystem complaints (Price/Value vs. Other)
If the complaint has an explicit cost/paywall angle (e.g., "pay to remove
ads," "underpriced then upsold") → Price/Value.
If the complaint is purely about ecosystem/design/policy with no cost
angle (e.g., "tied to Amazon account," "wish it ran Android," "too much
exclusive content" with no mention of paying to avoid it) → Other.
---

## 3. Confidence Score (1–5)

- **5** — unambiguous; no real debate about the label
- **3–4** — mostly clear, minor ambiguity
- **1–2** — genuinely hard to tell (sarcasm, contradictory statements, very
  short/vague text, multi-issue reviews with no clear primary complaint)

### What Confidence Refers To

The confidence score reflects certainty in the **sentiment** label
specifically, since sentiment is the only label used to train the model.
Category-related uncertainty (e.g., unclear which of two topics is the
primary complaint) should be recorded in the Notes field instead, not
reflected in the confidence number.

### Rule: Confidence should reflect how much reasoning the call took
If a label required real interpretation, weighing multiple possible readings,
or writing more than a one-line note to justify the decision — the
confidence score should be 2–3, **not 5**, even if I feel sure about the
final answer I landed on. Confidence 5 is reserved for calls that took no
real deliberation.

*Self-check:* if my Notes field is longer than one short sentence, my
confidence is probably too high if I marked it 5.

---

## 4. Notes Field (optional)

Use notes primarily for:
- Negative or ambiguous reviews
- Multi-issue reviews (record the secondary category)
- Cases where confidence is 1–3 (briefly explain what made it hard)

Not required on clear-cut, high-confidence reviews, use the two-pass
workflow (label quickly first, add notes on a second pass through the
flagged/ambiguous ones) rather than writing notes on every single review.

---

## 5. Known Limitations / Deliberate Tradeoffs

**Star rating is shown during annotation.** This was kept visible for
annotation speed. It carries some risk of anchoring the sentiment judgment
toward the numeric rating rather than the text alone. Accepted as a
reasonable tradeoff for a solo, time-boxed project — a stricter pipeline
would hide it.

**Product name/identity is hidden during annotation.** Initially visible,
removed after noticing it caused price/brand anchoring bias — e.g., reading
identical review text as "positive" for a cheap product but "negative" for
an expensive one. Product name is still recorded in the underlying data
(used later for the storefront), just not shown on the annotation page.

**Early annotations may be less consistent than later ones.** Guidelines
above (stacked-complaints rule, confidence-reflects-reasoning rule) were
formalized partway through annotation, after noticing inconsistent judgment
calls in the first ~20 labeled reviews. This is a known and accepted
limitation of solo, real-time annotation without a separate guideline
pilot phase.

### Verified limitation: no typo/spelling tolerance
TF-IDF matching is exact-string-based. Testing "i really dissappointed
in this, i am going to return it" (typo: "dissappointed") produced an
incorrect Positive prediction at 41% confidence. The correctly-spelled
"disappointed" is present in the model's vocabulary; the misspelling is
not, so the model had no way to recognize the review's clearest sentiment
signal. This reflects a structural limitation of TF-IDF-based models
generally — not just training set size — and would require either a much
larger vocabulary (to include common misspellings) or a fundamentally
different text representation (e.g., embeddings) to address.
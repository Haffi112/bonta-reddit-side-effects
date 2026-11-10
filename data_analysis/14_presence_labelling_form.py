"""Build the labelling form for the nine side-effect "false positives" (Sölvi's point 3, 2026-10-24).

In the human verification, nine GPT-5.6 Luna outputs in the medical and cosmetic groups listed a side
effect but had no extracted phrase judged correct. Whether those posts describe a side effect at all
decides the post-level presence measure that the error correction uses, and so far it rests on one
author's reading. This script writes a self-contained HTML form so the three annotators can answer
that question independently.

The record identifiers come from the private adjudication file, and the form shows only each post's
title and body, as the verification interface did. Cases are labelled A to I in a seeded random order;
the key that maps labels to records is written next to the form, in the git-ignored results folder.
Neither the form nor its key may be published. The form keeps answers in the browser and copies them as
plain text for an email, so nothing is sent anywhere.

Example (``--preview`` writes the same form with placeholder text, for checking the layout):

    uv run python data_analysis/14_presence_labelling_form.py
    uv run python data_analysis/14_presence_labelling_form.py --preview /tmp/preview.html
"""

from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
ROOT = ANALYSIS_DIR.parent
PRIVATE = ANALYSIS_DIR / "02_llm_extraction_results"
OUT_DIR = PRIVATE / "labelling"
FORM = OUT_DIR / "side_effect_labelling_9_posts.html"
KEY = OUT_DIR / "side_effect_labelling_key.json"
SEED = 20261024
CONTACT = "hafsteinne@hi.is"


def record_id(row: dict) -> str:
    rid = str(row.get("name") or "t3_" + str(row.get("id")))
    return rid if rid.startswith("t3_") else "t3_" + rid


def load_posts(ids: list[str]) -> dict[str, tuple[str, str]]:
    wanted, posts = set(ids), {}
    with (ROOT / "filtered_dataset" / "submissions.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if (rid := record_id(row)) in wanted:
                posts[rid] = (row.get("title") or "", row.get("selftext") or "")
    missing = wanted - posts.keys()
    if missing:
        raise SystemExit(f"{len(missing)} records not found in the corpus file")
    return posts


CSS = """
:root { --bg:#ffffff; --surface:#f6f7f8; --ink:#1f2a33; --muted:#5b6770; --rule:#d5dbe0;
        --accent:#1f5c99; --accent-ink:#ffffff; --warn-bg:#fff4e5; --warn-rule:#e3a857; --ok:#2e7d4f; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
        --bg:#14191e; --surface:#1d242b; --ink:#e6ebef; --muted:#9aa7b2; --rule:#33404a;
        --accent:#6ea8e0; --accent-ink:#0d1216; --warn-bg:#2b2316; --warn-rule:#a77a35; --ok:#6cc08f; } }
:root[data-theme="dark"] { --bg:#14191e; --surface:#1d242b; --ink:#e6ebef; --muted:#9aa7b2; --rule:#33404a;
        --accent:#6ea8e0; --accent-ink:#0d1216; --warn-bg:#2b2316; --warn-rule:#a77a35; --ok:#6cc08f; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 780px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 1.6rem; line-height: 1.25; margin: 0 0 4px; }
h2 { font-size: 1.15rem; margin: 32px 0 8px; }
.sub { color: var(--muted); margin: 0 0 20px; }
.notice { background: var(--warn-bg); border: 1px solid var(--warn-rule); border-radius: 8px;
          padding: 12px 14px; margin: 16px 0; }
.notice strong { display: block; margin-bottom: 4px; }
ol, ul { padding-left: 1.3em; }
li { margin: 4px 0; }
blockquote { margin: 8px 0; padding: 8px 12px; border-left: 3px solid var(--accent); background: var(--surface); }
label.name { display:block; font-weight:600; margin: 24px 0 6px; }
input[type=text], textarea { width:100%; font: inherit; color: var(--ink); background: var(--bg);
       border:1px solid var(--rule); border-radius:6px; padding:8px 10px; }
.case { border:1px solid var(--rule); border-radius:10px; margin: 20px 0; overflow:hidden; }
.case header { display:flex; justify-content:space-between; align-items:baseline; gap:12px;
               background: var(--surface); padding: 10px 14px; border-bottom:1px solid var(--rule); }
.case header h3 { margin:0; font-size:1.05rem; }
.status { font-size:.85rem; color: var(--muted); }
.status.done { color: var(--ok); }
.post { padding: 14px; }
.post .title { font-weight: 600; margin: 0 0 8px; }
.post .body { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 420px; overflow-y: auto;
              border:1px solid var(--rule); border-radius:6px; padding:10px 12px; background: var(--bg); }
.post .empty { color: var(--muted); font-style: italic; }
fieldset { border:0; margin:0; padding: 4px 14px 14px; }
legend { font-weight:600; padding:0; margin-bottom:8px; }
.choices { display:flex; flex-wrap:wrap; gap:8px; }
.choices label { display:flex; align-items:center; gap:6px; border:1px solid var(--rule); border-radius:999px;
                 padding:6px 12px; cursor:pointer; }
.choices input:checked + span { font-weight:600; }
.note { padding: 0 14px 14px; }
.note label { display:block; font-size:.9rem; color: var(--muted); margin-bottom:4px; }
.finish { border-top:1px solid var(--rule); margin-top: 32px; padding-top: 8px; }
button { font: inherit; font-weight:600; background: var(--accent); color: var(--accent-ink);
         border:0; border-radius:8px; padding: 10px 18px; cursor:pointer; }
button:focus-visible, input:focus-visible, textarea:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
#feedback { margin: 10px 0; min-height: 1.5em; }
#output { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .9rem; min-height: 180px; }
"""

JS = """
const CASES = __CASES__;
const KEY = "side-effect-labelling-9-posts-v1";
function load() { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch (e) { return {}; } }
function save(s) { try { localStorage.setItem(KEY, JSON.stringify(s)); } catch (e) { /* private window */ } }
function collect() {
  const s = { name: document.getElementById("annotator").value.trim(), answers: {}, notes: {} };
  for (const c of CASES) {
    const picked = document.querySelector(`input[name="case-${c}"]:checked`);
    s.answers[c] = picked ? picked.value : "";
    s.notes[c] = document.getElementById(`note-${c}`).value.trim();
  }
  return s;
}
function restore(s) {
  if (s.name) document.getElementById("annotator").value = s.name;
  for (const c of CASES) {
    const v = s.answers && s.answers[c];
    if (v) { const el = document.querySelector(`input[name="case-${c}"][value="${v}"]`); if (el) el.checked = true; }
    if (s.notes && s.notes[c]) document.getElementById(`note-${c}`).value = s.notes[c];
  }
}
function refresh() {
  const s = collect();
  save(s);
  let done = 0;
  for (const c of CASES) {
    const el = document.getElementById(`status-${c}`);
    if (s.answers[c]) { done += 1; el.textContent = "Answered"; el.classList.add("done"); }
    else { el.textContent = "Not answered yet"; el.classList.remove("done"); }
  }
  document.getElementById("progress").textContent = `${done} of ${CASES.length} cases answered`;
  return s;
}
function buildText(s) {
  const lines = ["Side-effect labelling (9 posts): my answers", `Annotator: ${s.name || "(name not given)"}`,
                 `Date: ${new Date().toISOString().slice(0, 10)}`, ""];
  for (const c of CASES) {
    const note = s.notes[c] ? ` | note: ${s.notes[c].replace(/\\s+/g, " ")}` : "";
    lines.push(`Case ${c}: ${s.answers[c] || "NOT ANSWERED"}${note}`);
  }
  return lines.join("\\n");
}
function fallbackCopy(area) {
  area.focus(); area.select();
  try { return document.execCommand("copy"); } catch (e) { return false; }
}
async function copyAnswers() {
  const s = refresh();
  const text = buildText(s);
  const area = document.getElementById("output");
  area.value = text;
  const missing = CASES.filter(c => !s.answers[c]);
  let ok = false;
  try { await navigator.clipboard.writeText(text); ok = true; } catch (e) { ok = fallbackCopy(area); }
  const fb = document.getElementById("feedback");
  const warn = (missing.length ? ` Note: case${missing.length > 1 ? "s" : ""} ${missing.join(", ")} still unanswered.` : "")
             + (s.name ? "" : " Note: you have not entered your name.");
  fb.textContent = ok
    ? `Copied. Paste it into an email to __CONTACT__.${warn}`
    : `Copying was blocked by the browser. Select the text in the box below, copy it, and paste it into an email to __CONTACT__.${warn}`;
}
document.addEventListener("DOMContentLoaded", () => {
  restore(load());
  document.querySelectorAll("input, textarea.note-text").forEach(el => el.addEventListener("input", refresh));
  document.querySelectorAll("input[type=radio]").forEach(el => el.addEventListener("change", refresh));
  document.getElementById("copy").addEventListener("click", copyAnswers);
  refresh();
  if (location.hash === "#selftest") {   // used only to test the preview copy
    document.getElementById("annotator").value = "Test Annotator";
    CASES.forEach((c, i) => {
      document.querySelector(`input[name="case-${c}"][value="${["Yes", "No", "Can't tell"][i % 3]}"]`).checked = true;
    });
    document.getElementById("note-A").value = "short note";
    const s = refresh();
    document.getElementById("output").value = buildText(s);
    document.getElementById("feedback").textContent = buildText(s);
    document.title = "SELFTEST OK";
  }
});
"""


def case_html(label: str, title: str, body: str) -> str:
    body_html = html.escape(body) if body.strip() else '<span class="empty">(no body text)</span>'
    choices = "".join(
        f'<label><input type="radio" name="case-{label}" value="{v}"><span>{t}</span></label>'
        for v, t in (
            ("Yes", "Yes, it does"),
            ("No", "No, it does not"),
            ("Can't tell", "Can't tell"),
        )
    )
    return f"""
<section class="case" id="case-{label}">
  <header><h3>Case {label}</h3><span class="status" id="status-{label}">Not answered yet</span></header>
  <div class="post">
    <p class="title">{html.escape(title)}</p>
    <div class="body">{body_html}</div>
  </div>
  <fieldset>
    <legend>Does this post describe at least one side effect of the author's own BoNT-A treatment?</legend>
    <div class="choices">{choices}</div>
  </fieldset>
  <div class="note">
    <label for="note-{label}">Optional note: which side effect, or why (a few words; please do not copy text from the post)</label>
    <textarea class="note-text" id="note-{label}" rows="2"></textarea>
  </div>
</section>"""


def page(cases: list[tuple[str, str, str]]) -> str:
    labels = [c[0] for c in cases]
    script = JS.replace("__CASES__", json.dumps(labels)).replace("__CONTACT__", CONTACT)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Side-effect labelling</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>Side-effect labelling: 9 posts</h1>
<p class="sub">BoNT-A Reddit study · for Ragna, Sölvi and Kristján Sölvi · about 15 to 20 minutes</p>

<div class="notice"><strong>Confidential: for the three of you only.</strong>
This file contains the text of Reddit posts from the study's private corpus, which the approved research
plan allows only the named researchers to see. Please do not forward it, upload it, or paste any of its
text into an AI tool or other online service. Delete the file once you have sent your answers.</div>

<h2>Why we need this labelling</h2>
<p>In the human verification, each of you judged every field and every extracted side effect. In nine
GPT-5.6 Luna outputs from the medical and cosmetic groups, the model listed at least one side effect, but
none of the phrases it extracted was judged correct. That verdict is about the extracted text, not about
the post. A post can describe a real side effect even when the model picked out the wrong words, and it
can also describe no side effect at all.</p>
<p>Following Sölvi's comments, the paper now keeps these two questions apart. Whether a post describes a
side effect is what the study's prevalence comparison measures, and it is what the correction for
extraction error uses (Methods; Supplementary Tables 6 and 7). For these nine posts, that answer currently
rests on one author's reading. Three independent readings, combined by majority vote as in the rest of
the verification, put them on the same footing as everything else.</p>
<p>Judge each post on its own. There is no expected answer, and the cases can go either way.</p>

<h2>Instructions</h2>
<ol>
<li>Enter your name below. Your answers are kept in this browser as you go, so you can close the file
and come back.</li>
<li>For each case, read the title and the body, then answer one question: <em>does this post describe at
least one side effect of the author's own BoNT-A treatment?</em></li>
<li>Use the definition from the extraction prompt:
<blockquote>A side effect is an unwanted symptom, sign, functional change, or cosmetic outcome the author
says occurred to them after treatment. A question such as “has anyone had ptosis?” is not a reported side
effect. Treatment failure or short duration alone is not a side effect unless the author describes an
unwanted outcome.</blockquote>
<ul>
<li>Answer <strong>Yes</strong> if the author describes at least one such outcome after their own
treatment, even if the post also discusses other things.</li>
<li>Answer <strong>No</strong> if the post only asks about side effects or describes someone else's; if
the only complaint is that treatment did not work, has not worked yet, or wore off; if it only describes
the condition being treated, as it was before treatment; or if the author clearly attributes the problem
to something other than the BoNT-A treatment.</li>
<li>Answer <strong>Can't tell</strong> only if the post is genuinely ambiguous after a careful read.</li>
</ul></li>
<li>If you like, add a short note on which side effect or why. Please describe it in your own words and do
not copy text from the post.</li>
<li>Work on your own, and please do not discuss the cases with each other or with Hafsteinn until all three
of you have sent your answers.</li>
<li>When you have finished, click <strong>Copy my answers</strong> at the bottom and paste the text into an
email to {CONTACT}.</li>
</ol>

<label class="name" for="annotator">Your name</label>
<input type="text" id="annotator" autocomplete="name">

<h2>The cases</h2>
<p class="status" id="progress"></p>
{"".join(case_html(*c) for c in cases)}

<div class="finish">
<h2>Send your answers</h2>
<p>Click the button, then paste into an email to {CONTACT}. The copied text holds only your name, the
case letters, your answers and your notes, never the posts.</p>
<button id="copy" type="button">Copy my answers</button>
<p id="feedback" role="status"></p>
<label for="output" class="status">Your answers as text (also usable for copying by hand)</label>
<textarea id="output" readonly rows="15"></textarea>
</div>
</main>
<script>{script}</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview", type=Path, help="write a placeholder-text version here instead"
    )
    args = parser.parse_args()

    adjudicated = json.loads((PRIVATE / "author_adjudication.json").read_text(encoding="utf-8"))
    ids = sorted(adjudicated["presence_false_positives"])
    random.Random(SEED).shuffle(ids)
    labels = [chr(ord("A") + i) for i in range(len(ids))]

    if args.preview:
        filler = "Placeholder post body for checking the layout.\n\nSecond paragraph. " * 3
        cases = [(lab, f"Placeholder title for case {lab}", filler) for lab in labels]
        args.preview.write_text(page(cases), encoding="utf-8")
        print(f"preview written: {args.preview}")
        return

    posts = load_posts(ids)
    cases = [(lab, *posts[rid]) for lab, rid in zip(labels, ids, strict=True)]
    OUT_DIR.mkdir(exist_ok=True)
    FORM.write_text(page(cases), encoding="utf-8")
    KEY.write_text(
        json.dumps(dict(zip(labels, ids, strict=True)), indent=2) + "\n", encoding="utf-8"
    )
    print(f"form written: {FORM}\nkey written: {KEY}")


if __name__ == "__main__":
    main()

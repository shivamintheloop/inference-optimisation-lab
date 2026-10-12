"""Command line: python -m rag <command>

    ingest [--retry-failed] [--force]   scan the corpus and index what changed
    status [-v]                         documents, chunk counts, failures
    search "<question>" [--mode M] [-k N]  show retrieved chunks and scores
    ask "<question>" [--show-prompt]    cited answer from the notes, streamed
    export-prompts                      freeze the eval questions' LLM prompts (bench/)
    bench-llm --label L [--baseline F] [--record]
                                        time the LLM on the frozen prompts
    bench-ab --variants baseline draft:8 ... [--record]
                                        interleaved, cool-down-checked A/B of server variants
    eval [--mode M ...] [--answers] [--record LABEL]
                                        hit@1 / hit@5 / MRR@10 per retrieval mode;
                                        --answers adds the full-pipeline answer eval
"""
from __future__ import annotations

import argparse
import sys

from rag.config import CONFIG

STAGES = ("parse", "chunk", "embed", "store")


def cmd_ingest(args: argparse.Namespace) -> int:
    from rag.ingest import ingest

    res = ingest(retry_failed=args.retry_failed, force=args.force)
    t, a = res.trace, res.trace.attrs
    print(f"Ingest finished in {t.duration_ms / 1000:.1f} s  "
          f"(trace {t.trace_id[:8]} -> {CONFIG.traces_dir.name}/ingest-{t.start[:10]}.jsonl)")
    for key in ("scanned", "new", "changed", "unchanged", "deleted", "failed",
                "still_failed", "chunks_written"):
        hint = "   (use --retry-failed)" if key == "still_failed" and a[key] else ""
        print(f"  {key.replace('_', ' '):<15}{a[key]:>7,}{hint}")

    totals = {name: sum(s.duration_ms or 0 for s in t.spans if s.name == name) for name in STAGES}
    if any(totals.values()):
        print("Time by stage (all documents):")
        embed_tokens = sum(s.attrs.get("tokens", 0) for s in t.spans if s.name == "embed")
        for name in STAGES:
            extra = ""
            if name == "embed" and totals[name]:
                extra = f"   ({embed_tokens:,} tokens, {embed_tokens / (totals[name] / 1000):,.0f} tok/s)"
            print(f"  {name:<8}{totals[name] / 1000:>7.1f} s{extra}")

    if res.failed:
        print("Failed:")
        for doc_id, err in res.failed.items():
            print(f"  {doc_id}: {err}")
    return 1 if res.failed else 0


def cmd_status(args: argparse.Namespace) -> int:
    from rag import docs_db, vector_store

    conn = docs_db.connect()
    rows = list(docs_db.all_documents(conn).values())
    ok = [r for r in rows if r["status"] == "ok"]
    bad = [r for r in rows if r["status"] == "failed"]
    expected = sum(r["chunk_count"] for r in rows if r["content_hash"])
    qdrant, fts = vector_store.count(), docs_db.fts_count(conn)
    same = "consistent" if qdrant == fts == expected else "MISMATCH"

    print(f"Documents: {len(ok)} ok, {len(bad)} failed")
    print(f"Chunks:    Qdrant {qdrant:,} | keyword index {fts:,} | "
          f"DocumentsDB says {expected:,}  -> {same}")
    if args.verbose:
        for r in sorted(rows, key=lambda r: r["doc_id"]):
            print(f"  {r['status']:<7}{r['chunk_count']:>5}  {r['doc_id']}")
    if bad:
        print("Failed:")
        for r in bad:
            print(f"  {r['doc_id']}  (attempts {r['attempts']}): {r['error']}")
    return 0 if same == "consistent" else 1


def _short(doc_id: str) -> str:
    return doc_id.removesuffix("-flash.md").removesuffix(".md")


def cmd_search(args: argparse.Namespace) -> int:
    import textwrap

    from rag import docs_db
    from rag.retrieve import search
    from rag.trace import trace

    conn = docs_db.connect()
    with trace("query", config=CONFIG.snapshot()) as t:
        t.set("command", "search")
        t.set("mode", args.mode)
        t.set("question", args.question)
        hits = search(conn, args.question, args.mode)
    print(f"{args.mode} search: {args.question!r}  "
          f"({t.duration_ms:.0f} ms, {len(hits)} hits, trace {t.trace_id[:8]})\n")
    if args.mode == "rerank":
        from rag.rerank import is_not_found
        if is_not_found(hits):
            print(f"  -> NOT FOUND: top score below threshold {CONFIG.not_found_threshold}\n")
    print(f" {'#':>2}  {'score':>7}  {'vec':>3} {'bm25':>4} {'hyb':>3}  source")
    for i, h in enumerate(hits[:args.k], 1):
        p = h.payload or {}
        section = " > ".join(p.get("section", "").split(" > ")[1:]) or "-"
        where = f"p.{p['page']}" if p.get("page") else section
        vec, kw, hyb = (h.ranks.get(k, "-") for k in ("vector", "bm25", "hybrid"))
        print(f" {i:>2}  {h.score:>7.4f}  {vec:>3} {kw:>4} {hyb:>3}  {_short(h.doc_id)}  |  {where[:70]}")
        snippet = " ".join(p.get("text", "").split())[:220]
        print(textwrap.indent(textwrap.fill(snippet, 100), " " * 25))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    import textwrap

    from rag import docs_db
    from rag.context import format_source
    from rag.generate import answer, build_messages
    from rag.trace import trace

    conn = docs_db.connect()
    print(f"Q: {args.question}\n")
    with trace("query", config=CONFIG.snapshot()) as t:
        t.set("command", "ask")
        t.set("question", args.question)
        a = answer(conn, args.question, on_token=lambda s: print(s, end="", flush=True))
        t.set("answer", a.text)
    print("\n")

    if a.sources:
        print("Sources:")
        for s in a.sources:
            mark = "*" if s.n in a.cited else " "
            print(f" {mark}[S{s.n}] {s.label()[:80]}  ({s.source_path})")
        print("  (* = cited in the answer)")
        if a.invalid_citations:
            print(f"  WARNING: answer cites sources that don't exist: "
                  f"{', '.join(f'S{n}' for n in a.invalid_citations)}")

    spans = {sp.name: sp for sp in t.spans}
    def ms(name: str) -> float:
        return spans[name].duration_ms if name in spans else 0.0
    parts = [f"search {ms('retrieve') - ms('rerank'):.0f} ms",   # "retrieve" includes rerank
             f"rerank {ms('rerank'):.0f} ms"]
    if "rerank.load_model" in spans:
        parts[-1] += f" (incl. model load {ms('rerank.load_model') / 1000:.1f} s)"
    m = a.metrics
    if m:
        parts.append(f"TTFT {m['ttft_ms']:.0f} ms ({m['prompt_tokens']} prompt tokens)")
        parts.append(f"{m['output_tokens']} tokens @ {m['tpot_ms']:.1f} ms/token"
                     + (f" ({m['decode_tok_s']} tok/s)" if m.get("decode_tok_s") else ""))
        if (m.get("load_ms") or 0) > 100:
            parts.append(f"LLM cold load {m['load_ms'] / 1000:.1f} s")
        if m.get("done_reason") == "length":
            parts.append("CUT OFF at max_output_tokens")
    print(f"\n{' · '.join(parts)} · total {t.duration_ms / 1000:.1f} s · trace {t.trace_id[:8]}")

    if args.show_prompt and a.sources:
        context = "\n\n".join(format_source(s) for s in a.sources)
        print("\n--- prompt sent to the LLM ---")
        for msg in build_messages(args.question, context):
            print(f"[{msg['role']}]\n{textwrap.indent(msg['content'], '  ')}\n")
    return 0


def cmd_export_prompts(args: argparse.Namespace) -> int:
    from rag.bench_llm import export_extract_prompts, export_prompts, file_hash

    if args.extract:
        path, n = export_extract_prompts()
        print(f"Wrote {n} quote-the-evidence prompts to {path} (hash {file_hash(path)})")
        return 0
    path, n, skipped = export_prompts()
    print(f"Wrote {n} prompts to {path} (hash {file_hash(path)})"
          + (f"; skipped (not found): {', '.join(skipped)}" if skipped else ""))
    return 0


def cmd_bench_llm(args: argparse.Namespace) -> int:
    from rag.bench_llm import record_inference, run_bench

    res = run_bench(args.label, passes=args.passes, baseline=args.baseline)
    s = res["summary"]
    print(f"{args.label}  ({res['server'].get('cmd') or res['server']['engine']})")
    print(f"  saved {res['path']}")
    print(f"  TTFT p50 {s['ttft_ms']:.0f} ms · TPOT p50 {s['tpot_ms']:.2f} ms "
          f"(passes {s['tpot_ms_range'][0]:.2f}–{s['tpot_ms_range'][1]:.2f}) · "
          f"decode {s.get('decode_tok_s') or 0:.0f} tok/s · answer p50 {s['e2e_ms'] / 1000:.2f} s · "
          f"output p50 {s['output_tokens']} tokens")
    if "acceptance_rate" in s:
        print(f"  drafts: acceptance {s['acceptance_rate']:.2f} · tokens per check "
              f"{s['tokens_per_step']}")
    if "same_as_baseline" in s:
        print(f"  vs baseline: speedup TPOT {s.get('speedup_tpot')}× · answer "
              f"{s.get('speedup_answer')}× · same answers {s['same_as_baseline']}/{s['prompts']}")
    print(f"  stable across passes: {s['stable_across_passes']}/{s['prompts']}")
    if args.record:
        print(f"  recorded in {record_inference(res, args.note or '')}")
    return 0


def cmd_bench_ab(args: argparse.Namespace) -> int:
    from rag.bench_ab import as_bench_result, run_ab
    from rag.bench_llm import record_inference

    from pathlib import Path
    res = run_ab(args.variants, passes=args.passes, block_size=args.block_size,
                 pause_s=args.pause, idle_s=args.idle, prompts_path=Path(args.prompts),
                 limit=args.limit)
    print(f"\nA/B run: {len(args.variants)} variants x {args.passes} passes in "
          f"{res['minutes']} min, saved {res['path']}")
    print(f"  {'variant':<12} {'TPOT p50':>9} {'range':>13} {'speedup':>8} {'answer':>7} "
          f"{'accept':>7} {'tok/check':>9} {'same':>6} {'probe tok/s':>14}")
    for v in args.variants:
        s = res["summaries"][v]
        lo, hi = s["tpot_ms_range"]
        pr = s["probe_tok_s"]
        print(f"  {v:<12} {s['tpot_ms']:>7.1f} ms {lo:>5.1f}–{hi:<5.1f} ms "
              f"{s.get('speedup_tpot') or '–':>7}{'×' if s.get('speedup_tpot') else ' '} "
              f"{s.get('speedup_answer') or '–':>6}{'×' if s.get('speedup_answer') else ' '} "
              f"{s.get('acceptance_rate') or '–':>7} {s.get('tokens_per_step') or '–':>9} "
              f"{str(s.get('same_as_baseline', '–')) + ('/' + str(s['prompts']) if 'same_as_baseline' in s else ''):>6} "
              f"{pr[0]:>6.1f}–{pr[-1]:<6.1f}")
    for v in args.variants:
        print(f"  {v:<12} block drift (last/first TPOT): {res['summaries'][v]['block_drift']}")
    if args.record:
        for v in args.variants:
            s = res["summaries"][v]
            pr, drift = s["probe_tok_s"], s["block_drift"]
            log = record_inference(as_bench_result(res, v),
                                   f"{res['prompts']} ({res['n_prompts']}); "
                                   f"A/B interleaved vs {args.variants[0]}, {args.passes} passes, "
                                   f"idle {args.idle:.0f} s; probe {pr[0]:.0f}–{pr[-1]:.0f} tok/s; "
                                   f"block drift max {max(drift, default=1):.2f}×")
        print(f"  recorded {len(args.variants)} rows in {log}")
    return 0


def cmd_mlx_micro(args: argparse.Namespace) -> int:
    import json
    from datetime import datetime

    from rag.bench_mlx import micro
    res = micro(args.target, args.draft)
    step = res["target"][1]
    print(f"MLX micro-bench, {args.target} (depth {res['depth']}):")
    for n, ms in res["target"].items():
        print(f"  pass over {n:>2} tokens: {ms:6.1f} ms  ({ms / step:4.1f} normal steps)")
    print(f"  draft step ({args.draft}): {res['draft'][1]:.1f} ms  (c = {res['draft'][1] / step:.2f})")
    out = CONFIG.results_dir / f"mlx-micro-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"saved {out}")
    return 0


def cmd_mlx_ab(args: argparse.Namespace) -> int:
    from pathlib import Path

    from rag.bench_llm import record_inference
    from rag.bench_mlx import as_bench_result, run_ab
    res = run_ab(args.variants, args.target, args.draft, Path(args.prompts), args.limit,
                 idle_s=args.idle)
    print(f"\nMLX A/B: {res['n_prompts']} prompts in {res['minutes']} min, saved {res['path']}")
    for v in args.variants:
        s = res["summaries"][v]
        print(f"  {v:<9} TPOT p50 {s['tpot_ms']:5.1f} ms · speedup {s.get('speedup_tpot', '–')} · "
              f"answer {s.get('speedup_answer', '–')} · from draft {s.get('from_draft_share', '–')} · "
              f"same {s.get('same_as_baseline', '–')} · drift {s['block_drift']}")
    if args.record:
        for v in args.variants:
            s = res["summaries"][v]
            log = record_inference(as_bench_result(res, v),
                                   f"{res['prompts']} ({res['n_prompts']}); MLX A/B vs {args.variants[0]}, "
                                   f"idle {args.idle:.0f} s; tokens from kept guesses "
                                   f"{s.get('from_draft_share', '–')}; block drift max "
                                   f"{max(s['block_drift'], default=1):.2f}×")
        print(f"  recorded {len(args.variants)} rows in {log}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from rag.evaluate import run_eval

    from rag.retrieve import MODES

    modes = tuple(args.mode) if args.mode else MODES
    res = run_eval(modes)
    s = res["summary"]
    print(f"Retrieval eval  (config {res['config_hash']}, questions {res['questions_hash']}, "
          f"saved {res['path']})\n")
    if args.answers:
        from rag.evaluate import run_answer_eval
        print("Running the full pipeline on every question (this calls the LLM)...\n")
        res["answers"] = run_answer_eval()
    if args.record:
        from rag.evaluate import record_results
        log = record_results(res, args.record, args.note or "")
        print(f"Recorded as {args.record!r} in {log}\n")
    if res.get("answers"):
        g = res["answers"]["summary"]
        print(f"Answers ({g['llm_calls']} LLM calls, saved {res['answers']['path']}):")
        print(f"  refused right {g['refused_right']}/{g['n_unanswerable']} · refused wrong "
              f"{g['refused_wrong']}/{g['n_answerable']} · cited {g['cited']:.2f} · cites expected "
              f"{g['cites_expected']:.2f} · invalid cites {g['invalid_citations']} · cut off {g['cut_off']}")
        print(f"  TTFT p50 {g['ttft_p50_ms']:.0f} / p95 {g['ttft_p95_ms']:.0f} ms · TPOT p50 "
              f"{g['tpot_p50_ms']:.1f} ms · prompt {g['prompt_tokens_p50']:.0f} tok · output "
              f"{g['output_tokens_p50']:.0f} tok · per question p50 {g['e2e_p50_ms'] / 1000:.1f} s\n")
        for r in res["answers"]["per_question"]:
            if (r["style"] == "unanswerable") != (r["not_found"] or r["llm_refused"]) \
                    or (r["style"] != "unanswerable" and not r["cites_expected"]):
                print(f"  check {r['id']} ({r['style']}): {' '.join(r['answer'].split())[:110]}")
        print()
    print(f"{'mode':<8} {'hit@1':>6} {'hit@5':>6} {'MRR@10':>7}   "
          f"{'keyword hit@5':>13} {'paraphrase hit@5':>16}   {'p50 ms':>6}")
    for mode in modes:
        a, k, p = s[mode]["all"], s[mode]["keyword"], s[mode]["paraphrase"]
        print(f"{mode:<8} {a['hit@1']:>6.2f} {a['hit@5']:>6.2f} {a['mrr@10']:>7.3f}   "
              f"{k['hit@5']:>13.2f} {p['hit@5']:>16.2f}   {s[mode]['p50_ms']:>6.0f}")

    rows = res["per_question"]
    print("\nMisses (expected file not in top 5):")
    any_miss = False
    for r in rows:
        if r["style"] != "unanswerable" and not (r["rank"] and r["rank"] <= 5):
            any_miss = True
            print(f"  {r['id']} {r['mode']:<7} rank {str(r['rank'] or '>10'):>4}  "
                  f"top: {_short(r['top_doc'] or '-')}")
    if not any_miss:
        print("  none")

    th = s.get("rerank", {}).get("threshold")
    if th:
        print("\nNot-found threshold (re-rank top score):")
        print(f"  answerable questions: lowest {th['answerable_min']}, 10th pct {th['answerable_p10']}")
        print(f"  unanswerable:         highest {th['unanswerable_max']}  "
              f"-> {'cleanly separable' if th['separable'] else 'OVERLAP'}")
        print(f"  suggested threshold {th['suggested_threshold']}: keeps "
              f"{th['kept_answerable']}/{th['n_answerable']} answerable, refuses "
              f"{th['refused_unanswerable']}/{th['n_unanswerable']} unanswerable")
        cur = th["at_current"]
        print(f"  current threshold {cur['threshold']}: keeps {cur['kept_answerable']}/"
              f"{th['n_answerable']}, refuses {cur['refused_unanswerable']}/{th['n_unanswerable']}")
        print("  per question (re-rank top score):")
        for r in sorted((r for r in rows if r["mode"] == "rerank"), key=lambda r: r["top_score"]):
            print(f"    {r['top_score']:>7.2f}  {r['id']}  {r['style']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="scan the corpus and index what changed")
    pi.add_argument("--retry-failed", action="store_true",
                    help="retry files whose current version failed before")
    pi.add_argument("--force", action="store_true",
                    help="re-index every file (e.g. after changing chunk settings)")
    pi.set_defaults(func=cmd_ingest)

    ps = sub.add_parser("status", help="documents, chunk counts, failures")
    ps.add_argument("-v", "--verbose", action="store_true", help="list every document")
    ps.set_defaults(func=cmd_status)

    from rag.retrieve import MODES

    pq = sub.add_parser("search", help="show retrieved chunks and scores")
    pq.add_argument("question")
    pq.add_argument("--mode", choices=MODES, default="hybrid")
    pq.add_argument("-k", type=int, default=8, help="how many results to show")
    pq.set_defaults(func=cmd_search)

    pa = sub.add_parser("ask", help="answer a question from the notes, with citations")
    pa.add_argument("question")
    pa.add_argument("--show-prompt", action="store_true", help="print the messages sent to the LLM")
    pa.set_defaults(func=cmd_ask)

    px = sub.add_parser("export-prompts",
                        help="freeze the LLM prompts of the eval questions into bench/prompts-v1.jsonl")
    px.add_argument("--extract", action="store_true",
                    help="derive bench/prompts-extract-v1.jsonl (answers quote the sources word for word)")
    px.set_defaults(func=cmd_export_prompts)

    pb = sub.add_parser("bench-llm", help="time the LLM on the frozen prompts (inference experiments)")
    pb.add_argument("--label", required=True, help='e.g. "baseline" or "draft 1B k=8"')
    pb.add_argument("--passes", type=int, default=3)
    pb.add_argument("--baseline", help="results/llm-*.json to compare speed and answers with")
    pb.add_argument("--record", action="store_true", help="append a row to results/INFERENCE.md")
    pb.add_argument("--note", help="short note for the --record row")
    pb.set_defaults(func=cmd_bench_llm)

    pab = sub.add_parser("bench-ab", help="interleaved A/B timing of llama-server variants "
                                          "(restarts the server per block)")
    pab.add_argument("--variants", nargs="+", required=True,
                     help="first one is the baseline, e.g. baseline draft:8 ngram:16:12")
    pab.add_argument("--passes", type=int, default=2)
    pab.add_argument("--block-size", type=int, default=4)
    pab.add_argument("--pause", type=float, default=8.0, help="seconds between requests")
    pab.add_argument("--idle", type=float, default=120.0, help="idle seconds before each block's probe")
    pab.add_argument("--prompts", default="bench/prompts-v1.jsonl", help="frozen prompts file")
    pab.add_argument("--limit", type=int, help="only the first N prompts (quick screening)")
    pab.add_argument("--record", action="store_true", help="append one row per variant to results/INFERENCE.md")
    pab.set_defaults(func=cmd_bench_ab)

    from rag.bench_mlx import DRAFT as MLX_DRAFT, TARGET as MLX_TARGET
    pmm = sub.add_parser("mlx-micro", help="MLX: target pass cost vs tokens, draft step")
    pmm.add_argument("--target", default=MLX_TARGET)
    pmm.add_argument("--draft", default=MLX_DRAFT)
    pmm.set_defaults(func=cmd_mlx_micro)

    pma = sub.add_parser("mlx-ab", help="MLX: interleaved A/B of baseline vs draft:<k>")
    pma.add_argument("--variants", nargs="+", required=True, help="e.g. baseline draft:3 draft:6")
    pma.add_argument("--target", default=MLX_TARGET)
    pma.add_argument("--draft", default=MLX_DRAFT)
    pma.add_argument("--prompts", default="bench/prompts-v1.jsonl")
    pma.add_argument("--limit", type=int)
    pma.add_argument("--idle", type=float, default=120.0)
    pma.add_argument("--record", action="store_true")
    pma.set_defaults(func=cmd_mlx_ab)

    pe = sub.add_parser("eval", help="hit@1 / hit@5 / MRR@10 per retrieval mode")
    pe.add_argument("--answers", action="store_true",
                    help="also run the full pipeline (LLM) and score refusals, citations, timing")
    pe.add_argument("--mode", choices=MODES, action="append",
                    help="repeat to pick modes (default: all)")
    pe.add_argument("--record", metavar="LABEL",
                    help='append the results table to results/RESULTS.md, e.g. "Phase 4: ..."')
    pe.add_argument("--note", help="extra line saved with --record")
    pe.set_defaults(func=cmd_eval)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

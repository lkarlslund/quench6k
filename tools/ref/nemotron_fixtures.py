#!/usr/bin/env python3
"""Nemotron-H (Nemotron 3 Super, Munin 2.0) versions of the oracle fixtures: each Qwen fixture's
conversation and completion re-rendered with the checkpoint's chat template (the completion as the assistant
turn: its thinking as reasoning_content, the rest as content; the tool fixture's tools passed as `tools`, so
the system prompt is the template's own) and tokenized with the checkpoint's tokenizer.json, plus `danish`,
a Danish conversation (Munin is a Danish model).

The completions were written by other models, so they serve as teacher-forced text. A completion that ends
with the end of its turn keeps that <|im_end|> as its last token; one that was cut off ends where its text
ends (nothing of the template's turn closing is invented).

    python3 -I tools/ref/nemotron_fixtures.py --model <checkpoint> --qwen <qwen checkpoint> --out out/nemotron-fixtures
"""
import argparse
import json
import pathlib
import re

import jinja2
from tokenizers import Tokenizer

TURN = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>\n?", re.S)
TOOLS = re.compile(r"^(.*?)# Tools\n.*?<tools>\n(.*?)\n</tools>.*$", re.S)

DANISH = [
    {"role": "user", "content": "Forklar kort, hvordan et folketingsvalg i Danmark foregår, og hvordan mandaterne "
                                "bliver fordelt mellem partierne."},
    {"role": "assistant", "content": (
        "Ved et folketingsvalg vælger vælgerne de 179 medlemmer af Folketinget. Heraf vælges 175 i Danmark, mens "
        "Grønland og Færøerne hver vælger to.\n\n"
        "**Hvem kan stemme?** Danske statsborgere, der er fyldt 18 år og har fast bopæl i riget, har stemmeret.\n\n"
        "**Hvornår?** Der skal holdes valg mindst hvert fjerde år, men statsministeren kan udskrive valg når som "
        "helst inden da.\n\n"
        "**Sådan fordeles mandaterne:**\n"
        "1. Landet er delt i tre landsdele og ti storkredse. I storkredsene fordeles 135 kredsmandater efter "
        "d'Hondts metode på grundlag af partiernes stemmetal.\n"
        "2. De sidste 40 mandater er tillægsmandater. De sikrer, at hvert partis samlede antal pladser svarer så "
        "nøje som muligt til partiets andel af stemmerne på landsplan.\n"
        "3. Som hovedregel skal et parti have mindst 2 procent af de gyldige stemmer, spærregrænsen, for at få "
        "del i tillægsmandaterne.\n\n"
        "**Hvem bliver valgt?** Inden for hvert parti afgør opstillingsformen og kandidaternes personlige "
        "stemmer, hvem der får partiets pladser.\n\n"
        "Resultatet ligger derfor tæt på et rent forholdstalsvalg, samtidig med at alle dele af landet er "
        "repræsenteret.")},
]


def render(template, messages, thinking, tools=None, generation_prompt=False):
    # As transformers' apply_chat_template renders templates.
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def tojson(v, ensure_ascii=False, sort_keys=False, separators=None, **kw):
        return json.dumps(v, ensure_ascii=ensure_ascii, sort_keys=sort_keys, separators=separators, **kw)

    def raise_exception(msg):
        raise RuntimeError(msg)

    env.filters["tojson"] = tojson
    env.filters["items"] = lambda d: list(d.items())
    env.globals["raise_exception"] = raise_exception
    kw = {"tools": tools} if tools else {}
    return env.from_string(template).render(messages=messages, add_generation_prompt=generation_prompt,
                                            enable_thinking=thinking, **kw)


def build(name, tok, template, messages, reply, thinking, tools, finished, out_dir):
    """Write <out_dir>/<name>/sequence.json: the prompt (generation prompt) then the reply's tokens."""
    prompt = render(template, messages, thinking, tools, generation_prompt=True)
    full = render(template, messages + [reply], thinking, tools)
    if finished:
        full = full[:full.rindex("<|im_end|>") + len("<|im_end|>")]
    else:
        body = (reply.get("content") or reply.get("reasoning_content")).rstrip()
        tail = body[-48:]
        full = full[:full.rindex(tail) + len(tail)]
    if not full.startswith(prompt):
        raise SystemExit(f"{name}: the conversation's text does not start with the prompt's")
    p_ids = tok.encode(prompt, add_special_tokens=False).ids
    ids = tok.encode(full, add_special_tokens=False).ids
    if ids[:len(p_ids)] != p_ids:
        raise SystemExit(f"{name}: the conversation's tokens do not start with the prompt's")
    out = pathlib.Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "sequence.json").write_text(json.dumps({
        "generator": f"text of oracle/fixtures/{name}, Nemotron-H template and tokenizer" if name != "danish"
        else "tools/ref/nemotron_fixtures.py DANISH, Nemotron-H template and tokenizer",
        "prompt_len": len(p_ids), "tokens": ids}) + "\n")
    print(f"{name:14s} prompt {len(p_ids):6d} completion {len(ids) - len(p_ids):5d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--qwen", required=True, help="checkpoint with the Qwen tokenizer.json of the fixtures")
    ap.add_argument("--fixtures", default="oracle/fixtures")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = pathlib.Path(args.model)
    tok = Tokenizer.from_file(str(model / "tokenizer.json"))
    qwen = Tokenizer.from_file(str(pathlib.Path(args.qwen) / "tokenizer.json"))
    template = (model / "chat_template.jinja").read_text()
    for d in sorted(pathlib.Path(args.fixtures).iterdir()):
        if not (d / "spec.json").exists():
            continue
        spec = json.loads((d / "spec.json").read_text())
        seq = json.loads((d / "sequence.json").read_text())
        thinking = bool(spec.get("enable_thinking"))
        messages, tools = [], None
        for role, body in TURN.findall(spec["text"]):
            if role == "assistant":
                body = re.sub(r"^<think>.*?</think>\s*", "", body, flags=re.S)
            if role == "system" and (m := TOOLS.match(body)):
                tools = [json.loads(line) for line in m.group(2).splitlines() if line.strip()]
                body = m.group(1).strip()
            messages.append({"role": role, "content": body})
        completion = qwen.decode(seq["tokens"][seq["prompt_len"]:], skip_special_tokens=False)
        finished = "<|im_end|>" in completion or "<|endoftext|>" in completion
        completion = completion.replace("<|im_end|>", "").replace("<|endoftext|>", "")
        # Thinking fixtures' completions start inside the thinking (the Qwen prompt opens it).
        completion = re.sub(r"^\s*<think>", "", completion)
        if thinking:
            reasoning, closed, answer = completion.partition("</think>")
        else:
            reasoning, closed, answer = "", "", re.sub(r"^\s*</think>", "", completion)
        reply = {"role": "assistant", "content": answer.strip()}
        if reasoning.strip():
            reply["reasoning_content"] = reasoning.strip()
        if thinking and not closed:
            finished = False
        build(d.name, tok, template, messages, reply, thinking, tools, finished, args.out)
    build("danish", tok, template, DANISH[:1], DANISH[1], False, None, True, args.out)


if __name__ == "__main__":
    main()

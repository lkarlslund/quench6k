/* quench6k inspector: POST /v1/inspect and show the logit attribution. */
'use strict';
document.addEventListener('alpine:init', () => {
  Alpine.data('inspector', () => {
    const escape = s => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    return {
      prompt: 'The capital of France is', mode: 'raw', thinking: false, model: '', models: [], target: '',
      busy: false, error: '', result: null, open: null,
      // Model edits (see DOCS.md "Inspection"), the unedited top predictions, and the side-by-side texts.
      edits: [], baseTop: null, tokens: 48, compare: null, comparing: false,
      manual: { layer: '', expert: '', unit: '', scale: '0', add: '0' },
      get r() { return this.result.report; },
      get scale() {
        const r = this.r;
        return Math.max(Math.abs(r.embedding), Math.abs(r.ple), ...r.layers.flatMap(l => [Math.abs(l.mixer_contribution), Math.abs(l.moe_contribution)])) || 1;
      },
      get neuronScale() { return Math.max(...this.r.neurons.map(n => Math.abs(n.contribution))) || 1; },
      async init() {
        try {
          const res = await fetch('/v1/models');
          this.models = (await res.json()).data.map(m => m.id);
          this.model = this.models[0] || '';
        } catch (e) { this.error = 'Could not list models: ' + e; }
      },
      request(withEdits) {
        const body = { model: this.model };
        if (this.mode === 'chat') {
          body.messages = [{ role: 'user', content: this.prompt }];
          body.chat_template_kwargs = { enable_thinking: this.thinking };
        } else {
          body.prompt = this.prompt;
        }
        if (withEdits && this.edits.length) body.edits = this.edits.map(e => e.spec);
        return body;
      },
      async post(path, body) {
        const res = await fetch(path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error?.message || res.statusText);
        return data;
      },
      async run() {
        this.busy = true;
        this.error = '';
        const body = this.request(true);
        const t = this.target.trim();
        if (t) body.target = /^\d+$/.test(t) ? Number(t) : this.target;
        try {
          const data = await this.post('/v1/inspect', body);
          // With edits, the unedited prediction for comparison (its prompt is cached by now).
          let base = null;
          if (this.edits.length) {
            const plain = this.request(false);
            plain.target = data.report.target;
            const b = await this.post('/v1/inspect', plain);
            base = Object.fromEntries(b.report.top.map(p => [p.id, p.value]));
            base.target = b.report.logprob;
            Object.assign(data.tokens, b.tokens);
          }
          this.result = data;
          this.baseTop = base;
          this.open = null;
        } catch (e) {
          this.error = String(e.message || e);
        } finally {
          this.busy = false;
        }
      },
      // Greedy continuations without and with the edits, side by side.
      async generate() {
        this.comparing = true;
        this.error = '';
        const run = async withEdits => {
          const body = this.request(withEdits);
          Object.assign(body, { max_tokens: Number(this.tokens) || 48, temperature: 0 });
          const chat = this.mode === 'chat';
          const data = await this.post(chat ? '/v1/chat/completions' : '/v1/completions', body);
          const c = data.choices[0];
          return chat ? (c.message.reasoning_content ? '<think>' + c.message.reasoning_content + '</think>' : '') + (c.message.content || '') : c.text;
        };
        try {
          const base = await run(false);
          const edited = this.edits.length ? await run(true) : null;
          this.compare = { base, edited };
        } catch (e) {
          this.error = String(e.message || e);
        } finally {
          this.comparing = false;
        }
      },
      addEdit(spec, label) {
        const key = JSON.stringify(spec);
        if (!this.edits.some(e => JSON.stringify(e.spec) === key)) this.edits.push({ spec, label });
      },
      editNeuron(n, scale, add) {
        const spec = { layer: n.layer, unit: n.unit, scale, add: add || 0 };
        if (n.expert === null) spec.shared = true; else spec.expert = n.expert;
        this.addEdit(spec, this.neuronName(n) + (add ? ' +' + add : scale === 0 ? ' off' : ' ×' + scale));
      },
      banExpert(layer, expert) { this.addEdit({ layer, expert, ban: true }, 'L' + layer + '.E' + expert + ' banned'); },
      addManual() {
        const m = this.manual, layer = Number(m.layer), unit = m.unit === '' ? null : Number(m.unit);
        if (m.layer === '' || m.expert === '') { this.error = 'An edit needs a layer and an expert (or "shared").'; return; }
        const shared = m.expert.trim() === 'shared';
        if (unit === null) {
          if (shared) { this.error = 'A router edit needs an expert number.'; return; }
          this.banExpert(layer, Number(m.expert));
        } else {
          this.editNeuron({ layer, unit, expert: shared ? null : Number(m.expert) }, Number(m.scale), Number(m.add));
        }
      },
      delta(p) {
        if (!this.baseTop) return '';
        const before = this.baseTop[p.id];
        return before === undefined ? ' (new)' : ' (was ' + this.pct(before) + ')';
      },
      // A token's text with whitespace made visible.
      show(id) {
        const text = this.result.tokens[id] ?? '#' + id;
        return text.replace(/ /g, '␣').replace(/\n/g, '↵').replace(/\t/g, '⇥') || '∅';
      },
      fixed(v) { return (v >= 0 ? '+' : '') + v.toFixed(3); },
      pct(logprob) { const p = 100 * Math.exp(logprob); return (p >= 10 ? p.toFixed(1) : p.toPrecision(2)) + '%'; },
      sign(v) { return v >= 0 ? 'pos-text' : 'neg-text'; },
      bar(v, scale) {
        const w = Math.min(50, 50 * Math.abs(v) / (scale || this.scale));
        return '<i class="' + (v >= 0 ? 'pos' : 'neg') + '" style="width:' + w + '%"></i>';
      },
      pushes(p) {
        const tok = (t, cls) => '<span class="tok ' + cls + '" title="' + t.value.toFixed(3) + '">' + escape(this.show(t.id)) + '</span>';
        return '<span class="label">up</span>' + p.up.map(t => tok(t, 'up')).join('') +
          '<span class="label" style="margin-left:8px">down</span>' + p.down.map(t => tok(t, 'down')).join('');
      },
      neuronName(n) { return 'L' + n.layer + '.' + (n.expert === null ? 'shared' : 'E' + n.expert) + '.N' + n.unit; },
    };
  });
});

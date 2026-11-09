/* quench6k inspector: POST /v1/inspect and show the logit attribution. */
'use strict';
document.addEventListener('alpine:init', () => {
  Alpine.data('inspector', () => {
    const escape = s => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    return {
      prompt: 'The capital of France is', mode: 'raw', thinking: false, model: '', models: [], target: '',
      busy: false, error: '', result: null, open: null,
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
      async run() {
        this.busy = true;
        this.error = '';
        const body = { model: this.model };
        if (this.mode === 'chat') {
          body.messages = [{ role: 'user', content: this.prompt }];
          body.chat_template_kwargs = { enable_thinking: this.thinking };
        } else {
          body.prompt = this.prompt;
        }
        const t = this.target.trim();
        if (t) body.target = /^\d+$/.test(t) ? Number(t) : this.target;
        try {
          const res = await fetch('/v1/inspect', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
          const data = await res.json();
          if (!res.ok) throw new Error(data.error?.message || res.statusText);
          this.result = data;
          this.open = null;
        } catch (e) {
          this.error = String(e.message || e);
        } finally {
          this.busy = false;
        }
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

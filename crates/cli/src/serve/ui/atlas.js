/* quench6k neuron atlas: build atlases, rank neurons and experts, send edits to the inspector. */
'use strict';
document.addEventListener('alpine:init', () => {
  Alpine.data('atlas', () => {
    const escape = s => s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    const visible = s => s.replace(/\n/g, '↵\n');
    let timer = null;
    return {
      models: [], model: '', atlases: [], atlasDir: false, error: '', added: '',
      name: '', source: 'topic', topic: '', samples: 16, maxTokens: 512, upload: null, file: '', creating: false,
      baseChosen: false,
      query: { atlas: '', base: '', by: 'mean', layer: '', shared: false }, result: null, loading: false, lens: {},
      get ready() { return this.atlases.filter(a => a.phase === 'ready'); },
      async init() {
        try {
          this.models = (await (await fetch('/v1/models')).json()).data.map(m => m.id);
          this.model = this.models[0] || '';
        } catch (e) { this.error = String(e); }
        await this.refresh();
        timer = setInterval(() => this.refresh(), 2000);
      },
      async refresh() {
        try {
          const data = await (await fetch('/v1/atlas')).json();
          this.atlases = data.data;
          this.atlasDir = data.atlas_dir;
          if (!this.ready.some(a => a.name === this.query.atlas)) this.query.atlas = this.ready[0]?.name || '';
          // A baseline by default: another ready atlas.
          if (!this.query.base && !this.baseChosen) this.query.base = this.ready.find(a => a.name !== this.query.atlas)?.name || '';
        } catch (e) { /* the next poll retries */ }
      },
      async call(method, path, body) {
        const res = await fetch(path, { method, headers: { 'content-type': 'application/json' }, body: body && JSON.stringify(body) });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error?.message || res.statusText);
        return data;
      },
      readFile(ev) {
        const f = ev.target.files[0];
        if (!f) return;
        const r = new FileReader();
        r.onload = () => { this.upload = { name: f.name, text: r.result }; if (!this.name) this.name = f.name.replace(/\.[^.]*$/, '').slice(0, 64); };
        r.readAsText(f);
      },
      async create() {
        this.error = '';
        const body = { name: this.name.trim(), model: this.model };
        if (this.source === 'topic') Object.assign(body, { topic: this.topic, samples: Number(this.samples), max_tokens: Number(this.maxTokens) });
        else if (this.source === 'upload') { if (!this.upload) { this.error = 'Choose a file first.'; return; } Object.assign(body, { text: this.upload.text, filename: this.upload.name }); }
        else body.file = this.file;
        this.creating = true;
        try { await this.call('POST', '/v1/atlas', body); await this.refresh(); }
        catch (e) { this.error = String(e.message || e); }
        finally { this.creating = false; }
      },
      async remove(name) {
        try { await this.call('DELETE', '/v1/atlas/' + encodeURIComponent(name)); await this.refresh(); }
        catch (e) { this.error = String(e.message || e); }
      },
      params() {
        const p = new URLSearchParams({ by: this.query.by, limit: '60' });
        if (this.query.base) p.set('base', this.query.base);
        if (this.query.layer !== '') p.set('layer', this.query.layer);
        if (this.query.shared) p.set('shared', 'true');
        return p;
      },
      async loadNeurons() {
        this.loading = true; this.error = '';
        try {
          this.result = await this.call('GET', '/v1/atlas/' + encodeURIComponent(this.query.atlas) + '/neurons?' + this.params());
          this.lens = {};
          // What the top neurons write, in batches of 64.
          const ns = this.result.neurons.slice(0, 64).map(n => n.expert === null ? { layer: n.layer, shared: true, unit: n.unit } : { layer: n.layer, expert: n.expert, unit: n.unit });
          if (ns.length) {
            const l = await this.call('POST', '/v1/lens', { neurons: ns });
            Object.assign(this.result.texts, l.texts);
            const lens = {};
            l.neurons.forEach(n => { lens[this.nname(n)] = n.pushes; });
            this.lens = lens;
          }
        } catch (e) { this.error = String(e.message || e); }
        finally { this.loading = false; }
      },
      async loadExperts() {
        this.loading = true; this.error = '';
        try { this.result = await this.call('GET', '/v1/atlas/' + encodeURIComponent(this.query.atlas) + '/experts?' + this.params()); }
        catch (e) { this.error = String(e.message || e); }
        finally { this.loading = false; }
      },
      nname(n) { return 'L' + n.layer + '.' + (n.expert === null || n.expert === undefined ? 'shared' : 'E' + n.expert) + '.N' + n.unit; },
      text(id) { return this.result.texts[id] ?? '#' + id; },
      snippet(n) {
        if (!n.peak) return '<span class="muted">never fired</span>';
        return n.peak.context.map((id, i) => {
          const t = escape(visible(this.text(id)));
          return i === n.peak.at ? '<b>' + t + '</b>' : t;
        }).join('');
      },
      // The neuron's effect at its peak: a negative activation writes the lens reversed.
      pushes(p, peak) {
        const flip = peak && peak.value < 0;
        const [up, down] = flip ? [p.down, p.up] : [p.up, p.down];
        const tok = (t, cls) => '<span class="tok ' + cls + '" title="' + t.value.toFixed(2) + '">' + escape(this.text(t.id).replace(/ /g, '␣').replace(/\n/g, '↵') || '∅') + '</span>';
        return '<span class="label">' + (flip ? 'at its (negative) peak: up' : 'up') + '</span>' + up.map(t => tok(t, 'up')).join('') + '<span class="label" style="margin-left:6px">down</span>' + down.map(t => tok(t, 'down')).join('');
      },
      // Edits go to the inspector through local storage (it reads them when it loads).
      sendEdit(spec, label) {
        let list = [];
        try { list = JSON.parse(localStorage.getItem('quench6k.edits') || '[]'); } catch (e) { list = []; }
        if (!list.some(e => JSON.stringify(e.spec) === JSON.stringify(spec))) list.push({ spec, label });
        try { localStorage.setItem('quench6k.edits', JSON.stringify(list)); this.added = 'Added ' + label + ' to the inspector\'s edits (' + list.length + ' in total).'; }
        catch (e) { this.error = 'Could not save the edit: ' + e; }
      },
      toInspector(n, scale, add) {
        const spec = { layer: n.layer, unit: n.unit, scale, add: add || 0 };
        if (n.expert === null) spec.shared = true; else spec.expert = n.expert;
        this.sendEdit(spec, this.nname(n) + (add ? ' +' + add : ' off'));
      },
      banToInspector(e) { this.sendEdit({ layer: e.layer, expert: e.expert, ban: true }, 'L' + e.layer + '.E' + e.expert + ' banned'); },
    };
  });
});

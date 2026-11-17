import React, { useEffect, useState } from 'react';
import { fileToBase64, importPdf, importRecords, previewProtocol, proposeProtocol } from '../utils/api';

// The guided review: four steps that turn the question into a review protocol before anything is searched.
// Each step asks for a decision and says what it changes. The protocol then drives the searches and selection,
// and is stored with the results.

const STEPS = ['Purpose', 'Question', 'Eligibility', 'Search preview'];

const PURPOSES = [
  ['effect', 'Does something work, or is it linked to an outcome?', 'e.g. "Does vitamin D reduce fractures?"'],
  ['map', 'What research exists on a topic, and where is it thin?', 'e.g. "Machine learning in drug discovery"'],
  ['overview', 'An overview or background on a topic', 'e.g. for an introduction or a grant'],
];

const REVIEW_TYPES = [
  ['narrative', 'Narrative review'],
  ['systematic_prisma', 'Systematic-style review (PRISMA 2020)'],
  ['scoping', 'Scoping review (PRISMA-ScR)'],
];

const ORGANIZATIONS = [
  ['thematic', 'By theme'],
  ['chronological', 'Over time'],
  ['methodological', 'By study method'],
  ['theoretical', 'By theory'],
];

const ROLES = [
  ['intervention', 'Intervention'],
  ['exposure', 'Exposure'],
  ['outcome', 'Outcome'],
  ['population', 'Population'],
  ['comparison', 'Comparison (recorded, not searched)'],
  ['topic', 'Topic'],
  ['method', 'Method'],
];

const DESIGNS = [
  ['rct', 'Randomised controlled trials'],
  ['controlled_observational', 'Cohort, case-control and other controlled studies'],
  ['descriptive_observational', 'Cross-sectional and survey studies'],
  ['experimental_lab', 'Laboratory studies (cell, animal, assay)'],
  ['computational', 'Computational and modelling studies'],
  ['theoretical', 'Theoretical studies'],
  ['qualitative', 'Qualitative studies'],
  ['primary_other', 'Other primary research'],
  ['case_report', 'Case reports and series'],
  ['evidence_synthesis', 'Systematic reviews and meta-analyses'],
  ['review', 'Narrative reviews'],
  ['commentary', 'Editorials and commentaries'],
];

const box = 'bg-white border border-slate-200 rounded-xl p-4 space-y-3';
const btn = 'px-4 py-2 rounded-lg text-sm font-semibold transition';
const btnPrimary = `${btn} bg-indigo-600 hover:bg-indigo-500 text-white disabled:opacity-50`;
const btnPlain = `${btn} bg-white border border-slate-300 text-slate-700 hover:bg-slate-50`;
const input = 'bg-white border border-slate-300 rounded-lg px-2.5 py-1.5 text-sm text-slate-900 focus:border-indigo-500 outline-none';

function Changes({ children }) {
  return (
    <p className="text-xs text-slate-600 bg-slate-50 border border-slate-200 rounded-lg px-3 py-2">
      <strong className="text-slate-700">What this changes:</strong> {children}
    </p>
  );
}

// Full class names (Tailwind only generates classes it can find written out in the source)
const TONES = {
  indigo: 'bg-indigo-50 border-indigo-200 text-indigo-800',
  emerald: 'bg-emerald-50 border-emerald-200 text-emerald-800',
  slate: 'bg-slate-100 border-slate-200 text-slate-700',
};

function Chips({ items, onRemove, tone = 'indigo' }) {
  return (
    <span className="inline-flex flex-wrap gap-1">
      {items.map((s, i) => (
        <span
          key={`${s}-${i}`}
          className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs border ${TONES[tone] || TONES.indigo}`}
        >
          {s}
          {onRemove && (
            <button type="button" aria-label={`Remove ${s}`} onClick={() => onRemove(i)} className="font-bold opacity-60 hover:opacity-100">
              ×
            </button>
          )}
        </span>
      ))}
    </span>
  );
}

function AddText({ placeholder, onAdd }) {
  const [v, setV] = useState('');
  const add = () => {
    const t = v.trim();
    if (t) onAdd(t);
    setV('');
  };
  return (
    <input
      value={v}
      onChange={(e) => setV(e.target.value)}
      onKeyDown={(e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          add();
        }
      }}
      onBlur={add}
      placeholder={placeholder}
      className={`${input} text-xs w-40`}
    />
  );
}

export default function GuidedReview({ query, domain, config, hasLlm, running, onCancel, onRun }) {
  const [step, setStep] = useState(1);
  const [purpose, setPurpose] = useState(null);
  const [protocol, setProtocol] = useState(null);
  const [recommended, setRecommended] = useState(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [preview, setPreview] = useState(null);
  const [showStrings, setShowStrings] = useState(false);
  // Records and full texts the user brings (local installs only)
  const [imports, setImports] = useState([]);
  const [pdfs, setPdfs] = useState([]);
  const [importSource, setImportSource] = useState({});

  const uploadExport = async (file, search) => {
    setBusy(`Reading ${file.name}...`);
    setError('');
    try {
      const res = await importRecords({
        filename: file.name,
        content_base64: await fileToBase64(file),
        source: importSource[search.database] || search.database,
        search_string: search.query,
      });
      setImports((prev) => [...prev, res]);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy('');
    }
  };

  const uploadPdfs = async (files) => {
    setError('');
    for (const file of files) {
      setBusy(`Reading ${file.name}...`);
      try {
        const res = await importPdf({ filename: file.name, content_base64: await fileToBase64(file) });
        setPdfs((prev) => [...prev, { ...res, force_include: false, reason: '' }]);
      } catch (e) {
        setError(`${file.name}: ${e.message}`);
      }
    }
    setBusy('');
  };

  const propose = async (p, acceptDefaults = false) => {
    setBusy(hasLlm ? 'Reading the question and asking the model for synonyms...' : 'Reading the question...');
    setError('');
    try {
      const proto = await proposeProtocol({
        query,
        mode: 'deep_research',
        purpose: p,
        domain: domain || null,
        suggest: hasLlm,
        config,
      });
      proto.accepted_defaults = acceptDefaults;
      setProtocol(proto);
      setRecommended(proto.review_type);
      setPreview(null);
      if (acceptDefaults) setStep(4);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy('');
    }
  };

  // The search preview is fetched whenever step 4 is shown for the current protocol
  useEffect(() => {
    if (step !== 4 || !protocol || preview) return;
    let cancelled = false;
    setBusy('Checking how many records each database holds...');
    previewProtocol({ protocol, mode: 'deep_research', config })
      .then((p) => !cancelled && setPreview(p))
      .catch((e) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setBusy(''));
    return () => {
      cancelled = true;
    };
  }, [step, protocol, preview, config]);

  // Every edit marks what the user changed, so the stored protocol shows which decisions were theirs
  const update = (fn) => {
    setProtocol((prev) => {
      const next = JSON.parse(JSON.stringify(prev));
      fn(next);
      next.accepted_defaults = false;
      return next;
    });
    setPreview(null);
  };
  const editConcept = (i, fn) =>
    update((p) => {
      fn(p.concepts[i]);
      p.concepts[i].set_by = 'user';
    });
  const changeEligibility = (field, fn) =>
    update((p) => {
      fn(p.eligibility);
      if (!p.eligibility.changed_by_user.includes(field)) p.eligibility.changed_by_user.push(field);
    });
  const acceptSuggestion = (k) =>
    update((p) => {
      const s = p.suggestions[k];
      if (s.kind === 'synonym') {
        const c = p.concepts.find((x) => x.name === s.concept);
        if (c && !c.synonyms.includes(s.value)) {
          c.synonyms.push(s.value);
          if (c.set_by === 'default') c.set_by = 'suggestion';
        }
      } else if (s.kind === 'concept') {
        p.concepts.push({ role: 'topic', name: s.value, synonyms: s.terms || [], main: false, set_by: 'suggestion' });
      } else if (s.kind === 'aspect') {
        p.aspects.push({ label: s.value, terms: s.terms || [], question: '', set_by: 'suggestion' });
      }
      p.suggestions.splice(k, 1);
    });

  const searchable = protocol && protocol.concepts.some((c) => c.name.trim() && c.role !== 'comparison');
  const goTo = (n) => {
    setError('');
    setStep(n);
  };

  return (
    <div className="bg-indigo-50/60 border border-indigo-200 rounded-2xl p-5 space-y-4" aria-label="Guided review set-up">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-base font-bold text-slate-900">Set up the review</h2>
          <p className="text-xs text-slate-600">
            Four decisions before anything is searched. They are saved with the results as the review protocol.
          </p>
        </div>
        <button type="button" onClick={onCancel} className="text-xs font-semibold text-slate-600 hover:text-slate-900">
          Cancel
        </button>
      </div>

      <ol className="flex flex-wrap gap-2 text-xs">
        {STEPS.map((label, i) => (
          <li
            key={label}
            className={`px-3 py-1 rounded-full border font-semibold ${
              step === i + 1
                ? 'bg-indigo-600 border-indigo-600 text-white'
                : step > i + 1
                ? 'bg-white border-indigo-300 text-indigo-700'
                : 'bg-white border-slate-200 text-slate-400'
            }`}
          >
            {i + 1}. {label}
          </li>
        ))}
      </ol>

      {busy && <p className="text-xs text-indigo-700 font-medium">{busy}</p>}
      {error && <p className="text-xs text-rose-700 bg-rose-50 border border-rose-200 rounded-lg px-3 py-2">{error}</p>}

      {step === 1 && (
        <div className={box}>
          <h3 className="text-sm font-bold text-slate-800">1. What is this review for?</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
            {PURPOSES.map(([key, title, example]) => (
              <button
                key={key}
                type="button"
                disabled={!!busy}
                onClick={() => {
                  setPurpose(key);
                  propose(key);
                }}
                className={`text-left p-3 rounded-lg border transition ${
                  purpose === key ? 'border-indigo-500 bg-indigo-50' : 'border-slate-200 hover:border-indigo-300'
                }`}
              >
                <div className="text-sm font-semibold text-slate-800">{title}</div>
                <div className="text-xs text-slate-500 mt-0.5">{example}</div>
              </button>
            ))}
          </div>
          {protocol && purpose && (
            <div className="space-y-2">
              <p className="text-sm text-slate-700">
                <strong>Recommended:</strong> {REVIEW_TYPES.find(([v]) => v === recommended)?.[1]}.{' '}
                <span className="text-slate-600">{protocol.review_type_reason}</span>
              </p>
              <div className="flex flex-wrap gap-3 text-sm">
                <label className="inline-flex items-center gap-2">
                  Review type
                  <select
                    value={protocol.review_type}
                    onChange={(e) =>
                      update((p) => {
                        p.review_type = e.target.value;
                        p.review_type_reason = `Chosen by the user (recommended: ${
                          REVIEW_TYPES.find(([v]) => v === recommended)?.[1]
                        }).`;
                      })
                    }
                    className={input}
                  >
                    {REVIEW_TYPES.map(([v, l]) => (
                      <option key={v} value={v}>
                        {l}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="inline-flex items-center gap-2">
                  Organise findings
                  <select
                    value={protocol.organization}
                    onChange={(e) => update((p) => (p.organization = e.target.value))}
                    className={input}
                  >
                    {ORGANIZATIONS.map(([v, l]) => (
                      <option key={v} value={v}>
                        {l}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <Changes>
                which studies count as eligible (a systematic-style review takes every eligible primary study and
                excludes reviews), whether each study's risk of bias is judged, and how the report is laid out.
              </Changes>
            </div>
          )}
          <div className="flex flex-wrap items-center gap-3 pt-1">
            <button type="button" className={btnPrimary} disabled={!protocol || !purpose || !!busy} onClick={() => goTo(2)}>
              Confirm and continue
            </button>
            <button
              type="button"
              className="text-xs font-semibold text-indigo-700 hover:underline disabled:opacity-50"
              disabled={!!busy}
              onClick={() => {
                setPurpose(null);
                propose(null, true);
              }}
            >
              Skip and accept all proposed settings (recorded as such)
            </button>
          </div>
        </div>
      )}

      {step === 2 && protocol && (
        <div className={box}>
          <h3 className="text-sm font-bold text-slate-800">2. Check each part of the question</h3>
          <p className="text-xs text-slate-600">
            Every included study must mention each concept (or one of its synonyms) in its title or abstract. Tick
            the main topic. Comparisons are recorded but not searched for.
          </p>
          <div className="space-y-2">
            {protocol.concepts.map((c, i) => (
              <div key={i} className="flex flex-wrap items-center gap-2 border border-slate-200 rounded-lg p-2">
                <select
                  value={c.role}
                  onChange={(e) => editConcept(i, (x) => (x.role = e.target.value))}
                  className={`${input} text-xs`}
                  aria-label="Role"
                >
                  {ROLES.map(([v, l]) => (
                    <option key={v} value={v}>
                      {l}
                    </option>
                  ))}
                </select>
                <input
                  value={c.name}
                  onChange={(e) => editConcept(i, (x) => (x.name = e.target.value))}
                  className={`${input} w-56`}
                  aria-label="Concept"
                />
                <label className="inline-flex items-center gap-1 text-xs text-slate-600">
                  <input type="checkbox" checked={!!c.main} onChange={(e) => editConcept(i, (x) => (x.main = e.target.checked))} />
                  main topic
                </label>
                <span className="text-xs text-slate-500">synonyms:</span>
                <Chips items={c.synonyms} onRemove={(k) => editConcept(i, (x) => x.synonyms.splice(k, 1))} />
                <AddText placeholder="add a synonym" onAdd={(t) => editConcept(i, (x) => x.synonyms.push(t))} />
                <button
                  type="button"
                  onClick={() => update((p) => p.concepts.splice(i, 1))}
                  className="ml-auto text-xs text-rose-600 hover:underline"
                >
                  remove
                </button>
              </div>
            ))}
            <button
              type="button"
              onClick={() =>
                update((p) => p.concepts.push({ role: 'topic', name: '', synonyms: [], main: false, set_by: 'user' }))
              }
              className="text-xs font-semibold text-indigo-700 hover:underline"
            >
              + Add a concept
            </button>
          </div>

          <div className="space-y-1">
            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wide">Populations and places</h4>
            <p className="text-xs text-slate-600">Every included study must name one of these spellings.</p>
            {protocol.populations_and_places.map((c, i) => (
              <div key={i} className="flex flex-wrap items-center gap-2">
                <input
                  value={c.name}
                  onChange={(e) =>
                    update((p) => {
                      p.populations_and_places[i].name = e.target.value;
                      p.populations_and_places[i].set_by = 'user';
                    })
                  }
                  className={`${input} w-44`}
                  aria-label="Population or place"
                />
                <Chips
                  items={c.spellings}
                  tone="emerald"
                  onRemove={(k) =>
                    update((p) => {
                      p.populations_and_places[i].spellings.splice(k, 1);
                      p.populations_and_places[i].set_by = 'user';
                    })
                  }
                />
                <AddText
                  placeholder="add a spelling"
                  onAdd={(t) =>
                    update((p) => {
                      p.populations_and_places[i].spellings.push(t.toLowerCase());
                      p.populations_and_places[i].set_by = 'user';
                    })
                  }
                />
                <button type="button" onClick={() => update((p) => p.populations_and_places.splice(i, 1))} className="text-xs text-rose-600 hover:underline">
                  remove
                </button>
              </div>
            ))}
            <button
              type="button"
              onClick={() => update((p) => p.populations_and_places.push({ kind: 'population', name: '', spellings: [], set_by: 'user' }))}
              className="text-xs font-semibold text-indigo-700 hover:underline"
            >
              + Add a population or place
            </button>
          </div>

          <div className="space-y-1">
            <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wide">Aspects searched separately</h4>
            <p className="text-xs text-slate-600">Each gets its own search to widen coverage; studies need not mention them.</p>
            <Chips
              items={protocol.aspects.map((a) => a.label)}
              tone="slate"
              onRemove={(k) => update((p) => p.aspects.splice(k, 1))}
            />
          </div>

          {protocol.suggestions.length > 0 && (
            <div className="space-y-1 border-t border-slate-100 pt-2">
              <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wide">Suggested by the model</h4>
              <p className="text-xs text-slate-600">Nothing here is used unless you add it.</p>
              <div className="flex flex-wrap gap-1.5">
                {protocol.suggestions.map((s, k) => (
                  <button
                    key={`${s.kind}-${s.value}-${k}`}
                    type="button"
                    onClick={() => acceptSuggestion(k)}
                    className="px-2 py-0.5 rounded-full text-xs border border-dashed border-indigo-300 text-indigo-800 hover:bg-indigo-50"
                  >
                    + {s.kind === 'synonym' ? `${s.value} (synonym of ${s.concept})` : s.kind === 'concept' ? `concept: ${s.value}` : `aspect: ${s.value}`}
                  </button>
                ))}
              </div>
            </div>
          )}

          <Changes>the search strings sent to every database, and which studies are counted as addressing the question.</Changes>
          <div className="flex gap-2">
            <button type="button" className={btnPlain} onClick={() => goTo(1)}>
              Back
            </button>
            <button type="button" className={btnPrimary} disabled={!searchable} onClick={() => goTo(3)}>
              Confirm and continue
            </button>
          </div>
          {!searchable && <p className="text-xs text-rose-700">Add at least one concept to search for.</p>}
        </div>
      )}

      {step === 3 && protocol && (
        <div className={box}>
          <h3 className="text-sm font-bold text-slate-800">3. Which studies are eligible?</h3>
          <div>
            <div className="text-xs font-bold text-slate-700 uppercase tracking-wide mb-1">Study designs</div>
            <p className="text-xs text-slate-600 mb-1">
              {protocol.eligibility.study_designs.length ? 'Only the ticked designs are eligible.' : 'No design ticked: every design is eligible.'}
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-1">
              {DESIGNS.map(([key, label]) => (
                <label key={key} className="inline-flex items-center gap-2 text-sm text-slate-700">
                  <input
                    type="checkbox"
                    checked={protocol.eligibility.study_designs.includes(key)}
                    onChange={(e) =>
                      changeEligibility('study_designs', (el) => {
                        el.study_designs = e.target.checked
                          ? [...el.study_designs, key]
                          : el.study_designs.filter((d) => d !== key);
                      })
                    }
                  />
                  {label}
                </label>
              ))}
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-sm text-slate-700">
            <span className="text-xs font-bold text-slate-700 uppercase tracking-wide">Publication years</span>
            <input
              type="number"
              placeholder="from"
              value={protocol.eligibility.year_from ?? ''}
              onChange={(e) => changeEligibility('years', (el) => (el.year_from = e.target.value ? Number(e.target.value) : null))}
              className={`${input} w-24`}
              aria-label="From year"
            />
            <input
              type="number"
              placeholder="to"
              value={protocol.eligibility.year_to ?? ''}
              onChange={(e) => changeEligibility('years', (el) => (el.year_to = e.target.value ? Number(e.target.value) : null))}
              className={`${input} w-24`}
              aria-label="To year"
            />
          </div>
          <div className="flex flex-wrap items-center gap-3 text-sm text-slate-700">
            <span className="text-xs font-bold text-slate-700 uppercase tracking-wide">Existing reviews</span>
            {[
              ['default', protocol.review_type === 'systematic_prisma' ? 'Exclude and cite in the Discussion (usual)' : 'Include (usual)'],
              ['include', 'Include'],
              ['exclude', 'Exclude'],
            ].map(([v, l]) => (
              <label key={v} className="inline-flex items-center gap-1.5">
                <input
                  type="radio"
                  name="reviews"
                  checked={protocol.eligibility.reviews === v}
                  onChange={() => changeEligibility('reviews', (el) => (el.reviews = v))}
                />
                {l}
              </label>
            ))}
          </div>
          <Changes>
            studies outside these criteria are excluded before ranking, and each exclusion is counted under its own
            reason in the record flow.
          </Changes>
          <div className="flex gap-2">
            <button type="button" className={btnPlain} onClick={() => goTo(2)}>
              Back
            </button>
            <button type="button" className={btnPrimary} onClick={() => goTo(4)}>
              Confirm and preview the searches
            </button>
          </div>
        </div>
      )}

      {step === 4 && protocol && (
        <div className={box}>
          <h3 className="text-sm font-bold text-slate-800">4. Search preview</h3>
          {protocol.accepted_defaults && (
            <p className="text-xs text-amber-800 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
              You accepted the proposed settings without changes; this is recorded with the results.
            </p>
          )}
          {preview && (
            <>
              <div className="text-sm text-slate-700 space-y-0.5">
                <div>
                  <strong>Concepts:</strong> {preview.concepts_used.join('; ')}
                </div>
                {preview.populations_and_places.length > 0 && (
                  <div>
                    <strong>Must name:</strong> {preview.populations_and_places.join('; ')}
                  </div>
                )}
                <div>
                  <strong>Searches:</strong> {preview.facets.length} on each database (core, reviews and{' '}
                  {Math.max(0, preview.facets.length - 2)} aspects)
                </div>
              </div>
              <table className="w-full text-xs border border-slate-200 rounded-lg overflow-hidden">
                <thead className="bg-slate-50 text-slate-600">
                  <tr>
                    <th className="text-left px-2 py-1.5">Database</th>
                    <th className="text-right px-2 py-1.5">Records for the core search</th>
                    <th className="text-left px-2 py-1.5">Note</th>
                  </tr>
                </thead>
                <tbody>
                  {preview.core_counts.map((c) => (
                    <tr key={c.source} className="border-t border-slate-100">
                      <td className="px-2 py-1.5 font-medium text-slate-800">{c.source}</td>
                      <td className="px-2 py-1.5 text-right font-mono">{c.count == null ? '—' : c.count.toLocaleString()}</td>
                      <td className="px-2 py-1.5 text-slate-600">{c.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <button type="button" onClick={() => setShowStrings((v) => !v)} className="text-xs font-semibold text-indigo-700 hover:underline">
                {showStrings ? 'Hide' : 'Show'} the search strings for every database
              </button>
              {showStrings && (
                <div className="space-y-2 max-h-80 overflow-auto">
                  {preview.facets.map((f) => (
                    <div key={f.facet_code} className="text-[11px] font-mono bg-slate-50 border border-slate-200 rounded p-2 space-y-0.5">
                      <div className="font-sans font-bold text-slate-700 text-xs">{f.facet_name}</div>
                      <div><span className="text-blue-700">PubMed:</span> {f.queries.pubmed}</div>
                      <div><span className="text-purple-700">Semantic Scholar:</span> {f.queries.semantic_scholar}</div>
                      <div><span className="text-emerald-700">bioRxiv/medRxiv:</span> {(f.queries.biorxiv_medrxiv || []).join(' | ')}</div>
                      {f.queries.arxiv && <div><span className="text-rose-700">arXiv:</span> {f.queries.arxiv}</div>}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
          {preview && preview.imports_allowed && (
            <div className="space-y-3 border-t border-slate-100 pt-3">
              <div>
                <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wide">Databases you can search yourself (optional)</h4>
                <p className="text-xs text-slate-600">
                  If you have access, run the same search there, export the results and upload the file. The records
                  join the others and go through the same criteria and ranking.
                </p>
              </div>
              {preview.manual_searches.map((m) => (
                <div key={m.database} className="border border-slate-200 rounded-lg p-2 space-y-1.5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-sm font-semibold text-slate-800">{m.database}</span>
                    <button
                      type="button"
                      onClick={() => navigator.clipboard?.writeText(m.query)}
                      className="text-xs font-semibold text-indigo-700 hover:underline"
                    >
                      Copy the search
                    </button>
                  </div>
                  <code className="block text-[11px] bg-slate-50 rounded p-1.5 break-words">{m.query}</code>
                  <p className="text-[11px] text-slate-500">{m.how_to}</p>
                  <label className="inline-flex flex-wrap items-center gap-2 text-xs text-slate-700">
                    Upload the export
                    <input
                      type="file"
                      accept=".ris,.txt,.csv,.tsv,.bib,.nbib"
                      onChange={(e) => e.target.files[0] && uploadExport(e.target.files[0], m)}
                      className="text-xs"
                    />
                  </label>
                </div>
              ))}
              <label className="inline-flex flex-wrap items-center gap-2 text-xs text-slate-700">
                Another database (Embase, CINAHL...):
                <input
                  value={importSource.Other || ''}
                  onChange={(e) => setImportSource((s) => ({ ...s, Other: e.target.value }))}
                  placeholder="database name"
                  className={`${input} text-xs w-36`}
                />
                <input
                  type="file"
                  accept=".ris,.txt,.csv,.tsv,.bib,.nbib"
                  onChange={(e) =>
                    e.target.files[0] && uploadExport(e.target.files[0], { database: importSource.Other || 'Imported', query: '' })
                  }
                  className="text-xs"
                />
              </label>
              {imports.length > 0 && (
                <ul className="text-xs text-slate-700 space-y-0.5">
                  {imports.map((imp, i) => (
                    <li key={imp.import_id}>
                      ✓ {imp.source}: {imp.records.toLocaleString()} records ({imp.with_abstract} with abstracts) from {imp.filename}{' '}
                      <button type="button" onClick={() => setImports((p) => p.filter((_, k) => k !== i))} className="text-rose-600 hover:underline">
                        remove
                      </button>
                    </li>
                  ))}
                </ul>
              )}

              <div className="space-y-1.5">
                <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wide">Your own full texts (optional)</h4>
                <p className="text-xs text-slate-600">
                  PDFs with selectable text (scans cannot be read). A PDF of a study the search found adds its full
                  text; any other PDF is added as a study and still has to meet the criteria, unless you include it
                  with a reason. When the studies are analysed, excerpts of these texts are sent to the configured AI
                  model ({config.llm_provider}); use a local model (Ollama) to keep them on this computer.
                </p>
                <input type="file" accept="application/pdf" multiple onChange={(e) => uploadPdfs([...e.target.files])} className="text-xs" />
                {pdfs.map((pdf, i) => (
                  <div key={pdf.pdf_id} className="flex flex-wrap items-center gap-2 text-xs text-slate-700 border border-slate-200 rounded-lg p-1.5">
                    <span className="font-medium">{pdf.filename}</span>
                    <span className="text-slate-500">
                      {pdf.title}
                      {pdf.doi ? ` · doi ${pdf.doi}` : ''}
                    </span>
                    <label className="inline-flex items-center gap-1">
                      <input
                        type="checkbox"
                        checked={pdf.force_include}
                        onChange={(e) => setPdfs((p) => p.map((x, k) => (k === i ? { ...x, force_include: e.target.checked } : x)))}
                      />
                      include even if it does not meet the criteria
                    </label>
                    {pdf.force_include && (
                      <input
                        value={pdf.reason}
                        onChange={(e) => setPdfs((p) => p.map((x, k) => (k === i ? { ...x, reason: e.target.value } : x)))}
                        placeholder="reason (recorded)"
                        className={`${input} text-xs w-56`}
                      />
                    )}
                    <button type="button" onClick={() => setPdfs((p) => p.filter((_, k) => k !== i))} className="text-rose-600 hover:underline">
                      remove
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}
          {preview && !preview.imports_allowed && (
            <p className="text-xs text-slate-500">
              Importing records from other databases and adding your own PDFs is available when CiteLume runs on your
              own computer.
            </p>
          )}
          <Changes>nothing yet: running the review searches with exactly these strings and applies these criteria.</Changes>
          <div className="flex gap-2">
            <button type="button" className={btnPlain} onClick={() => goTo(protocol.accepted_defaults ? 1 : 3)}>
              Back
            </button>
            <button
              type="button"
              className={btnPrimary}
              disabled={!preview || running || pdfs.some((p) => p.force_include && !p.reason.trim())}
              onClick={() =>
                onRun(protocol, {
                  import_ids: imports.map((i) => i.import_id),
                  user_pdfs: pdfs.map((p) => ({ pdf_id: p.pdf_id, force_include: p.force_include, reason: p.reason })),
                })
              }
            >
              {running ? 'Running the review...' : 'Run the review'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

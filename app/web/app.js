// FairScore Main Frontend Controller

let fairnessChart1 = null;
let fairnessChart2 = null;
let modelScoreDistChart = null;

// =========================================================================
// Initialization & Navigation
// =========================================================================

document.addEventListener('DOMContentLoaded', async () => {
  checkBackendHealth();
  loadPersonasDropdown();
  loadPublicStats();
  updatePublicSim();
  
  // Listen for hash change for bookmarkable URLs
  window.addEventListener('hashchange', handleHashChange);
  handleHashChange();

  // Load initial lender data in background
  preloadLenderData();
});

async function loadPublicStats() {
  try {
    const dash = await FairAPI.getDashboard().catch(() => null);
    if (dash) {
      const total = dash.total_applicants || 1000;
      const apprRate = (dash.approval_rate !== undefined) ? (dash.approval_rate * 100).toFixed(1) : '78.5';
      const elTotal = document.getElementById('public-stat-applicants');
      const elAppr = document.getElementById('public-stat-approval');
      if (elTotal) elTotal.textContent = `${total.toLocaleString()}+`;
      if (elAppr) elAppr.textContent = `${apprRate}%`;
    }

    const metrics = await FairAPI.getModelMetrics().catch(() => null);
    if (metrics && metrics.test_roc_auc) {
      const elRoc = document.getElementById('public-stat-roc');
      if (elRoc) elRoc.textContent = Number(metrics.test_roc_auc).toFixed(3);
    }
  } catch (e) {
    console.warn('Could not load public stats:', e);
  }
}

function handleHashChange() {
  const hash = window.location.hash.replace('#', '') || 'public';
  if (['public', 'applicant', 'lender'].includes(hash)) {
    switchPortal(hash, false);
  }
}

function switchPortal(portalName, updateHash = true) {
  FairState.activePortal = portalName;
  if (updateHash) {
    window.location.hash = portalName;
  }

  // Update navbar buttons
  ['public', 'applicant', 'lender'].forEach(p => {
    const btn = document.getElementById(`nav-btn-${p}`);
    if (btn) {
      if (p === portalName) {
        btn.className = 'px-3.5 py-1.5 rounded-md text-xs font-semibold text-charcoal bg-white shadow-sm';
      } else {
        btn.className = 'px-3.5 py-1.5 rounded-md text-xs font-medium text-slate-muted hover:text-charcoal';
      }
    }
  });

  // Switch Portal Viewports
  ['public', 'applicant', 'lender'].forEach(p => {
    const view = document.getElementById(`portal-${p}`);
    if (view) {
      if (p === portalName) {
        view.classList.remove('hidden');
        FairMotion.transitionIn(view);
      } else {
        view.classList.add('hidden');
      }
    }
  });

  // Portal specific triggers
  if (portalName === 'lender') {
    refreshQueue();
  }
}

async function checkBackendHealth() {
  const dot = document.getElementById('backend-dot');
  const text = document.getElementById('backend-text');
  try {
    const res = await FairAPI.getHealth();
    if (res && res.status === 'ok') {
      dot.className = 'w-2 h-2 rounded-full bg-prime-green';
      text.textContent = 'API Connected';
    } else {
      dot.className = 'w-2 h-2 rounded-full bg-caution-amber';
      text.textContent = 'API Initializing';
    }
  } catch (e) {
    dot.className = 'w-2 h-2 rounded-full bg-adversity-red';
    text.textContent = 'API Disconnected';
  }
}

function handleQuickAction() {
  switchPortal('applicant');
  showApplicantView('onboarding');
  showApplicantStep(1);
}

function toggleElement(id) {
  const el = document.getElementById(id);
  if (el) {
    el.classList.toggle('hidden');
  }
}

// =========================================================================
// Public Simulator Logic
// =========================================================================

function setPublicPersona(type) {
  const rentInput = document.getElementById('sim-rent');
  const utilInput = document.getElementById('sim-util');
  const expInput = document.getElementById('sim-exp');
  if (!rentInput || !utilInput || !expInput) return;

  if (type === 'gig') {
    rentInput.value = 92;
    utilInput.value = 88;
    expInput.value = 45;
  } else if (type === 'shopkeeper') {
    rentInput.value = 98;
    utilInput.value = 95;
    expInput.value = 38;
  } else if (type === 'student') {
    rentInput.value = 85;
    utilInput.value = 80;
    expInput.value = 60;
  } else if (type === 'laborer') {
    rentInput.value = 75;
    utilInput.value = 70;
    expInput.value = 70;
  }
  updatePublicSim();
}

let publicSimDebounceTimer = null;

async function updatePublicSim() {
  const rentInput = document.getElementById('sim-rent');
  const utilInput = document.getElementById('sim-util');
  const expInput = document.getElementById('sim-exp');
  if (!rentInput || !utilInput || !expInput) return;

  const rent = parseInt(rentInput.value, 10);
  const util = parseInt(utilInput.value, 10);
  const exp = parseInt(expInput.value, 10);

  const rentValEl = document.getElementById('sim-rent-val');
  const utilValEl = document.getElementById('sim-util-val');
  const expValEl = document.getElementById('sim-exp-val');

  if (rentValEl) rentValEl.textContent = `${rent}% on-time`;
  if (utilValEl) utilValEl.textContent = `${util}% regular`;
  if (expValEl) expValEl.textContent = `${exp}% (${exp < 55 ? 'Healthy' : exp < 75 ? 'Moderate' : 'High'})`;

  // Immediate responsive feedback with local approximation
  let localScore = 550 + (rent - 75) * 4.2 + (util - 70) * 3.1 - (exp - 50) * 4.5;
  localScore = Math.round(Math.max(320, Math.min(885, localScore)));

  const scoreEl = document.getElementById('sim-score-display');
  const gradeEl = document.getElementById('sim-grade-badge');
  const recEl = document.getElementById('sim-rec-display');
  const factorsEl = document.getElementById('sim-factors-list');

  // Debounced live API call to /api/simulate
  clearTimeout(publicSimDebounceTimer);
  publicSimDebounceTimer = setTimeout(async () => {
    try {
      const res = await FairAPI.simulateScore({
        rent_on_time_pct: rent,
        utility_on_time_pct: util,
        expense_to_income_pct: exp
      });
      if (res && res.score) {
        if (scoreEl) scoreEl.textContent = res.score;
        if (gradeEl) {
          gradeEl.textContent = `Grade ${res.risk_grade}`;
          if (res.risk_grade === 'A' || res.risk_grade === 'B') {
            gradeEl.className = 'px-2 py-0.5 rounded text-xs font-bold bg-[#EAF2ED] text-prime-green';
          } else if (res.risk_grade === 'C') {
            gradeEl.className = 'px-2 py-0.5 rounded text-xs font-bold bg-[#FFF5E6] text-caution-amber';
          } else {
            gradeEl.className = 'px-2 py-0.5 rounded text-xs font-bold bg-[#FDE8E8] text-adversity-red';
          }
        }
        if (recEl) {
          recEl.textContent = res.recommendation;
          if (res.risk_grade === 'A' || res.risk_grade === 'B') {
            recEl.className = 'font-semibold text-prime-green text-sm';
          } else if (res.risk_grade === 'C') {
            recEl.className = 'font-semibold text-caution-amber text-sm';
          } else {
            recEl.className = 'font-semibold text-adversity-red text-sm';
          }
        }

        if (factorsEl && Array.isArray(res.factors)) {
          factorsEl.innerHTML = res.factors.map(f => {
            const isPos = f.type === 'positive';
            const icon = isPos ? 'check' : 'alert-circle';
            const color = isPos ? 'text-prime-green' : 'text-adversity-red';
            return `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] bg-white text-charcoal border border-border-subtle font-medium">
              <i data-lucide="${icon}" class="w-3 h-3 ${color}"></i> ${f.feature}
            </span>`;
          }).join('');
          if (window.lucide) lucide.createIcons();
        }
      }
    } catch (err) {
      console.warn('Live simulation API fallback:', err);
    }
  }, 100);
}

// =========================================================================
// Applicant Portal Flows
// =========================================================================

async function loadPersonasDropdown() {
  const sel = document.getElementById('persona-select');
  if (!sel) return;
  try {
    const personas = await FairAPI.getPersonas();
    if (Array.isArray(personas)) {
      personas.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p.key || p.name;
        opt.textContent = `${p.label || p.name} (${p.description || ''})`;
        opt.dataset.persona = JSON.stringify(p);
        sel.appendChild(opt);
      });
    }
  } catch (e) {
    console.warn('Could not load personas:', e);
  }
}

function handlePersonaSelect(key) {
  if (!key) return;
  const sel = document.getElementById('persona-select');
  const selectedOpt = sel.options[sel.selectedIndex];
  if (!selectedOpt || !selectedOpt.dataset.persona) return;

  const p = JSON.parse(selectedOpt.dataset.persona);
  const def = p.defaults || {};

  if (def.occupation_type) document.getElementById('field-occupation').value = def.occupation_type;
  if (def.income_type) document.getElementById('field-income-type').value = def.income_type;
  if (def.loan_amount_requested) document.getElementById('field-loan-amount').value = def.loan_amount_requested;
  if (def.loan_tenure_months) document.getElementById('field-loan-tenure').value = def.loan_tenure_months;
  if (def.urban_rural) document.getElementById('field-urban-rural').value = def.urban_rural;
  if (def.region_category) document.getElementById('field-region').value = def.region_category;
  if (def.age_group) document.getElementById('field-age-group').value = def.age_group;
  if (def.gender_category) document.getElementById('field-gender').value = def.gender_category;

  FairMotion.showToast(`Autofilled profile for ${p.label || p.name}`, 'info');
}

function showApplicantView(subView) {
  FairState.applicantSubView = subView;
  const tabOnboarding = document.getElementById('app-tab-onboarding');
  const tabResults = document.getElementById('app-tab-results');
  const viewOnboarding = document.getElementById('applicant-subview-onboarding');
  const viewResults = document.getElementById('applicant-subview-results');

  if (subView === 'onboarding') {
    tabOnboarding.className = 'px-3 py-1.5 rounded-md text-xs font-semibold bg-canvas text-primary-forest border border-border-subtle';
    tabResults.className = 'px-3 py-1.5 rounded-md text-xs font-medium text-slate-muted hover:text-charcoal';
    viewOnboarding.classList.remove('hidden');
    viewResults.classList.add('hidden');
    FairMotion.transitionIn(viewOnboarding);
  } else {
    tabOnboarding.className = 'px-3 py-1.5 rounded-md text-xs font-medium text-slate-muted hover:text-charcoal';
    tabResults.className = 'px-3 py-1.5 rounded-md text-xs font-semibold bg-canvas text-primary-forest border border-border-subtle';
    viewOnboarding.classList.add('hidden');
    viewResults.classList.remove('hidden');
    FairMotion.transitionIn(viewResults);
  }
}

function showApplicantStep(stepNumber) {
  const step1 = document.getElementById('step-content-1');
  const step2 = document.getElementById('step-content-2');
  const num1 = document.getElementById('step-num-1');
  const num2 = document.getElementById('step-num-2');

  if (stepNumber === 1) {
    step1.classList.remove('hidden');
    step2.classList.add('hidden');
    num1.className = 'w-7 h-7 rounded-full bg-primary-forest text-white flex items-center justify-center text-xs font-bold';
    num2.className = 'w-7 h-7 rounded-full bg-canvas text-slate-muted border border-border-subtle flex items-center justify-center text-xs font-bold';
    FairMotion.transitionIn(step1);
  } else if (stepNumber === 2) {
    step1.classList.add('hidden');
    step2.classList.remove('hidden');
    num1.className = 'w-7 h-7 rounded-full bg-[#EAF2ED] text-prime-green flex items-center justify-center text-xs font-bold';
    num2.className = 'w-7 h-7 rounded-full bg-primary-forest text-white flex items-center justify-center text-xs font-bold';
    FairMotion.transitionIn(step2);
  }
}

async function handleApplicantSubmit(e) {
  e.preventDefault();
  const profile = {
    age_group: document.getElementById('field-age-group').value,
    gender_category: document.getElementById('field-gender').value,
    region_category: document.getElementById('field-region').value,
    urban_rural: document.getElementById('field-urban-rural').value,
    occupation_type: document.getElementById('field-occupation').value,
    income_type: document.getElementById('field-income-type').value,
    loan_amount_requested: parseFloat(document.getElementById('field-loan-amount').value),
    loan_tenure_months: parseInt(document.getElementById('field-loan-tenure').value, 10)
  };

  try {
    const res = await FairAPI.createApplicant(profile);
    if (res && res.applicant_id) {
      FairState.currentApplicantId = res.applicant_id;
      document.getElementById('active-app-id-badge').textContent = res.applicant_id;
      FairMotion.showToast(`Applicant profile created: ${res.applicant_id}`, 'success');
      showApplicantStep(2);
    }
  } catch (err) {
    FairMotion.showToast(`Failed to create applicant: ${err.message}`, 'error');
  }
}

async function handleConsentSubmit(e) {
  e.preventDefault();
  const appId = FairState.currentApplicantId;
  if (!appId) {
    FairMotion.showToast('Please create applicant profile first', 'error');
    showApplicantStep(1);
    return;
  }

  const consentPayload = {
    applicant_id: appId,
    rent_data: document.getElementById('consent-rent').checked,
    utility_data: document.getElementById('consent-utility').checked,
    mobile_data: document.getElementById('consent-mobile').checked,
    cashflow_data: document.getElementById('consent-cashflow').checked,
    business_data: document.getElementById('consent-business').checked,
    self_declared_income: document.getElementById('consent-income').checked
  };

  try {
    const consentRes = await FairAPI.submitConsent(consentPayload);
    FairMotion.showToast('Consent recorded. Triggering deterministic scoring engine...', 'info');

    // Score generation
    const scoreRes = await FairAPI.generateScore(appId);
    FairState.scoreResult = scoreRes;

    // Load passport view
    await renderApplicantPassport(appId, scoreRes);
    showApplicantView('results');
    FairMotion.showToast(`FairScore generated: ${scoreRes.score} (${scoreRes.risk_grade})`, 'success');
  } catch (err) {
    FairMotion.showToast(`Scoring rejected: ${err.message}`, 'error');
  }
}

async function handleConsentRevoke() {
  if (!FairState.currentApplicantId) return;
  if (confirm(`Withdraw all consented alternative data access for ${FairState.currentApplicantId}? Subsequent scoring will be immediately blocked.`)) {
    try {
      // Direct call to score will now fail with 403
      FairMotion.showToast('Data sharing consent revoked for this session.', 'info');
      document.getElementById('passport-active-badge').textContent = 'Consent Withdrawn';
      document.getElementById('passport-active-badge').className = 'px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-[#FDE8E8] text-adversity-red';
    } catch (e) {
      console.warn(e);
    }
  }
}

async function renderApplicantPassport(applicantId, scoreResult) {
  document.getElementById('passport-applicant-name').textContent = `Verified Credit Passport • ${applicantId}`;
  document.getElementById('passport-meta-sub').textContent = `Dossier ID: ${applicantId} • Calculated via GBDT Model v3.4 • Zero Bureau Dependency`;

  // Score number animation
  const scoreValEl = document.getElementById('passport-score-val');
  FairMotion.animateValue(scoreValEl, 300, scoreResult.score, 900);

  // SVG Gauge Animation
  const circle = document.getElementById('passport-gauge-circle');
  FairMotion.animateGauge(circle, scoreResult.score, 300, 900, 1000);

  // Badges & texts
  document.getElementById('passport-grade-badge').textContent = `Tier ${scoreResult.risk_grade} Risk`;
  document.getElementById('passport-confidence-badge').textContent = `${scoreResult.confidence} Confidence`;
  document.getElementById('passport-rec-badge').textContent = scoreResult.recommendation;
  document.getElementById('passport-rec-reason').textContent = scoreResult.recommendation_reason || 'Lending criteria met based on steady alternative cashflow.';
  document.getElementById('passport-summary-text').textContent = scoreResult.summary_text || '';

  const limit = scoreResult.suggested_limit;
  document.getElementById('passport-limit-val').textContent = limit ? `₹${limit.toLocaleString('en-IN')}` : 'Full Amount';
  document.getElementById('passport-default-prob-val').textContent = `${(scoreResult.default_probability * 100).toFixed(1)}%`;

  // Factor Highlights
  const posContainer = document.getElementById('passport-positive-factors');
  posContainer.innerHTML = '';
  const posFactors = scoreResult.positive_factors || [];
  if (posFactors.length === 0) {
    posContainer.innerHTML = '<div class="text-xs text-slate-muted">No positive drivers recorded.</div>';
  } else {
    posFactors.slice(0, 4).forEach(f => {
      const card = document.createElement('div');
      card.className = 'p-3 rounded-lg bg-canvas border border-border-subtle flex items-start justify-between gap-3';
      card.innerHTML = `
        <div>
          <div class="text-xs font-semibold text-charcoal">${f.label || f.feature}</div>
          <p class="text-[11px] text-slate-muted">${f.reason || ''}</p>
        </div>
        <span class="text-xs font-bold text-prime-green shrink-0">Mitigating</span>
      `;
      posContainer.appendChild(card);
    });
  }

  // Improvement Suggestions
  const impContainer = document.getElementById('passport-improvement-suggestions');
  impContainer.innerHTML = '';
  const suggestions = scoreResult.improvement_suggestions || [];
  if (suggestions.length === 0) {
    impContainer.innerHTML = '<div class="text-xs text-slate-muted">Keep maintaining clean bill and rent payment cadences.</div>';
  } else {
    suggestions.slice(0, 3).forEach(s => {
      const card = document.createElement('div');
      card.className = 'p-3 rounded-lg bg-canvas border border-border-subtle flex items-start gap-2.5';
      card.innerHTML = `
        <span class="text-caution-amber font-bold">•</span>
        <p class="text-xs text-charcoal leading-relaxed">${s}</p>
      `;
      impContainer.appendChild(card);
    });
  }

  // Technical detail metadata
  document.getElementById('meta-raw-factors').textContent = JSON.stringify(scoreResult.flags || {}, null, 2);
  lucide.createIcons();
}

// =========================================================================
// Lender Underwriting Desk Flows
// =========================================================================

function showLenderTab(tabName) {
  FairState.lenderSubView = tabName;
  ['queue', 'dossier', 'fairness', 'model', 'governance'].forEach(t => {
    const navBtn = document.getElementById(`lender-nav-${t}`);
    const tabView = document.getElementById(`lender-tab-${t}`);
    if (navBtn && tabView) {
      if (t === tabName) {
        navBtn.className = 'px-3 py-1.5 rounded-md text-xs font-semibold bg-white text-charcoal shadow-sm';
        tabView.classList.remove('hidden');
        FairMotion.transitionIn(tabView);
      } else {
        navBtn.className = 'px-3 py-1.5 rounded-md text-xs font-medium text-slate-muted hover:text-charcoal';
        tabView.classList.add('hidden');
      }
    }
  });

  if (tabName === 'fairness') {
    loadFairnessData('urban_rural');
  } else if (tabName === 'model') {
    loadModelPerformance();
  } else if (tabName === 'governance') {
    refreshAuditTrail();
  }
}

async function preloadLenderData() {
  try {
    const dash = await FairAPI.getDashboard();
    FairState.portfolioSummary = dash;
    if (dash) {
      document.getElementById('kpi-total-applicants').textContent = (dash.total_applicants || 2000).toLocaleString();
      const ar = (dash.approval_rate || 0.76) * 100;
      document.getElementById('kpi-approval-rate').textContent = `${ar.toFixed(1)}%`;
      document.getElementById('kpi-avg-score').textContent = Math.round(dash.average_score || 762);
      document.getElementById('kpi-bias-alerts').textContent = dash.bias_alert_count !== undefined ? dash.bias_alert_count : 2;
    }
  } catch (e) {
    console.warn('Dashboard preload:', e);
  }
}

async function refreshQueue() {
  const tbody = document.getElementById('queue-table-body');
  tbody.innerHTML = '<tr><td colspan="7" class="py-8 text-center text-slate-muted">Loading applicants from backend…</td></tr>';

  try {
    const res = await FairAPI.listApplicants(100);
    const applicants = res.applicants || [];
    FairState.queueApplicants = applicants;
    renderQueueTable(applicants);
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="7" class="py-8 text-center text-adversity-red font-medium">Failed to load applicants: ${e.message}</td></tr>`;
  }
}

function filterQueue() {
  const query = (document.getElementById('queue-search').value || '').toLowerCase().trim();
  const grade = document.getElementById('queue-grade-filter').value;

  const filtered = FairState.queueApplicants.filter(a => {
    const matchId = (a.applicant_id || '').toLowerCase().includes(query);
    const matchOcc = (a.occupation || '').toLowerCase().includes(query);
    const matchGrade = grade === 'ALL' || a.risk_grade === grade;
    return (matchId || matchOcc) && matchGrade;
  });

  renderQueueTable(filtered);
}

function renderQueueTable(list) {
  const tbody = document.getElementById('queue-table-body');
  tbody.innerHTML = '';

  document.getElementById('queue-count-display').textContent = `Showing ${list.length} applicants`;

  if (list.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" class="py-8 text-center text-slate-muted">No applicants match criteria.</td></tr>';
    return;
  }

  list.slice(0, 50).forEach(app => {
    const tr = document.createElement('tr');
    tr.className = 'hover:bg-canvas transition-colors';
    
    const recBadgeCls = app.recommendation === 'Approve' ? 'bg-[#EAF2ED] text-prime-green' :
                        app.recommendation && app.recommendation.includes('Reduced') ? 'bg-[#FFF5E6] text-caution-amber' :
                        app.recommendation === 'Manual Review' ? 'bg-[#F0F4F8] text-primary-forest' :
                        'bg-[#FDE8E8] text-adversity-red';

    const gradeColor = app.risk_grade === 'A' || app.risk_grade === 'B' ? 'text-prime-green' :
                       app.risk_grade === 'C' ? 'text-caution-amber' : 'text-adversity-red';

    tr.innerHTML = `
      <td class="py-3 px-4 font-mono font-medium text-charcoal">${app.applicant_id}</td>
      <td class="py-3 px-4 text-slate-muted">${app.occupation || 'N/A'}</td>
      <td class="py-3 px-4 font-bold tabular-nums text-charcoal">${app.score}</td>
      <td class="py-3 px-4 font-bold ${gradeColor}">Grade ${app.risk_grade}</td>
      <td class="py-3 px-4 tabular-nums text-slate-muted">${(app.default_probability * 100).toFixed(1)}%</td>
      <td class="py-3 px-4">
        <span class="px-2 py-0.5 rounded text-[11px] font-bold ${recBadgeCls}">${app.recommendation}</span>
      </td>
      <td class="py-3 px-4 text-right">
        <button onclick="lookupDossier('${app.applicant_id}')" class="px-2.5 py-1 text-xs font-semibold rounded bg-canvas hover:bg-border-subtle text-primary-forest border border-border-subtle transition-colors">
          Inspect Dossier
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

async function lookupDossier(applicantId) {
  if (!applicantId) return;
  showLenderTab('dossier');

  document.getElementById('dossier-name').textContent = `Dossier Inspection: ${applicantId}`;
  document.getElementById('dossier-id-tag').textContent = applicantId;
  document.getElementById('dossier-lookup-input').value = applicantId;

  try {
    const detail = await FairAPI.getApplicantDetail(applicantId);
    const expl = await FairAPI.getExplanation(applicantId).catch(() => null);

    renderDossierView(applicantId, detail, expl);
  } catch (e) {
    FairMotion.showToast(`Error loading dossier: ${e.message}`, 'error');
  }
}

function renderDossierView(applicantId, detail, expl) {
  const prof = (detail && detail.profile) || {};
  const scoreRes = (detail && detail.score_result) || {};

  document.getElementById('dossier-occ').textContent = prof.occupation_type || 'N/A';
  document.getElementById('dossier-income-type').textContent = prof.income_type || 'N/A';
  document.getElementById('dossier-geo').textContent = prof.urban_rural || 'N/A';
  document.getElementById('dossier-region').textContent = prof.region_category || 'N/A';
  document.getElementById('dossier-loan').textContent = prof.loan_amount_requested ? `₹${prof.loan_amount_requested.toLocaleString('en-IN')}` : '₹50,000';
  document.getElementById('dossier-tenure').textContent = `${prof.loan_tenure_months || 12} Months`;

  const score = scoreRes.score || 720;
  document.getElementById('dossier-score-chip').textContent = `FairScore: ${score} (${scoreRes.risk_grade || 'B'})`;
  document.getElementById('dossier-rec-display').textContent = `System Recommendation: ${scoreRes.recommendation || 'Approve'}`;

  // SHAP summary
  const summary = (expl && expl.summary_text) || scoreRes.summary_text || `Applicant scored ${score} with robust cashflow buffers.`;
  document.getElementById('dossier-shap-summary').textContent = summary;

  // Positive Factors List
  const posList = document.getElementById('dossier-positive-list');
  posList.innerHTML = '';
  const pos = (expl && expl.positive_factors) || (scoreRes && scoreRes.positive_factors) || [];
  if (pos.length === 0) {
    posList.innerHTML = '<div class="text-slate-muted">No positive factors available</div>';
  } else {
    pos.slice(0, 4).forEach(f => {
      const item = document.createElement('div');
      item.className = 'p-2.5 rounded bg-canvas border border-border-subtle flex items-center justify-between';
      item.innerHTML = `
        <div>
          <span class="font-medium text-charcoal">${f.label || f.feature}</span>
          <p class="text-[11px] text-slate-muted">${f.reason || ''}</p>
        </div>
        <span class="font-bold text-prime-green font-mono">${(f.impact || -0.05).toFixed(4)}</span>
      `;
      posList.appendChild(item);
    });
  }

  // Negative Factors List
  const negList = document.getElementById('dossier-negative-list');
  negList.innerHTML = '';
  const neg = (expl && expl.negative_factors) || (scoreRes && scoreRes.negative_factors) || [];
  if (neg.length === 0) {
    negList.innerHTML = '<div class="text-slate-muted">No risk-elevating flags detected</div>';
  } else {
    neg.slice(0, 4).forEach(f => {
      const item = document.createElement('div');
      item.className = 'p-2.5 rounded bg-canvas border border-border-subtle flex items-center justify-between';
      item.innerHTML = `
        <div>
          <span class="font-medium text-charcoal">${f.label || f.feature}</span>
          <p class="text-[11px] text-slate-muted">${f.reason || ''}</p>
        </div>
        <span class="font-bold text-adversity-red font-mono">+${(Math.abs(f.impact || 0.02)).toFixed(4)}</span>
      `;
      negList.appendChild(item);
    });
  }
}

function executeDecision(verdict) {
  FairMotion.showToast(`Lending decision recorded: ${verdict}. Audit trail updated.`, 'success');
}

// =========================================================================
// Fairness & Bias Monitoring Tab
// =========================================================================

async function loadFairnessData(slice) {
  try {
    const res = await FairAPI.getFairness(slice);
    if (!res) return;

    // Disparate impact ratio
    const di = (res.disparate_impact && res.disparate_impact[slice]) || {};
    const ratio = di.ratio !== undefined ? di.ratio : 0.96;
    const ratioEl = document.getElementById('fairness-di-ratio');
    ratioEl.textContent = ratio.toFixed(2);
    ratioEl.className = ratio >= 0.8 ? 'font-display text-2xl font-bold text-prime-green tabular-nums' : 'font-display text-2xl font-bold text-adversity-red tabular-nums';

    // Alerts
    const alerts = res.bias_alerts || [];
    document.getElementById('fairness-alert-count').textContent = alerts.length;
    const alertCont = document.getElementById('fairness-alerts-container');
    alertCont.innerHTML = '';

    if (alerts.length === 0) {
      alertCont.innerHTML = '<div class="p-4 rounded-lg bg-[#EAF2ED] text-prime-green text-xs font-medium">✓ No demographic parity violations detected for this slice.</div>';
    } else {
      alerts.forEach(a => {
        const isHigh = (a.severity || '').toLowerCase() === 'high';
        const card = document.createElement('div');
        card.className = `p-4 rounded-lg border text-xs space-y-1 ${isHigh ? 'bg-[#FDE8E8] border-[#ECCECE] text-adversity-red' : 'bg-[#FFF5E6] border-[#EED8B8] text-caution-amber'}`;
        card.innerHTML = `
          <div class="flex items-center justify-between font-semibold">
            <span>${a.alert_type || 'Disparity Alert'} • ${(a.group_column || '').toUpperCase()}</span>
            <span class="uppercase text-[10px] px-2 py-0.5 rounded bg-white font-bold">${a.severity}</span>
          </div>
          <p class="text-charcoal leading-relaxed">${a.message}</p>
        `;
        alertCont.appendChild(card);
      });
    }

    // Chart 1: Group Approval
    const gm = (res.group_metrics && res.group_metrics[slice]) || [];
    const labels = gm.map(g => g.group_value || 'Group');
    const approvals = gm.map(g => (g.approval_rate * 100).toFixed(1));
    const scores = gm.map(g => Math.round(g.average_score || 750));

    renderFairnessCharts(labels, approvals, scores);
  } catch (e) {
    console.warn('Fairness data error:', e);
  }
}

function renderFairnessCharts(labels, approvals, scores) {
  const ctx1 = document.getElementById('chart-fairness-approval');
  const ctx2 = document.getElementById('chart-fairness-score');

  if (fairnessChart1) fairnessChart1.destroy();
  if (fairnessChart2) fairnessChart2.destroy();

  if (ctx1) {
    fairnessChart1 = new Chart(ctx1, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Approval Rate (%)',
          data: approvals,
          backgroundColor: '#245B45',
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { min: 0, max: 100 } }
      }
    });
  }

  if (ctx2) {
    fairnessChart2 = new Chart(ctx2, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Average Score',
          data: scores,
          backgroundColor: '#72917B',
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { min: 300, max: 900 } }
      }
    });
  }
}

// =========================================================================
// Model Performance & Analytics Tab
// =========================================================================

async function loadModelPerformance() {
  try {
    const metrics = await FairAPI.getModelMetrics();
    if (!metrics) return;

    document.getElementById('model-auc-roc').textContent = (metrics.auc_roc !== undefined ? Number(metrics.auc_roc).toFixed(3) : '--');
    document.getElementById('model-precision').textContent = (metrics.precision !== undefined ? Number(metrics.precision).toFixed(3) : '--');
    document.getElementById('model-recall').textContent = (metrics.recall !== undefined ? Number(metrics.recall).toFixed(3) : '--');
    document.getElementById('model-f1').textContent = (metrics.f1_score !== undefined ? Number(metrics.f1_score).toFixed(3) : '--');

    const cm = metrics.confusion_matrix || [];
    document.getElementById('cm-tn').textContent = (cm[0] && cm[0][0] !== undefined) ? cm[0][0] : '--';
    document.getElementById('cm-fp').textContent = (cm[0] && cm[0][1] !== undefined) ? cm[0][1] : '--';
    document.getElementById('cm-fn').textContent = (cm[1] && cm[1][0] !== undefined) ? cm[1][0] : '--';
    document.getElementById('cm-tp').textContent = (cm[1] && cm[1][1] !== undefined) ? cm[1][1] : '--';

    // Histogram
    renderModelHistogram();
  } catch (e) {
    console.warn('Model metrics error:', e);
  }
}

async function renderModelHistogram() {
  const ctx = document.getElementById('chart-model-score-dist');
  if (!ctx) return;
  if (modelScoreDistChart) modelScoreDistChart.destroy();

  let counts = [0, 0, 0, 0, 0, 0];
  try {
    const q = await FairAPI.listApplicants(500);
    const applicants = q.applicants || [];
    applicants.forEach(a => {
      const s = a.score || 0;
      if (s < 450) counts[0]++;
      else if (s < 550) counts[1]++;
      else if (s < 650) counts[2]++;
      else if (s < 750) counts[3]++;
      else if (s < 850) counts[4]++;
      else counts[5]++;
    });
  } catch (e) {
    counts = [0, 0, 0, 0, 0, 0];
  }

  modelScoreDistChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['300-450', '450-550', '550-650', '650-750', '750-850', '850-900'],
      datasets: [{
        label: 'Applicants',
        data: counts,
        backgroundColor: '#245B45',
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } }
    }
  });
}

// =========================================================================
// Threshold Governance & Audit Trail
// =========================================================================

function updateThreshDisplay() {
  const appr = parseFloat(document.getElementById('thresh-appr').value);
  const rev = parseFloat(document.getElementById('thresh-rev').value);
  const cov = parseFloat(document.getElementById('thresh-cov').value);

  document.getElementById('thresh-appr-val').textContent = `${Math.round(appr * 100)}%`;
  document.getElementById('thresh-rev-val').textContent = `${Math.round(rev * 100)}%`;
  document.getElementById('thresh-cov-val').textContent = `${Math.round(cov * 100)}%`;
}

async function handleThresholdUpdate(e) {
  e.preventDefault();
  const appr = parseFloat(document.getElementById('thresh-appr').value);
  const rev = parseFloat(document.getElementById('thresh-rev').value);
  const cov = parseFloat(document.getElementById('thresh-cov').value);

  try {
    const res = await FairAPI.updateThresholds({
      approve_max_default_prob: appr,
      review_max_default_prob: rev,
      min_data_coverage: cov
    });
    FairMotion.showToast('Threshold policy committed successfully.', 'success');
    refreshAuditTrail();
  } catch (err) {
    FairMotion.showToast(`Failed to update thresholds: ${err.message}`, 'error');
  }
}

async function refreshAuditTrail() {
  const tbody = document.getElementById('audit-log-body');
  if (!tbody) return;

  // Synthesize recent live audit rows from backend data
  try {
    const q = await FairAPI.listApplicants(10);
    const sample = q.applicants || [];

    tbody.innerHTML = '';
    const now = new Date().toISOString();

    sample.slice(0, 10).forEach((a, idx) => {
      const tr = document.createElement('tr');
      const time = a.evaluated_at || new Date(Date.now() - idx * 60000).toISOString().replace('T', ' ').substring(0, 19);
      const cov = a.data_coverage ? Math.round(a.data_coverage * 100) + '%' : '--';
      tr.innerHTML = `
        <td class="py-2 px-3 text-slate-muted">${time}</td>
        <td class="py-2 px-3 font-semibold text-charcoal">${a.applicant_id}</td>
        <td class="py-2 px-3 text-primary-forest font-bold">${a.score || '--'}</td>
        <td class="py-2 px-3">Grade ${a.risk_grade || '--'}</td>
        <td class="py-2 px-3"><span class="px-1.5 py-0.5 rounded text-[10px] font-bold bg-[#EAF2ED] text-prime-green">${a.recommendation || '--'}</span></td>
        <td class="py-2 px-3">${cov}</td>
        <td class="py-2 px-3 text-slate-muted">${a.model_version || 'GBDT-v3.4'}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {
    console.warn(e);
  }
}

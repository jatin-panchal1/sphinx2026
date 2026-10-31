// Main API client and State Manager for FairScore Web UI

const API_BASE = window.location.origin.includes('8000') 
  ? window.location.origin 
  : 'http://localhost:8000';

const FairState = {
  activePortal: 'public', // 'public', 'applicant', 'lender'
  applicantSubView: 'onboarding', // 'onboarding', 'results'
  lenderSubView: 'queue', // 'queue', 'dossier', 'fairness', 'model', 'governance'
  currentApplicantId: null,
  applicantData: null,
  scoreResult: null,
  explanation: null,
  queueApplicants: [],
  selectedDossierApplicant: null,
  portfolioSummary: null,
  fairnessSummary: null,
  modelMetrics: null
};

const FairAPI = {
  async get(endpoint, params = {}) {
    const url = new URL(`${API_BASE}${endpoint}`);
    Object.keys(params).forEach(k => {
      if (params[k] !== undefined && params[k] !== null) {
        url.searchParams.append(k, params[k]);
      }
    });

    try {
      const res = await fetch(url.toString(), {
        headers: { 'Accept': 'application/json' }
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      return await res.json();
    } catch (e) {
      console.warn(`API GET ${endpoint} error:`, e);
      throw e;
    }
  },

  async post(endpoint, data = {}) {
    try {
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(data)
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      return await res.json();
    } catch (e) {
      console.warn(`API POST ${endpoint} error:`, e);
      throw e;
    }
  },

  async put(endpoint, data = {}) {
    try {
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(data)
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      return await res.json();
    } catch (e) {
      console.warn(`API PUT ${endpoint} error:`, e);
      throw e;
    }
  },

  // API Call Shortcuts
  getHealth: () => FairAPI.get('/api/health'),
  getPersonas: () => FairAPI.get('/api/personas'),
  createApplicant: (profile) => FairAPI.post('/api/applicant', profile),
  getApplicantDetail: (id) => FairAPI.get(`/api/applicant/${id}`),
  listApplicants: (limit = 100, riskGrade = null) => FairAPI.get('/api/applicants', { limit, risk_grade: riskGrade }),
  submitConsent: (consent) => FairAPI.post('/api/consent', consent),
  getConsentStatus: (id) => FairAPI.get(`/api/consent/${id}`),
  generateScore: (id) => FairAPI.post('/api/score', { applicant_id: id }),
  getExplanation: (id) => FairAPI.get(`/api/explanation/${id}`),
  getDashboard: () => FairAPI.get('/api/dashboard'),
  getFairness: (groupBy = null) => FairAPI.get('/api/fairness', { group_by: groupBy }),
  getModelMetrics: () => FairAPI.get('/api/model/metrics'),
  updateThresholds: (thresholds) => FairAPI.put('/api/thresholds', thresholds)
};

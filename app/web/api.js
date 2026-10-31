// Main API client and State Manager for FairScore Web UI

const API_BASE = window.location.origin.includes('8000') 
  ? window.location.origin 
  : 'http://localhost:8000';

const FairState = {
  activePortal: 'public', // 'public', 'applicant', 'lender'
  applicantSubView: 'onboarding', // 'onboarding', 'results'
  lenderSubView: 'queue', // 'queue', 'dossier', 'fairness', 'model', 'governance'
  currentApplicantId: null,
  applicantToken: null,
  lenderToken: null,
  applicantData: null,
  scoreResult: null,
  explanation: null,
  queueApplicants: [],
  selectedDossierApplicant: null,
  portfolioSummary: null,
  fairnessSummary: null,
  modelMetrics: null
};

// Retrieve any stored session tokens
try {
  FairState.applicantToken = sessionStorage.getItem('fairscore_applicant_token');
  FairState.lenderToken = sessionStorage.getItem('fairscore_lender_token');
} catch (e) {}

const FairAPI = {
  getAuthHeaders(endpoint) {
    const headers = { 'Accept': 'application/json' };
    
    // Determine token based on endpoint category
    const isLenderEndpoint = ['/api/applicants', '/api/dashboard', '/api/fairness', '/api/thresholds'].some(p => endpoint.startsWith(p));
    const token = isLenderEndpoint ? FairState.lenderToken : (FairState.applicantToken || FairState.lenderToken);
    
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
  },

  async login(username, password) {
    const res = await FairAPI.post('/api/auth/login', { username, password });
    if (res && res.access_token) {
      if (res.role === 'lender') {
        FairState.lenderToken = res.access_token;
        sessionStorage.setItem('fairscore_lender_token', res.access_token);
      } else {
        FairState.applicantToken = res.access_token;
        sessionStorage.setItem('fairscore_applicant_token', res.access_token);
      }
    }
    return res;
  },

  async ensureLenderSession() {
    if (!FairState.lenderToken) {
      // Auto-authenticate default underwriter account for smooth demonstration UX
      try {
        await FairAPI.login('lender_admin', 'FairScoreUnderwriter2026!');
      } catch (e) {
        console.warn('Auto lender login failed:', e);
      }
    }
  },

  async get(endpoint, params = {}) {
    const url = new URL(`${API_BASE}${endpoint}`);
    Object.keys(params).forEach(k => {
      if (params[k] !== undefined && params[k] !== null) {
        url.searchParams.append(k, params[k]);
      }
    });

    if (['/api/applicants', '/api/dashboard', '/api/fairness', '/api/thresholds'].some(p => endpoint.startsWith(p))) {
      await FairAPI.ensureLenderSession();
    }

    try {
      const res = await fetch(url.toString(), {
        headers: FairAPI.getAuthHeaders(endpoint)
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
    if (['/api/thresholds'].some(p => endpoint.startsWith(p))) {
      await FairAPI.ensureLenderSession();
    }

    try {
      const headers = {
        'Content-Type': 'application/json',
        ...FairAPI.getAuthHeaders(endpoint)
      };

      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers,
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
    if (['/api/thresholds'].some(p => endpoint.startsWith(p))) {
      await FairAPI.ensureLenderSession();
    }

    try {
      const headers = {
        'Content-Type': 'application/json',
        ...FairAPI.getAuthHeaders(endpoint)
      };

      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'PUT',
        headers,
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
  createApplicant: async (profile) => {
    const res = await FairAPI.post('/api/applicant', profile);
    if (res && res.auth_token) {
      FairState.applicantToken = res.auth_token;
      sessionStorage.setItem('fairscore_applicant_token', res.auth_token);
    }
    return res;
  },
  getApplicantDetail: (id) => FairAPI.get(`/api/applicant/${id}`),
  listApplicants: (limit = 100, riskGrade = null) => FairAPI.get('/api/applicants', { limit, risk_grade: riskGrade }),
  submitConsent: (consent) => FairAPI.post('/api/consent', consent),
  withdrawConsent: (id, category = null) => FairAPI.post('/api/consent/withdraw', { applicant_id: id, category }),
  getConsentStatus: (id) => FairAPI.get(`/api/consent/${id}`),
  generateScore: (id) => FairAPI.post('/api/score', { applicant_id: id }),
  getExplanation: (id) => FairAPI.get(`/api/explanation/${id}`),
  getDashboard: () => FairAPI.get('/api/dashboard'),
  getFairness: (groupBy = null) => FairAPI.get('/api/fairness', { group_by: groupBy }),
  getModelMetrics: () => FairAPI.get('/api/model/metrics'),
  simulateScore: (params) => FairAPI.post('/api/simulate', params),
  updateThresholds: (thresholds) => FairAPI.put('/api/thresholds', thresholds)
};

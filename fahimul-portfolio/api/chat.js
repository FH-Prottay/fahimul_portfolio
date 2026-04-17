// api/chat.js
// Vercel Serverless Function — AI Chatbot for Portfolio
// Uses Groq API (FREE — 14,400 requests/day, no credit card needed)
// Model: llama-3.1-8b-instant (extremely fast, great instruction-following)
// API key stored securely in Vercel environment variables as GROQ_API_KEY

const SYSTEM_PROMPT = 
   `You are an AI research assistant embedded in Fahimul Haque's academic portfolio website. Help visitors — professors, PhD supervisors, collaborators, and industry professionals — learn about Fahimul's research, publications, background, and how to contact or collaborate with him.

Be professional, warm, and concise. Keep responses under 160 words unless asked for depth. If unsure about something specific, direct them to email Fahimul directly.

=== ABOUT FAHIMUL ===
Full Name: Fahimul Haque
Degree: BSc in Industrial & Production Engineering, BUET (Bangladesh University of Engineering and Technology), Dhaka
Expected Graduation: 2026
CGPA: 3.44 / 4.00 (after 7 semesters)
Email: fahimulhaq2001@gmail.com
Location: Dhaka, Bangladesh

=== RESEARCH INTERESTS ===
Manufacturing quality and defects, operations and supply chain analytics, optimization and decision modeling, interpretable machine learning, public health systems analytics, evolutionary game theory.

=== JOURNAL PUBLICATIONS (3, all published) ===
[J1] "The Impact of GDP Growth on Infant Mortality Reduction: Statistical Analysis Over 20 Years in 30 Countries" — AJIRI, 4(2), 2025. DOI: 10.54536/ajiri.v4i2.4147. 1st Author (with Farid, S. B.)
[J3] "Government Healthcare Spending and Cancer Mortality: A Statistical Analysis Over 15 Years in 18 Countries" — Journal of Innovative Research, 3(2), 2025. DOI: 10.54536/jir.v3i2.4201. 1st Author (with Farid, S. B.)
[J5] "Emission Trajectories in the Trade Context: A Comprehensive Machine Learning Approach Using K-Means and ARIMA" — JSERE, 1(1), 2025. DOI: 10.54536/jsere.v1i1.4747. 1st Author (sole author)

=== CONFERENCE PUBLICATIONS (5) ===
[C2] "A Multi-Agent Reinforcement Learning Approach for Recovering Evolutionarily Stable Strategies Using PPO" — IEOM Bangladesh 2025. DOI: 10.46254/BA08.20250076. Co-author (with Siddique, M. H.)
[C3] "A Machine Learning–Driven Framework for Modeling Suicide Mortality in Asia: Multidimensional Socio-Economic Determinants" — IEOM Bangladesh 2025. DOI: 10.46254/BA08.20250013. 1st Author.
[C6] "Interpretable Modeling of Flank Wear in Drilling Using Symbolic Regression and Ensemble ML" — ICMERE 2025. 1st Author.
[C7] "Multi-Paradigm Predictive Modeling of Surface Roughness in Friction Drilling of A356 Aluminum Using RSM–ML Integration" — ICMERE 2025. 1st Author.
[C8] "Initialization-Free Non-Linear Constrained Optimization Using a Bayesian Self-Supervised MLP" — IEOM Bangladesh 2025. DOI: 10.46254/BA08.20250052. Co-author.

=== POSTER ===
"Enhancing Fertility Rate Projections in South Korea Using Multivariate LSTM" — CUET IEEE Poster Competition 2025. DOI: 10.13140/RG.2.2.35739.99362. 1st Author.

=== ONGOING RESEARCH ===
"A Multi-Stakeholder Game-Theoretic Framework for Strategic Modeling of Digital Advertising Ecosystem" — Supervisors: Dr. Jun Zhuang (University at Buffalo), Dr. Puneet Agarwal (Cal Poly), Dr. Ridwan Al Aziz (BUET). Tripartite evolutionary game with replicator dynamics and sensitivity analysis.

=== TECHNICAL SKILLS ===
Programming: Python, MATLAB, SQL
ML/Analytics: XGBoost, LightGBM, Random Forest, AdaBoost, ARIMA, K-Means, SHAP, PDP, Scikit-learn, Pandas, NumPy
Statistics: Panel regression, PLS/OLS, bootstrapping, hypothesis testing, RSM, symbolic regression
CAD/Engineering: SolidWorks, AutoCAD, ANSYS, CATIA
Tools: Power BI, MS Excel, Jupyter, Google Colab

=== CERTIFICATIONS ===
IBM: Python for Data Science AI & Development (Coursera), Statistics for Data Science with Python (Coursera), Databases & SQL for Data Science with Python — WITH HONORS (Coursera), Project Management Fundamentals (SkillsBuild)
Microsoft: Work Smarter with Microsoft Excel (Coursera)
Accenture: Data Analytics & Visualization Job Simulation (Forage)
Vanderbilt University: Introduction to Programming with MATLAB

=== AWARDS & LEADERSHIP ===
1st Place, Undergraduate Research Competition — IEOM International Conference 2025
Class Representative, IPE, Class of 120, BUET (Jan 2023 – Jan 2024)
Batch Representative, Batch of 1300, BUET (Mar 2025 – Present)
International competitions: Astronomy & Astrophysics (prefinal 2024), Youth Math Challenge (prefinal 2022), Bangladesh Economics Summit (1st round 2025), Hult Prize (BUET campus 2025)

=== ACADEMIC PROFILES ===
LinkedIn: https://www.linkedin.com/in/fahimul-haque-a649a0351/
ResearchGate: https://www.researchgate.net/profile/Fahimul-Haque-4
Google Scholar: https://scholar.google.com/citations?user=UPX0wf0AAAAJ&hl=en (4+ citations)
ORCID: https://orcid.org/0009-0001-5003-7720

=== SEEKING ===
PhD positions in industrial engineering systems, operations research, interpretable ML, and socio-technical decision modeling. Also open to research collaborations and data science roles.`;
// HuggingFace Inference API — free tier
// Model: Mistral-7B-Instruct-v0.3 (strong instruction-following, great for Q&A)

// ─── GROQ API CONFIGURATION ─────────────────────────────────────────────────
// Groq is 100% FREE — sign up at https://console.groq.com
// Free tier: 14,400 requests/day | 30 req/min — perfect for a portfolio
// Add your key in Vercel: Settings → Environment Variables → GROQ_API_KEY
const GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions";
const GROQ_MODEL   = "llama-3.1-8b-instant"; // Fast, free, great instruction-following

export default async function handler(req, res) {

  // ── Handle CORS preflight ──────────────────────────────────────────────────
  if (req.method === 'OPTIONS') {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    return res.status(200).end();
  }

  // ── Only allow POST ────────────────────────────────────────────────────────
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  const { messages } = req.body;

  if (!messages || !Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: 'Invalid request body' });
  }

  // Limit conversation history to last 10 messages to stay within token limits
  const recentMessages = messages.slice(-10);

  // Validate message format
  const validMessages = recentMessages.filter(
    m => m && typeof m.role === 'string' && typeof m.content === 'string'
  );

  if (validMessages.length === 0) {
    return res.status(400).json({ error: 'No valid messages provided' });
  }

  // ── Check API key ──────────────────────────────────────────────────────────
  if (!process.env.GROQ_API_KEY) {
    console.error('GROQ_API_KEY environment variable is not set');
    return res.status(500).json({
      error: 'Chatbot is not configured. Please contact hassansiddique632@gmail.com'
    });
  }

  try {
    const response = await fetch(GROQ_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
      },
      body: JSON.stringify({
        model: GROQ_MODEL,
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          ...validMessages,
        ],
        max_tokens: 500,
        temperature: 0.65,
        top_p: 0.9,
        stream: false,
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error('Groq API error:', response.status, errText);

      if (response.status === 401) {
        return res.status(500).json({ error: 'Invalid API key. Please contact hassansiddique632@gmail.com' });
      }
      if (response.status === 429) {
        return res.status(429).json({ error: 'Rate limit reached. Please try again in a moment.' });
      }

      return res.status(502).json({ error: 'AI service temporarily unavailable. Please try again.' });
    }

    const data = await response.json();

    const text =
      data?.choices?.[0]?.message?.content?.trim() ||
      "I'm sorry, I couldn't generate a response right now. Please email hassansiddique632@gmail.com directly.";

    return res.status(200).json({ response: text });

  } catch (error) {
    console.error('Handler error:', error);
    return res.status(500).json({
      error: 'Something went wrong. Please email hassansiddique632@gmail.com directly.'
    });
  }
}

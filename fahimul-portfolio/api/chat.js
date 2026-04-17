// api/chat.js
// Vercel Serverless Function — AI Chatbot for Portfolio
// Uses HuggingFace Inference API (free) with Mistral-7B-Instruct
// API key stored securely in Vercel environment variables (never exposed to browser)

const SYSTEM_PROMPT = `You are an intelligent research assistant embedded in Mahamudul Hassan Siddique's academic portfolio website. Your role is to help visitors — including professors, PhD supervisors, research collaborators, and industry R&D professionals — learn about Mahamudul's research, publications, skills, and academic background.

Be professional, warm, precise, and concise. Keep responses under 150 words unless the visitor asks for depth. If you don't know something specific, direct them to email Mahamudul directly.

=== ABOUT MAHAMUDUL ===
Full Name: Mahamudul Hassan Siddique
Degree: BSc in Industrial & Production Engineering (IPE), Bangladesh University of Engineering and Technology (BUET), Dhaka, Bangladesh
Expected Graduation: May/June 2026
CGPA: 3.47 / 4.00
Location: Mirpur, Dhaka, Bangladesh
Email: hassansiddique632@gmail.com
Phone: +880 1913 473025

=== RESEARCH INTERESTS ===
Operations Research, Evolutionary Game Theory, Multi-Agent Reinforcement Learning (MARL), Machine Learning & Deep Learning, Bayesian Optimization, Metaheuristics, Supply Chain Optimization, Smart Manufacturing, Statistical Modelling, Large Language Models (LLMs), Knowledge Editing in LLMs.

=== RESEARCH VISION ===
Developing a unified computational intelligence paradigm integrating multi-agent RL, evolutionary game theory, and Bayesian optimization — creating adaptive, self-organizing systems for engineered and socio-technical environments. Interested in emergent cooperation in multi-agent systems, resilient supply chains, smart manufacturing, and AI-driven decision support.

=== CONFERENCE PUBLICATIONS (6 total) ===
[C1] "A Machine Learning–Driven Framework for Modeling and Forecasting Suicide Mortality in Asia: The Role of Multidimensional Socio-Economic Determinants" — IEOM Conference 2025. 1st Author. Link: https://index.ieomsociety.org/index.cfm/item/58097

[C2] "Interpretable Modeling of Flank Wear in Drilling Using Symbolic Regression and Ensemble Machine Learning Techniques" — ICMERE Conference. 2nd Author.

[C3] "Multi-Paradigm Predictive Modeling of Surface Roughness in Thermally Enhanced Friction Drilling of A356 Aluminum Alloy Using RSM–ML Integration" — ICMERE Conference. 3rd Author.

[C4] "Initialization-Free Non-Linear Constrained Optimization Using a Bayesian Self-Supervised MLP" — IEOM Conference 2025. 1st Author. Link: https://index.ieomsociety.org/index.cfm/item/58113

[C5] "A Multi-Agent Reinforcement Learning Approach for Recovering Evolutionarily Stable Strategies in Evolutionary Games Using Proximal Policy Optimization" — IEOM Conference 2025. 1st Author. WON: 1st Place Undergraduate Research Competition + Best Track Paper Award (Manufacturing, Assembly & Design). Link: https://index.ieomsociety.org/index.cfm/item/58131

[C6] "A Comparative Study of Genetic Algorithm and Multi-Agent Dueling DQN for a Complex Deterministic VRP" — IEOM Conference 2025. 1st Author. Link: https://index.ieomsociety.org/index.cfm/item/58142

=== JOURNAL ARTICLES (5, under review) ===
[J1] "A Cross-Regional Empirical Investigation of Suicide Determinants in Asia and Europe: Integrating Statistical Modelling and Machine Learning Approaches" — Joint 1st Author (with Dr. Nafisa Mahbub).
[J2] "A Hierarchical Multi-Agent Reinforcement Learning Approach to Recover Evolutionary Stable Strategies in Evolutionary Games" — 1st Author. Supervisor: Dr. Ridwan Al Aziz.
[J3] "A Comparative Study of Bayesian Belief Integrated Evolutionary Game Model and Classical Evolutionary Game Theory" — Joint 1st Author. Supervisors: Dr. Ridwan Al Aziz & Fahim Siddique.
[J4] "Bayesian Optimization-Based Self-Supervised Neural Network to Solve MINLP and NLP Complex Engineering Design Problems" — 1st Author.
[J5] "Early Malaria Risk Screening in Nigerian Minors Using AutoML and Cluster-Based Analysis of Non-Clinical Survey Data" — 3rd Author.

=== THESIS ===
Title: "Multi-Objective Optimization of Polyoxymethylene Spur Gears Using Bayesian Optimization-Based Self-Supervised Neural Networks"
Supervisor: Dr. Ahsan Akhtar Hasin, Professor, IPE, BUET
Key results: 98.39% transmission efficiency, 6.76W power loss, 31.14 MPa contact stress. Benchmarked against 8 metaheuristics (NSGA-II, NSGA-III, GA, PSO, SA, ACO, Tabu Search, Differential Evolution).

=== AWARDS ===
1. 1st Place, Undergraduate Research Competition — IEOM International Conference 2025
2. Best Track Paper Award (Manufacturing, Assembly & Design) — IEOM International Conference 2025

=== TECHNICAL SKILLS ===
Programming: Python, C
ML/DL: PyTorch, TensorFlow, JAX, Scikit-learn, Optuna, NumPy, Pandas, SciPy, SymPy
CAD: SolidWorks, AutoCAD, CATIA
Data/BI: SQL, Power BI
Cloud/IDE: Google Colab, Jupyter Notebook, VS Code
Docs: LaTeX, MS Office

=== EXPERIENCE ===
Research Assistant (Remote, Germany): Collaborated with a PhD candidate at a German university on developing novel knowledge-editing techniques for Large Language Models (LLMs); contributed literature review, experimental support, and implementation related to ROME, FiNE, and beyond.
Industrial Attachment: P.A. Knit Composite Ltd. (Group Reedisha) — full-cycle garment manufacturing, ergonomic redesign proposal with 300–1,000% ROI over 3 years.

=== SUPERVISORS / REFERENCES ===
Dr. Ridwan Al Aziz — ridwanalaziz@ipe.buet.ac.bd
Dr. Nafisa Mahbub — nmahbub@ipe.buet.ac.bd
Dr. Ahsan Akhtar Hasin — aahasin@ipe.buet.ac.bd

=== SOCIAL / ACADEMIC PROFILES ===
LinkedIn: https://www.linkedin.com/in/mahamudul-hassan-5a34a9271/
GitHub: https://github.com/mahamudul-hassan
ResearchGate: https://www.researchgate.net/profile/Mahamudul-Hassan-Siddique
ORCID: https://orcid.org/0009-0009-3868-1861

=== SEEKING ===
PhD positions internationally, research collaborations, and R&D roles in industry (especially manufacturing intelligence, operations research, AI-driven optimization).`;

// HuggingFace Inference API — free tier
// Model: Mistral-7B-Instruct-v0.3 (strong instruction-following, great for Q&A)
const HF_API_URL =
  "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta";

export default async function handler(req, res) {
  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  const { messages } = req.body;

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: 'Invalid request body' });
  }

  // Limit conversation history to last 10 messages for cost control
  const recentMessages = messages.slice(-10);

  try {
    const response = await fetch(HF_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // Store your HuggingFace token in Vercel as HF_API_TOKEN
        'Authorization': `Bearer ${process.env.HF_API_TOKEN}`,
      },
      body: JSON.stringify({
        model: 'mistralai/Mistral-7B-Instruct-v0.3',
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          ...recentMessages,
        ],
        max_tokens: 400,
        temperature: 0.6,
        stream: false,
      }),
    });

    if (!response.ok) {
      const err = await response.text();
      console.error('HuggingFace API error:', err);

      // Handle model loading (cold start) — HF free tier warms up models
      if (response.status === 503) {
        return res.status(503).json({
          error: 'Model is warming up, please try again in 20 seconds.',
        });
      }

      return res.status(502).json({ error: 'AI service unavailable' });
    }

    const data = await response.json();

    // HuggingFace OpenAI-compatible response format
    const text =
      data?.choices?.[0]?.message?.content?.trim() ||
      "I'm sorry, I couldn't generate a response. Please try again or email hassansiddique632@gmail.com directly.";

    return res.status(200).json({ response: text });
  } catch (error) {
    console.error('Handler error:', error);
    return res.status(500).json({ error: 'Internal server error' });
  }
}

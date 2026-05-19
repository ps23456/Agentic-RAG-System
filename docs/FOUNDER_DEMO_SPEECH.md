# Founder Demo Speech — Agentic Insurance Document Intelligence
**Duration:** ~3 minutes | **Tone:** Professional, confident, product-led  
**Live URL:** https://isr.aventhic.com

---

## DELIVERY NOTES (read before you present)

- Speak at a calm, steady pace (~130–150 words per minute).
- Pause after each section so the founder can see the screen.
- Have these tabs ready: **Chat**, **Documents**, **Medical**, and **Slack** with RAGBOT.
- Total spoken words: ~420 (fits 3 minutes with short pauses).

---

## SECTION 1 — OPENING (~20 seconds)

> Good [morning/afternoon]. I want to show you something we’ve already built and deployed—not a slide deck, but a working product.
>
> We’ve built an **Agentic RAG system for insurance and medical documents**: policies, claims, and physician forms like APS reports. It’s live on our VPS at **isr.aventhic.com**, and I’ll walk you through the frontend, the intelligence layer, and how we’ve extended the same engine to Slack.

---

## SECTION 2 — DEPLOYED FRONTEND & CORE CHAT (~50 seconds)

**[DEMO: Open https://isr.aventhic.com — Chat view]**

> This is our production frontend: **React 19, TypeScript, Vite, and Tailwind**. It talks to a **FastAPI** backend over a secure API, served behind **Nginx** with HTTPS.
>
> The user experience is simple: upload documents, ask a question in natural language, and get an **answer with citations**—file name and page number—so every claim is traceable.
>
> Under the hood, retrieval is not “one search box.” We run **hybrid search**: keyword **BM25** plus **semantic vectors** stored in **ChromaDB**, fused with reciprocal rank fusion. For visual content—diagrams, form layouts, handwritten checkboxes—we use **CLIP** image embeddings in a separate collection and fuse text and image results.
>
> For scanned PDFs and complex forms, we use **Mistral OCR** so checkbox values and handwriting actually land in the index—not just template text.
>
> On top of retrieval, we run **agentic RAG**: a **Groq**-powered planner decides how to search, which patient or file to scope, and how to combine **page-level tree indexes**—a PageIndex-style hierarchy over PDFs—with chunk search. The model then streams a grounded summary.
>
> **[DEMO: Ask a live question, e.g. “How many hours can Teresa Brown stand?” — click a citation to open the document viewer]**
>
> Notice the **document viewer** on the right: one click from the answer to the exact page. That’s explainability—critical for insurance workflows.

---

## SECTION 3 — MEDICAL ANALYSIS MODULE (~35 seconds)

**[DEMO: Sidebar → Medical]**

> The second capability is **focused medical analysis**—designed for a quick, high-value demo in about thirty seconds.
>
> Here, adjusters or clinical reviewers select a **patient**, upload or pick **imaging and reports**—PDFs and images—and ask a targeted question: compare two studies, summarize a diagnosis section, or extract restrictions.
>
> The backend sends the selected pages to our **vision-capable LLM**—Groq with multimodal support—along with patient context from prior uploads. You get a structured **analysis report** in one shot, without running a full corpus search.
>
> **[DEMO: Select patient → one image → short query → Generate]**
>
> This is complementary to chat: chat is for **search across the whole library**; medical mode is for **deep read of specific clinical artifacts**.

---

## SECTION 4 — AUTO-INDEXING PIPELINE (~35 seconds)

**[DEMO: Sidebar → Upload, or Documents page]**

> Documents don’t sit idle after upload. **Indexing is automatic.**
>
> When a user uploads PDFs or images, the API saves them under our data store and **enqueues a background job** in a durable SQLite-backed queue. A worker process runs **document chunking**, **Mistral OCR where needed**, **text and image embedding**, and **page-tree construction**—without blocking the UI.
>
> The frontend polls **index status**; when the job finishes, those files are immediately searchable in chat. We also support **targeted re-index**—for example, one APS form after we fix OCR—via the API without rebuilding the entire corpus.
>
> That design matters for production: uploads, retries, and multi-tenant isolation are already modeled—not bolted on later.

---

## SECTION 5 — API SERVICE & SLACK CHATBOT (~40 seconds)

**[DEMO: Switch to Slack — @RAGBOT mention]**

> Everything you’ve seen in the UI is also a **productized API**.
>
> We wrapped the backend with authenticated REST endpoints—chat, streaming chat, query, upload, index jobs, documents, medical analysis—and deployed it in **Docker** on the VPS with persistent storage for vectors and tenant metadata.
>
> On top of that API, I built a **Slack integration**: a Node.js bot using **Slack Bolt** and Socket Mode. Teams can **@mention the bot**, ask the same insurance questions, and get cited answers in the channel where they already work. They can upload files to Slack and trigger the same auto-index path.
>
> **[DEMO: @RAGBOT “How many hours can Teresa Brown stand?” — show reply with citations]**
>
> Same brain, three surfaces: **web app**, **REST API**, and **Slack**—so we can sell to enterprises that live in chat tools, not only in a custom portal.

---

## CLOSING (~15 seconds)

> In summary: we have a **deployed, multi-modal, agentic document intelligence platform** for insurance and medical workflows—with explainable retrieval, automatic indexing, a dedicated medical analysis mode, and Slack-ready distribution.
>
> The foundation is in place; the next phase is performance tuning, tenant onboarding, and customer pilots. I’m happy to go deeper on architecture or a pilot timeline.

---

## QUICK REFERENCE — TECH STACK (if the founder asks)

| Layer | Technology |
|--------|------------|
| Frontend | React 19, Vite, TypeScript, Tailwind CSS |
| API | FastAPI, Uvicorn, Python 3.12 |
| Deployment | Docker, Nginx, Hostinger VPS, Let’s Encrypt |
| Text retrieval | BM25 + sentence-transformers embeddings (MiniLM), ChromaDB |
| Reranking | BGE CrossEncoder (optional) |
| OCR | Mistral OCR API |
| Images | OpenAI CLIP (ViT-L/14), Chroma image collection |
| LLM | Groq (primary), OpenAI fallback |
| Trees | Custom PageIndex-style PDF hierarchy |
| Jobs / tenants | SQLite tenant registry, background index worker |
| Slack | Node.js, @slack/bolt, Socket Mode |

---

## TIMING CHEAT SHEET

| Section | Target |
|---------|--------|
| Opening | 0:00 – 0:20 |
| Frontend & chat | 0:20 – 1:10 |
| Medical analysis | 1:10 – 1:45 |
| Auto-index | 1:45 – 2:20 |
| API & Slack | 2:20 – 3:00 |
| Close | 3:00 – 3:15 |

---

*Prepared for founder demo. Update live examples (patient name, file names) to match your indexed corpus on the VPS.*

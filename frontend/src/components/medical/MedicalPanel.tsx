import { useState, useEffect } from "react";
import { ArrowLeft, Loader2, Upload, Stethoscope } from "lucide-react";
import ReactMarkdown from "react-markdown";

interface Props {
  onBack: () => void;
}

export function MedicalPanel({ onBack }: Props) {
  const [patients, setPatients] = useState<string[]>([]);
  const [selectedPatient, setSelectedPatient] = useState("");
  const [query, setQuery] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [report, setReport] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/medical/patients")
      .then((r) => r.json())
      .then((d) => setPatients(d.patients || []))
      .catch(() => {});
  }, []);

  const handleAnalyze = async () => {
    if (!query.trim()) return;
    setAnalyzing(true);
    setError(null);
    setReport(null);
    try {
      const res = await fetch("/api/medical/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query.trim(),
          image_paths: [],
          patient: selectedPatient,
        }),
      });
      const data = await res.json();
      if (data.error) setError(data.error);
      else setReport(data.report);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-[var(--bg-primary)]">
      <div className="flex items-center gap-3 px-6 py-4 border-b border-[var(--border)]">
        <button
          onClick={onBack}
          className="p-1.5 rounded-lg hover:bg-[var(--bg-tertiary)] transition-colors"
        >
          <ArrowLeft size={18} className="text-[var(--text-secondary)]" />
        </button>
        <Stethoscope size={20} className="text-[var(--accent)]" />
        <h1 className="text-lg font-semibold text-[var(--text-primary)]">
          Medical Analysis Hub
        </h1>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-2xl mx-auto space-y-6">
          <div>
            <label className="text-sm font-medium text-[var(--text-primary)] block mb-2">
              Select Patient
            </label>
            <select
              value={selectedPatient}
              onChange={(e) => setSelectedPatient(e.target.value)}
              className="w-full bg-[var(--bg-primary)] border border-[var(--border)] rounded-xl px-4 py-2.5 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--accent)] focus:ring-2 focus:ring-[var(--accent-light)] transition-all"
            >
              <option value="">All patients</option>
              {patients.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-sm font-medium text-[var(--text-primary)] block mb-2">
              Upload Medical Documents
            </label>
            <div
              className="border-2 border-dashed border-[var(--border)] rounded-2xl p-8 text-center cursor-pointer hover:border-[var(--accent)] hover:bg-[var(--accent-bg)] transition-all"
              onClick={() => {
                const input = document.createElement("input");
                input.type = "file";
                input.multiple = true;
                input.accept = ".pdf,.png,.jpg,.jpeg,.tiff,.bmp";
                input.onchange = async () => {
                  if (!input.files) return;
                  const form = new FormData();
                  form.append("patient", selectedPatient || "unknown");
                  form.append("doc_type", "Medical Report");
                  Array.from(input.files).forEach((f) => form.append("files", f));
                  await fetch("/api/medical/upload", { method: "POST", body: form });
                };
                input.click();
              }}
            >
              <Upload
                size={28}
                className="mx-auto text-[var(--text-muted)] mb-2"
              />
              <p className="text-sm font-medium text-[var(--text-secondary)]">
                Drop medical documents here
              </p>
              <p className="text-xs text-[var(--text-muted)] mt-1">
                PDF, Images, TIFF
              </p>
            </div>
          </div>

          <div>
            <label className="text-sm font-medium text-[var(--text-primary)] block mb-2">
              Clinical Query
            </label>
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., Analyze the patient's physical capacities..."
              rows={3}
              className="w-full bg-[var(--bg-primary)] border border-[var(--border)] rounded-xl px-4 py-3 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--accent)] focus:ring-2 focus:ring-[var(--accent-light)] resize-none transition-all"
            />
          </div>

          <button
            onClick={handleAnalyze}
            disabled={analyzing || !query.trim()}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-semibold disabled:opacity-50 transition-all shadow-sm"
          >
            {analyzing ? (
              <>
                <Loader2 size={16} className="animate-spin" /> Analyzing...
              </>
            ) : (
              "Analyze"
            )}
          </button>

          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-sm text-red-600">
              {error}
            </div>
          )}

          {report && (
            <div className="prose prose-sm max-w-none bg-[var(--bg-secondary)] rounded-2xl p-6 border border-[var(--border-light)]">
              <ReactMarkdown>{report}</ReactMarkdown>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

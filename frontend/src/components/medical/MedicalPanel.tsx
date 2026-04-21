import { useState, useEffect } from "react";
import {
  ArrowLeft,
  Loader2,
  Stethoscope,
  X,
  User,
  Info,
  ChevronDown,
  CloudUpload,
  FileText,
  ImageIcon,
  Sparkles,
  ClipboardList,
} from "lucide-react";
import ReactMarkdown from "react-markdown";

interface FileItem {
  path: string;
  name: string;
  page?: number;
}

interface Props {
  onBack: () => void;
}

export function MedicalPanel({ onBack }: Props) {
  const [patients, setPatients] = useState<string[]>([]);
  const [selectedPatient, setSelectedPatient] = useState("");
  const [files, setFiles] = useState<FileItem[]>([]);
  const [uploading, setUploading] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  const [imageA, setImageA] = useState<string>("");
  const [imageB, setImageB] = useState<string>("");
  const [query, setQuery] = useState("");
  const [analyzing, setAnalyzing] = useState(false);
  const [report, setReport] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [patientMenuOpen, setPatientMenuOpen] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(true);

  useEffect(() => {
    fetch("/api/medical/patients")
      .then((r) => r.json())
      .then((d) => setPatients(d.patients || []))
      .catch(() => {});
  }, []);

  // Load existing files when patient changes
  useEffect(() => {
    if (!selectedPatient.trim()) {
      setFiles([]);
      setImageA("");
      setImageB("");
      return;
    }
    fetch(`/api/medical/files?patient=${encodeURIComponent(selectedPatient)}`)
      .then((r) => r.json())
      .then((d) => {
        const fileList = d.files || (d.paths || []).map((p: string) => ({
          path: p,
          name: p.split("/").pop() || p,
          page: 1,
        }));
        setFiles(fileList);
        setImageA("");
        setImageB("");
      })
      .catch(() => setFiles([]));
  }, [selectedPatient]);

  const handleAnalyze = async () => {
    if (!query.trim()) return;

    const pathsToAnalyze: string[] = [];
    const pagesToAnalyze: number[] = [];

    if (compareMode) {
      if (!imageA || !imageB || imageA === imageB) {
        setError("Select two different images for comparison.");
        return;
      }
      pathsToAnalyze.push(imageA, imageB);
      const fa = files.find((f) => f.path === imageA);
      const fb = files.find((f) => f.path === imageB);
      pagesToAnalyze.push(fa?.page ?? 1, fb?.page ?? 1);
    } else {
      if (!imageA) {
        setError("Select an image to analyze.");
        return;
      }
      pathsToAnalyze.push(imageA);
      const fa = files.find((f) => f.path === imageA);
      pagesToAnalyze.push(fa?.page ?? 1);
    }

    setAnalyzing(true);
    setError(null);
    setReport(null);
    try {
      const res = await fetch("/api/medical/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query.trim(),
          image_paths: pathsToAnalyze,
          pages: pagesToAnalyze,
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

  const handleUpload = async (fileList: FileList) => {
    if (!fileList?.length) return;
    setUploading(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("patient", selectedPatient || "unknown");
      form.append("doc_type", "Medical Report");
      Array.from(fileList).forEach((f) => form.append("files", f));
      const res = await fetch("/api/medical/upload", { method: "POST", body: form });
      const data = await res.json();
      if (data.saved && data.saved.length > 0) {
        if (!selectedPatient) {
          setSelectedPatient("unknown");
        }
        const newFiles: FileItem[] = data.saved.map((p: string) => ({
          path: p,
          name: p.split("/").pop() || p,
          page: 1,
        }));
        setFiles((prev) => [...prev, ...newFiles]);
        fetch("/api/medical/patients")
          .then((r) => r.json())
          .then((d) => setPatients(d.patients || []))
          .catch(() => {});
      } else if (data.error) {
        setError(data.error);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const removeFile = (path: string) => {
    setFiles((prev) => prev.filter((f) => f.path !== path));
    if (imageA === path) setImageA("");
    if (imageB === path) setImageB("");
  };

  const defaultQuery = compareMode
    ? "Compare these two images and describe changes."
    : "Describe this image and compare to history if available.";
  const selectedAName = files.find((f) => f.path === imageA)?.name;
  const selectedBName = files.find((f) => f.path === imageB)?.name;
  const canAnalyze = !!query.trim() && !!imageA && (!compareMode || !!imageB);
  const quickPrompts = [
    "Describe findings",
    "Compare with history",
    "Flag abnormalities",
    "Generate summary",
  ];

  const getInitials = (name: string) => {
    const words = name.trim().split(/\s+/).filter(Boolean);
    if (!words.length) return "U";
    return words.slice(0, 2).map((w) => w[0]?.toUpperCase() || "").join("");
  };

  const patientEntries = patients.map((name, idx) => ({
    name,
    id: `PT-${String(idx + 1).padStart(3, "0")}`,
    initials: getInitials(name),
  }));
  const selectedPatientEntry = patientEntries.find((p) => p.name === selectedPatient);

  const steps = [
    "Select Patient",
    "Upload Files",
    "Analyze",
    "View Report",
  ];

  const activeStep = !selectedPatient
    ? 1
    : files.length === 0
    ? 2
    : !report
    ? 3
    : 4;

  return (
    <div className="flex flex-col h-full bg-[var(--bg-primary)] font-sans">
      <div className="sticky top-0 z-20 border-b border-[var(--border)] bg-[var(--bg-primary)]/95 backdrop-blur supports-[backdrop-filter]:bg-[var(--bg-primary)]/85">
        <div className="px-6 py-4">
          <div className="flex items-center gap-3">
            <button
              onClick={onBack}
              className="p-2 rounded-xl hover:bg-[var(--bg-tertiary)] transition-colors duration-200"
            >
              <ArrowLeft size={18} className="text-[var(--text-secondary)]" />
            </button>
            <div className="w-10 h-10 rounded-xl bg-blue-100 dark:bg-blue-900/35 flex items-center justify-center">
              <Stethoscope size={18} className="text-[#2563EB]" />
            </div>
            <div>
              <h1 className="text-[20px] font-semibold tracking-tight text-[var(--text-primary)]">
                Medical Analysis Hub
              </h1>
              <p className="text-xs text-[var(--text-muted)]">
                Upload, compare, and generate structured clinical reports with AI assistance.
              </p>
            </div>
          </div>

          <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
            {steps.map((step, idx) => {
              const n = idx + 1;
              const reached = n <= activeStep;
              return (
                <div
                  key={step}
                  className={`flex items-center gap-2 rounded-xl border px-3 py-2 text-xs transition-all duration-200 ${
                    reached
                      ? "border-blue-200 bg-blue-50 text-blue-700 dark:border-blue-800 dark:bg-blue-950/30 dark:text-blue-300"
                      : "border-[var(--border)] bg-[var(--bg-secondary)] text-[var(--text-muted)]"
                  }`}
                >
                  <span className="font-semibold">{`0${n}`.slice(-2)}</span>
                  <span className="truncate">{step}</span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-7xl mx-auto grid gap-6 lg:grid-cols-[390px,1fr]">
          <div className="space-y-6">
            <section className="rounded-xl border border-[var(--border-light)] bg-[var(--bg-secondary)]/40 shadow-sm p-6">
              <div className="flex items-center justify-between mb-2">
                <label className="text-[12px] uppercase tracking-wide font-semibold text-[var(--text-muted)]">
                  Step 1 · Select Patient
                </label>
                <div className="relative group">
                  <Info size={14} className="text-[#2563EB]" />
                  <div className="pointer-events-none absolute right-0 top-5 w-52 rounded-lg border border-blue-200 bg-white p-2 text-[11px] text-blue-700 opacity-0 shadow-md transition-opacity duration-200 group-hover:opacity-100 dark:border-blue-800 dark:bg-slate-900 dark:text-blue-300">
                    Pick a patient first so uploads and analysis stay organized.
                  </div>
                </div>
              </div>

              <div className="relative">
                {showOnboarding && !selectedPatient && (
                  <div className="absolute -top-3 left-2 z-10 rounded-full bg-blue-600 px-2.5 py-1 text-[11px] font-medium text-white shadow">
                    Start here: Select a patient
                  </div>
                )}
                <button
                  type="button"
                  onClick={() => {
                    setPatientMenuOpen((v) => !v);
                    setShowOnboarding(false);
                  }}
                  className="w-full flex items-center justify-between rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] px-3.5 py-3 text-left transition-all duration-200 hover:border-blue-300 focus:outline-none focus:ring-2 focus:ring-blue-200"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="w-8 h-8 rounded-full bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 flex items-center justify-center text-xs font-semibold">
                      {selectedPatientEntry ? selectedPatientEntry.initials : <User size={14} />}
                    </div>
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-[var(--text-primary)] truncate">
                        {selectedPatientEntry ? selectedPatientEntry.name : "Choose a patient"}
                      </p>
                      <p className="text-xs text-[var(--text-muted)] truncate">
                        {selectedPatientEntry ? selectedPatientEntry.id : "No patient selected"}
                      </p>
                    </div>
                  </div>
                  <ChevronDown size={16} className={`text-[var(--text-muted)] transition-transform duration-200 ${patientMenuOpen ? "rotate-180" : ""}`} />
                </button>

                {patientMenuOpen && (
                  <div className="mt-2 rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] shadow-lg overflow-hidden">
                    <div className="max-h-56 overflow-auto">
                      {patientEntries.map((p) => (
                        <button
                          key={p.name}
                          type="button"
                          onClick={() => {
                            setSelectedPatient(p.name);
                            setPatientMenuOpen(false);
                          }}
                          className="w-full flex items-center gap-3 px-3.5 py-2.5 text-left hover:bg-blue-50 dark:hover:bg-blue-950/20 transition-colors duration-200"
                        >
                          <div className="w-7 h-7 rounded-full bg-indigo-100 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300 text-xs font-semibold flex items-center justify-center">
                            {p.initials}
                          </div>
                          <div className="min-w-0">
                            <p className="text-sm text-[var(--text-primary)] truncate">{p.name}</p>
                            <p className="text-xs text-[var(--text-muted)]">{p.id}</p>
                          </div>
                        </button>
                      ))}
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedPatient("unknown");
                        setPatientMenuOpen(false);
                      }}
                      className="w-full border-t border-[var(--border)] px-3.5 py-2.5 text-left text-sm font-medium text-[#2563EB] hover:bg-blue-50 dark:hover:bg-blue-950/20 transition-colors duration-200"
                    >
                      + Add New Patient
                    </button>
                  </div>
                )}
              </div>
            </section>

            <section className="rounded-xl border border-[var(--border-light)] bg-[var(--bg-secondary)]/40 shadow-sm p-6">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-[12px] uppercase tracking-wide font-semibold text-[var(--text-muted)]">
                  Step 2 · Upload Medical Files
                </h2>
              </div>
              <div
                className="relative min-h-[180px] border-2 border-dashed border-blue-400 rounded-xl p-6 text-center cursor-pointer bg-blue-50/20 hover:bg-blue-50/40 hover:border-blue-500 transition-all duration-200"
                onDragOver={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  const dropped = e.dataTransfer?.files;
                  if (dropped?.length) handleUpload(dropped);
                }}
                onClick={() => {
                  const input = document.createElement("input");
                  input.type = "file";
                  input.multiple = true;
                  input.accept = ".pdf,.png,.jpg,.jpeg,.tiff,.bmp";
                  input.onchange = () => {
                    if (input.files) handleUpload(input.files);
                  };
                  input.click();
                }}
              >
                <span className="absolute right-3 top-3 rounded-full bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 px-2.5 py-1 text-[11px] font-medium">
                  {files.length} files loaded
                </span>
                {uploading ? (
                  <Loader2 size={36} className="mx-auto text-[#2563EB] mb-3 animate-spin" />
                ) : (
                  <CloudUpload size={36} className="mx-auto text-[#2563EB] mb-3" />
                )}
                <p className="text-[14px] font-semibold text-[var(--text-primary)]">Drop files here</p>
                <p className="text-[12px] text-[var(--text-muted)] mt-1">PDF, PNG, JPG, TIFF supported</p>
              </div>

              {files.length > 0 && (
                <div className="mt-3">
                  <div className="flex gap-3 overflow-x-auto pb-1">
                    {files.map((f) => {
                      const isImage = /\.(png|jpg|jpeg|bmp|tiff|tif)$/i.test(f.name);
                      return (
                        <div
                          key={f.path}
                          className="relative shrink-0 w-24 rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] overflow-hidden"
                        >
                          <button
                            type="button"
                            onClick={() => removeFile(f.path)}
                            className="absolute right-1 top-1 z-10 p-0.5 rounded bg-white/90 dark:bg-black/60 text-[var(--text-muted)] hover:text-red-600"
                            title="Remove"
                          >
                            <X size={12} />
                          </button>
                          <div className="h-16 bg-[var(--bg-tertiary)] flex items-center justify-center">
                            {isImage ? (
                              <img
                                src={`/api/medical/image?path=${encodeURIComponent(f.path)}`}
                                alt={f.name}
                                className="w-full h-full object-cover"
                              />
                            ) : (
                              <FileText size={20} className="text-[var(--text-muted)]" />
                            )}
                          </div>
                          <p className="px-2 py-1 text-[11px] text-[var(--text-muted)] truncate">{f.name}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </section>

            <section className="rounded-xl border border-[var(--border-light)] bg-[var(--bg-secondary)]/40 shadow-sm p-6">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-[14px] font-semibold text-[var(--text-primary)]">Compare Mode</p>
                  <p className="text-[12px] text-[var(--text-muted)]">
                    Side-by-side Image A vs Image B analysis
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setCompareMode((v) => {
                      const next = !v;
                      if (!next) setImageB("");
                      return next;
                    });
                  }}
                  className={`relative w-11 h-6 rounded-full transition-colors duration-200 ${
                    compareMode ? "bg-indigo-600" : "bg-slate-300 dark:bg-slate-700"
                  }`}
                  aria-label="Toggle compare mode"
                >
                  <span
                    className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform duration-200 ${
                      compareMode ? "translate-x-5" : "translate-x-0.5"
                    }`}
                  />
                </button>
              </div>

              <div className={`mt-3 grid gap-3 transition-all duration-200 ${compareMode ? "md:grid-cols-2" : "grid-cols-1"}`}>
                <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] p-3">
                  <label className="text-[12px] font-medium text-[var(--text-muted)] block mb-1.5">Image A</label>
                  <select
                    value={imageA}
                    onChange={(e) => setImageA(e.target.value)}
                    className="w-full bg-[var(--bg-primary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] outline-none focus:border-[#2563EB]"
                  >
                    <option value="">Choose file</option>
                    {files.map((f) => (
                      <option key={f.path} value={f.path}>
                        {f.name}
                      </option>
                    ))}
                  </select>
                </div>
                {compareMode && (
                  <div className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] p-3">
                    <label className="text-[12px] font-medium text-[var(--text-muted)] block mb-1.5">Image B</label>
                    <select
                      value={imageB}
                      onChange={(e) => setImageB(e.target.value)}
                      className="w-full bg-[var(--bg-primary)] border border-[var(--border)] rounded-lg px-3 py-2 text-sm text-[var(--text-primary)] outline-none focus:border-[#2563EB]"
                    >
                      <option value="">Choose file</option>
                      {files.filter((f) => f.path !== imageA).map((f) => (
                        <option key={f.path} value={f.path}>
                          {f.name}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>
            </section>

            <section className="rounded-xl border border-[var(--border-light)] bg-[var(--bg-secondary)]/40 shadow-sm p-6">
              <label className="text-[12px] uppercase tracking-wide font-semibold text-[var(--text-muted)] block mb-2">
                Step 3 · Clinical Query
              </label>
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder={defaultQuery}
                rows={4}
                className="w-full bg-[var(--bg-primary)] border border-[var(--border)] rounded-xl px-4 py-3 text-[14px] text-[var(--text-primary)] outline-none focus:border-[#2563EB] focus:ring-2 focus:ring-blue-200 resize-none transition-all duration-200"
              />
              <div className="flex items-center justify-between mt-1">
                <span className="text-[12px] text-[var(--text-muted)]">Quick prompts</span>
                <span className="text-[12px] text-[var(--text-muted)]">{query.length} chars</span>
              </div>
              <div className="mt-2 flex flex-wrap gap-2">
                {quickPrompts.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    onClick={() => setQuery(prompt)}
                    className="px-2.5 py-1.5 rounded-full border border-blue-200 bg-blue-50 text-[#2563EB] text-[12px] font-medium hover:bg-blue-100 dark:border-blue-800 dark:bg-blue-950/20 dark:text-blue-300 transition-colors duration-200"
                  >
                    {prompt}
                  </button>
                ))}
              </div>

              <button
                onClick={handleAnalyze}
                disabled={analyzing || !canAnalyze}
                title={!files.length ? "Upload files to continue" : ""}
                className="mt-4 w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-gradient-to-r from-[#2563EB] to-[#7C3AED] text-white text-sm font-semibold disabled:opacity-50 disabled:cursor-not-allowed shadow-sm hover:shadow-md transition-all duration-200"
              >
                {analyzing ? (
                  <>
                    <Loader2 size={16} className="animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Sparkles size={16} />
                    Analyze & Compare
                  </>
                )}
              </button>
            </section>

            {error && (
              <div className="p-4 rounded-xl border border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950/20 text-sm text-red-600 dark:text-red-400">
                {error}
              </div>
            )}
          </div>

          <div className="space-y-6">
            <section className="rounded-xl border border-[var(--border-light)] bg-[var(--bg-secondary)]/40 shadow-sm p-6">
              <div className="flex items-center gap-2 mb-3">
                <ImageIcon size={16} className="text-[#2563EB]" />
                <h2 className="text-[14px] font-semibold text-[var(--text-primary)]">Step 4 · Selected Images</h2>
              </div>
              {imageA || imageB ? (
                <div className={`grid gap-4 ${imageA && imageB ? "md:grid-cols-2" : "grid-cols-1"}`}>
                  {imageA && (
                    <div className="rounded-xl border border-[var(--border)] overflow-hidden bg-[var(--bg-primary)]">
                      <div className="px-3 py-2 border-b border-[var(--border)] text-xs font-medium text-[var(--text-muted)]">Image A</div>
                      <div className="aspect-[4/3] bg-[var(--bg-tertiary)] flex items-center justify-center">
                        <img
                          src={`/api/medical/image?path=${encodeURIComponent(imageA)}`}
                          alt="Image A"
                          className="max-w-full max-h-full object-contain"
                        />
                      </div>
                      <p className="px-3 py-2 text-[12px] text-[var(--text-muted)] truncate">{selectedAName}</p>
                    </div>
                  )}
                  {imageB && (
                    <div className="rounded-xl border border-[var(--border)] overflow-hidden bg-[var(--bg-primary)]">
                      <div className="px-3 py-2 border-b border-[var(--border)] text-xs font-medium text-[var(--text-muted)]">Image B</div>
                      <div className="aspect-[4/3] bg-[var(--bg-tertiary)] flex items-center justify-center">
                        <img
                          src={`/api/medical/image?path=${encodeURIComponent(imageB)}`}
                          alt="Image B"
                          className="max-w-full max-h-full object-contain"
                        />
                      </div>
                      <p className="px-3 py-2 text-[12px] text-[var(--text-muted)] truncate">{selectedBName}</p>
                    </div>
                  )}
                </div>
              ) : (
                <div className="rounded-xl border border-dashed border-[var(--border)] bg-[var(--bg-primary)] p-8 text-center">
                  <ImageIcon size={28} className="mx-auto text-[var(--text-muted)] mb-2" />
                  <p className="text-sm font-medium text-[var(--text-secondary)]">No images selected yet</p>
                </div>
              )}
            </section>

            <section className="rounded-xl border border-[var(--border-light)] bg-[var(--bg-secondary)]/40 shadow-sm p-6">
              <div className="flex items-center gap-2 mb-3">
                <ClipboardList size={16} className="text-[#7C3AED]" />
                <h2 className="text-[14px] font-semibold text-[var(--text-primary)]">Clinical Analysis</h2>
              </div>
              {report ? (
                <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-primary)] p-4 prose prose-sm max-w-none dark:prose-invert prose-headings:mb-2 prose-p:my-2 prose-li:my-0.5">
                  <ReactMarkdown>{report}</ReactMarkdown>
                </div>
              ) : (
                <div className="rounded-xl border border-dashed border-[var(--border)] bg-[var(--bg-primary)] p-8 text-center">
                  <FileText size={28} className="mx-auto text-[var(--text-muted)] mb-2" />
                  <p className="text-sm font-medium text-[var(--text-secondary)]">
                    Your structured report will appear here
                  </p>
                </div>
              )}
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}

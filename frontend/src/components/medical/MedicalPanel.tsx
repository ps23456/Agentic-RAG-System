import { useState, useEffect } from "react";
import { ArrowLeft, Loader2, Upload, Stethoscope, X, ImagePlus } from "lucide-react";
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
            {!selectedPatient && (
              <p className="text-xs text-amber-600 dark:text-amber-400 mb-2">
                Select a patient first to upload and organize files.
              </p>
            )}
            <div
              className="border-2 border-dashed border-[var(--border)] rounded-2xl p-6 text-center cursor-pointer hover:border-[var(--accent)] hover:bg-[var(--accent-bg)] transition-all"
              onDragOver={(e) => {
                e.preventDefault();
                e.stopPropagation();
              }}
              onDrop={(e) => {
                e.preventDefault();
                e.stopPropagation();
                const files = e.dataTransfer?.files;
                if (files?.length) handleUpload(files);
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
              {uploading ? (
                <Loader2 size={28} className="mx-auto text-[var(--accent)] mb-2 animate-spin" />
              ) : (
                <Upload size={28} className="mx-auto text-[var(--text-muted)] mb-2" />
              )}
              <p className="text-sm font-medium text-[var(--text-secondary)]">
                Drop files or click to upload
              </p>
              <p className="text-xs text-[var(--text-muted)] mt-1">PDF, Images, TIFF</p>
            </div>

            {files.length > 0 && (
              <div className="mt-3 space-y-2">
                <p className="text-xs font-medium text-[var(--text-muted)]">
                  Files ({files.length}) — add or remove:
                </p>
                <div className="flex flex-wrap gap-2">
                  {files.map((f) => (
                    <div
                      key={f.path}
                      className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)] text-sm"
                    >
                      <span className="text-[var(--text-primary)] truncate max-w-[180px]">
                        {f.name}
                      </span>
                      <button
                        type="button"
                        onClick={() => removeFile(f.path)}
                        className="p-0.5 rounded hover:bg-red-100 dark:hover:bg-red-900/30 text-[var(--text-muted)] hover:text-red-600"
                        title="Remove"
                      >
                        <X size={14} />
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Compare mode */}
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="compare-mode"
              checked={compareMode}
              onChange={(e) => {
                setCompareMode(e.target.checked);
                if (!e.target.checked) setImageB("");
              }}
              className="rounded border-[var(--border)] text-[var(--accent)] focus:ring-[var(--accent)]"
            />
            <label htmlFor="compare-mode" className="text-sm font-medium text-[var(--text-primary)]">
              Compare two images side by side (Image A vs Image B)
            </label>
          </div>

          {/* Image selection */}
          {files.length > 0 && (
            <div className="space-y-4">
              <div>
                <label className="text-sm font-medium text-[var(--text-primary)] block mb-2">
                  {compareMode ? "Image A (Primary)" : "Select Image to Analyze"}
                </label>
                <select
                  value={imageA}
                  onChange={(e) => setImageA(e.target.value)}
                  className="w-full bg-[var(--bg-primary)] border border-[var(--border)] rounded-xl px-4 py-2.5 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--accent)]"
                >
                  <option value="">— Choose file —</option>
                  {files.map((f) => (
                    <option key={f.path} value={f.path}>
                      {f.name}
                    </option>
                  ))}
                </select>
              </div>

              {compareMode && (
                <div>
                  <label className="text-sm font-medium text-[var(--text-primary)] block mb-2">
                    Image B (Compare with)
                  </label>
                  <select
                    value={imageB}
                    onChange={(e) => setImageB(e.target.value)}
                    className="w-full bg-[var(--bg-primary)] border border-[var(--border)] rounded-xl px-4 py-2.5 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--accent)]"
                  >
                    <option value="">— Choose file —</option>
                    {files
                      .filter((f) => f.path !== imageA)
                      .map((f) => (
                        <option key={f.path} value={f.path}>
                          {f.name}
                        </option>
                      ))}
                  </select>
                </div>
              )}
            </div>
          )}

          {/* Image previews - side by side */}
          {(imageA || imageB) && (
            <div>
              <label className="text-sm font-medium text-[var(--text-primary)] block mb-2">
                Selected Images
              </label>
              <div className={`grid gap-4 ${imageA && imageB ? "grid-cols-2" : "grid-cols-1"}`}>
                {imageA && (
                  <div className="rounded-xl border border-[var(--border)] overflow-hidden bg-[var(--bg-secondary)]">
                    <div className="px-3 py-2 border-b border-[var(--border)] text-xs font-medium text-[var(--text-muted)]">
                      Image A
                    </div>
                    <div className="aspect-[4/3] flex items-center justify-center p-2 bg-[var(--bg-tertiary)]">
                      <img
                        src={`/api/medical/image?path=${encodeURIComponent(imageA)}`}
                        alt="Image A"
                        className="max-w-full max-h-full object-contain rounded-lg"
                      />
                    </div>
                    <p className="px-3 py-1.5 text-xs text-[var(--text-muted)] truncate">
                      {files.find((f) => f.path === imageA)?.name}
                    </p>
                  </div>
                )}
                {imageB && (
                  <div className="rounded-xl border border-[var(--border)] overflow-hidden bg-[var(--bg-secondary)]">
                    <div className="px-3 py-2 border-b border-[var(--border)] text-xs font-medium text-[var(--text-muted)]">
                      Image B
                    </div>
                    <div className="aspect-[4/3] flex items-center justify-center p-2 bg-[var(--bg-tertiary)]">
                      <img
                        src={`/api/medical/image?path=${encodeURIComponent(imageB)}`}
                        alt="Image B"
                        className="max-w-full max-h-full object-contain rounded-lg"
                      />
                    </div>
                    <p className="px-3 py-1.5 text-xs text-[var(--text-muted)] truncate">
                      {files.find((f) => f.path === imageB)?.name}
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          <div>
            <label className="text-sm font-medium text-[var(--text-primary)] block mb-2">
              Clinical Query
            </label>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
              placeholder={defaultQuery}
            rows={3}
              className="w-full bg-[var(--bg-primary)] border border-[var(--border)] rounded-xl px-4 py-3 text-sm text-[var(--text-primary)] outline-none focus:border-[var(--accent)] focus:ring-2 focus:ring-[var(--accent-light)] resize-none transition-all"
          />
        </div>

          {files.length === 0 && (
            <p className="text-xs text-amber-600 dark:text-amber-400">
              Select a patient and upload medical documents above before analyzing.
            </p>
          )}

        <button
          onClick={handleAnalyze}
            disabled={analyzing || !query.trim() || (!imageA && files.length > 0)}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-[var(--accent)] hover:bg-[var(--accent-hover)] text-white text-sm font-semibold disabled:opacity-50 transition-all shadow-sm"
          >
            {analyzing ? (
              <>
                <Loader2 size={16} className="animate-spin" /> Analyzing...
              </>
            ) : (
              <>
                <ImagePlus size={16} /> Analyze & Compare
              </>
            )}
        </button>

        {error && (
            <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl text-sm text-red-600 dark:text-red-400">
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

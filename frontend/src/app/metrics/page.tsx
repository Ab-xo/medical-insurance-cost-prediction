"use client";

import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ChartCard } from "@/components/ChartCard";
import type { ModalImage } from "@/components/ImageModal";
import {
  Trophy,
  Clock,
  CheckCircle2,
  BarChart2,
  TrendingUp,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

// ─── Types ────────────────────────────────────────────────────────────────────

type Metric = {
  model: string;
  rmse: number;
  mae: number;
  r2: number;
  train_time_sec: number;
  is_best: boolean;
};

type CVResult = {
  r2_mean: number;
  r2_std: number;
  rmse_mean: number;
};

type EvalFigures = {
  residuals:      string | null;
  pred_vs_actual: string | null;
  feature_imp:    string | null;
  coefficients:   string | null;
  learning_curve: string | null;
};

type ModelStem = { stem: string; label: string };

// ─── Helpers ─────────────────────────────────────────────────────────────────

const TABS = [
  { label: "Metrics Table", icon: <Trophy    size={14} /> },
  { label: "Charts",        icon: <BarChart2 size={14} /> },
  { label: "Per-Model",     icon: <TrendingUp size={14} /> },
  { label: "CV Results",    icon: <CheckCircle2 size={14} /> },
];

const EVAL_BASE = "http://localhost:8000/outputs/figures/evaluation";

/** Build a ModalImage list from an EvalFigures object (skipping nulls). */
function evalImages(figs: EvalFigures): ModalImage[] {
  return [
    figs.residuals      && { url: figs.residuals,      label: "Residuals" },
    figs.pred_vs_actual && { url: figs.pred_vs_actual, label: "Predicted vs Actual" },
    figs.feature_imp    && { url: figs.feature_imp,    label: "Feature Importance" },
    figs.coefficients   && { url: figs.coefficients,   label: "Coefficients" },
    figs.learning_curve && { url: figs.learning_curve, label: "Learning Curve" },
  ].filter(Boolean) as ModalImage[];
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function MetricsPage() {
  const [metrics,        setMetrics]        = useState<Metric[]>([]);
  const [cvData,         setCvData]         = useState<Record<string, CVResult>>({});
  const [models,         setModels]         = useState<ModelStem[]>([]);
  const [evalFigs,       setEvalFigs]       = useState<EvalFigures | null>(null);
  const [selectedModel,  setSelectedModel]  = useState("");
  const [activeTab,      setActiveTab]      = useState(0);
  const [loading,        setLoading]        = useState(true);

  useEffect(() => {
    Promise.all([
      fetch("http://localhost:8000/api/metrics").then((r) => r.json()),
      fetch("http://localhost:8000/api/cv").then((r) => r.json()),
      fetch("http://localhost:8000/api/eval-models").then((r) => r.json()),
    ])
      .then(([m, cv, ms]: [Metric[], Record<string, CVResult>, ModelStem[]]) => {
        setMetrics(m);
        setCvData(cv);
        setModels(ms);
        if (ms.length) setSelectedModel(ms[0].stem);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedModel) return;
    fetch(`http://localhost:8000/api/eval-figures/${selectedModel}`)
      .then((r) => r.json())
      .then(setEvalFigs)
      .catch(() => setEvalFigs(null));
  }, [selectedModel]);

  // ── loading / empty ──
  if (loading)
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );

  if (!metrics.length)
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] text-center space-y-4">
        <BarChart2 className="h-12 w-12 text-muted-foreground opacity-40" />
        <h2 className="text-xl font-bold">No metrics found</h2>
        <p className="text-muted-foreground">Run the training pipeline first.</p>
        <code className="bg-muted px-3 py-2 rounded text-xs font-mono">
          python -m src.train
        </code>
      </div>
    );

  // Build sibling arrays for modal navigation in "Charts" tab
  const summaryImages: ModalImage[] = [
    { url: `${EVAL_BASE}/model_comparison_rmse.png`, label: "RMSE Comparison" },
    { url: `${EVAL_BASE}/model_comparison_r2.png`,   label: "R² Comparison" },
    { url: `${EVAL_BASE}/model_leaderboard.png`,     label: "Multi-Metric Leaderboard" },
  ];

  const perModelImages = evalFigs ? evalImages(evalFigs) : [];

  return (
    <div className="space-y-8 max-w-7xl">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-1">Model Comparison</h1>
        <p className="text-muted-foreground">
          {metrics.length} regression models evaluated on the 20 % hold-out test set.
          Use <BarChart2 size={13} className="inline mb-0.5" /> icons on any chart to minimize or maximize.
        </p>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 bg-secondary/30 p-1 rounded-xl w-fit flex-wrap">
        {TABS.map(({ label, icon }, i) => (
          <button
            key={label}
            onClick={() => setActiveTab(i)}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 whitespace-nowrap ${
              activeTab === i
                ? "bg-card shadow-sm text-foreground"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            {icon} {label}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">

        {/* ── Tab 0: Metrics Table ── */}
        {activeTab === 0 && (
          <motion.div
            key="table"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            <Card className="bg-card/40 backdrop-blur-md border-primary/20 shadow-lg overflow-hidden">
              <CardHeader className="bg-secondary/20 pb-4 border-b border-border/50">
                <CardTitle className="flex items-center gap-2">
                  <Trophy className="text-yellow-400" size={18} /> Leaderboard
                </CardTitle>
                <CardDescription>
                  Sorted by RMSE — lower RMSE / MAE and higher R² = better model.
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border/50 bg-muted/30">
                      <th className="text-left px-4 py-3 font-semibold w-10">#</th>
                      <th className="text-left px-4 py-3 font-semibold">Model</th>
                      <th className="text-right px-4 py-3 font-semibold">RMSE ↓</th>
                      <th className="text-right px-4 py-3 font-semibold">MAE ↓</th>
                      <th className="text-right px-4 py-3 font-semibold">R² ↑</th>
                      <th className="text-right px-4 py-3 font-semibold hidden sm:table-cell">
                        <Clock size={13} className="inline mr-1" />Train
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.map((m, i) => (
                      <tr
                        key={m.model}
                        className={`transition-colors ${
                          m.is_best
                            ? "bg-primary/[0.07] hover:bg-primary/[0.11]"
                            : "hover:bg-secondary/20"
                        }`}
                      >
                        <td className="px-4 py-3 text-muted-foreground font-mono">{i + 1}</td>
                        <td className="px-4 py-3 font-semibold">
                          <span className="flex items-center gap-2 flex-wrap">
                            {m.model}
                            {m.is_best && (
                              <Badge className="text-[9px] px-1.5 bg-primary text-primary-foreground">
                                <CheckCircle2 size={10} className="mr-0.5" /> BEST
                              </Badge>
                            )}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-xs">
                          ${m.rmse.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-xs">
                          ${m.mae.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-xs text-primary font-semibold">
                          {m.r2.toFixed(4)}
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-xs text-muted-foreground hidden sm:table-cell">
                          {m.train_time_sec.toFixed(3)}s
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* ── Tab 1: Summary Charts ── */}
        {activeTab === 1 && (
          <motion.div
            key="charts"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="space-y-6"
          >
            <div className="grid md:grid-cols-2 gap-5">
              <ChartCard
                label="RMSE Comparison (lower is better)"
                url={summaryImages[0].url}
                icon={<BarChart2 size={12} />}
                siblings={summaryImages}
                siblingIndex={0}
              />
              <ChartCard
                label="R² Comparison (higher is better)"
                url={summaryImages[1].url}
                icon={<BarChart2 size={12} />}
                siblings={summaryImages}
                siblingIndex={1}
              />
            </div>

            <ChartCard
              label="Normalised Multi-Metric Leaderboard"
              url={summaryImages[2].url}
              icon={<TrendingUp size={12} />}
              siblings={summaryImages}
              siblingIndex={2}
            />
          </motion.div>
        )}

        {/* ── Tab 2: Per-Model Plots ── */}
        {activeTab === 2 && (
          <motion.div
            key="per-model"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="space-y-6"
          >
            {/* Model selector pills */}
            <div className="flex flex-wrap gap-2">
              {models.map((m) => (
                <button
                  key={m.stem}
                  onClick={() => setSelectedModel(m.stem)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-all ${
                    selectedModel === m.stem
                      ? "bg-primary text-primary-foreground border-primary shadow-sm"
                      : "bg-card/40 text-muted-foreground border-border/50 hover:border-primary/40 hover:text-foreground"
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>

            {evalFigs && perModelImages.length > 0 && (
              <>
                {/* Residuals + Pred vs Actual */}
                <div className="grid md:grid-cols-2 gap-5">
                  {[
                    { url: evalFigs.residuals,      label: "Residuals" },
                    { url: evalFigs.pred_vs_actual, label: "Predicted vs Actual" },
                  ]
                    .filter((x) => x.url)
                    .map(({ url, label }) => {
                      const idx = perModelImages.findIndex((p) => p.label === label);
                      return (
                        <ChartCard
                          key={label}
                          label={label}
                          url={url!}
                          icon={<BarChart2 size={12} />}
                          siblings={perModelImages}
                          siblingIndex={idx >= 0 ? idx : 0}
                        />
                      );
                    })}
                </div>

                {/* Feature importance / Coefficients */}
                <div className="grid md:grid-cols-2 gap-5">
                  {[
                    { url: evalFigs.feature_imp,  label: "Feature Importance" },
                    { url: evalFigs.coefficients, label: "Coefficients" },
                  ]
                    .filter((x) => x.url)
                    .map(({ url, label }) => {
                      const idx = perModelImages.findIndex((p) => p.label === label);
                      return (
                        <ChartCard
                          key={label}
                          label={label}
                          url={url!}
                          icon={<BarChart2 size={12} />}
                          siblings={perModelImages}
                          siblingIndex={idx >= 0 ? idx : 0}
                        />
                      );
                    })}
                </div>

                {/* Learning curve (full width) */}
                {evalFigs.learning_curve && (() => {
                  const idx = perModelImages.findIndex((p) => p.label === "Learning Curve");
                  return (
                    <ChartCard
                      label="Learning Curve"
                      url={evalFigs.learning_curve!}
                      icon={<TrendingUp size={12} />}
                      siblings={perModelImages}
                      siblingIndex={idx >= 0 ? idx : 0}
                    />
                  );
                })()}
              </>
            )}

            {evalFigs && perModelImages.length === 0 && (
              <p className="text-muted-foreground text-sm text-center py-8">
                No evaluation plots found for this model.
              </p>
            )}
          </motion.div>
        )}

        {/* ── Tab 3: CV Results ── */}
        {activeTab === 3 && (
          <motion.div
            key="cv"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
          >
            <Card className="bg-card/40 backdrop-blur-md border-primary/20 overflow-hidden">
              <CardHeader className="bg-secondary/20 pb-4 border-b border-border/50">
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp size={18} className="text-primary" />
                  5-Fold Cross-Validation
                </CardTitle>
                <CardDescription>
                  Generalisation performance across 5 folds (mean ± std).
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0 overflow-x-auto">
                {Object.keys(cvData).length ? (
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border/50 bg-muted/30">
                        <th className="text-left px-4 py-3 font-semibold">Model</th>
                        <th className="text-right px-4 py-3 font-semibold">CV R² (mean)</th>
                        <th className="text-right px-4 py-3 font-semibold">CV R² (±std)</th>
                        <th className="text-right px-4 py-3 font-semibold">CV RMSE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(cvData)
                        .sort(([, a], [, b]) => b.r2_mean - a.r2_mean)
                        .map(([model, cv]) => (
                          <tr key={model} className="hover:bg-secondary/20 transition-colors">
                            <td className="px-4 py-3 font-medium">{model}</td>
                            <td className="px-4 py-3 text-right font-mono text-xs text-primary">
                              {cv.r2_mean.toFixed(4)}
                            </td>
                            <td className="px-4 py-3 text-right font-mono text-xs text-muted-foreground">
                              ±{cv.r2_std.toFixed(4)}
                            </td>
                            <td className="px-4 py-3 text-right font-mono text-xs">
                              ${cv.rmse_mean.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                ) : (
                  <p className="text-muted-foreground text-sm p-6">
                    CV data not found. Run the training pipeline first.
                  </p>
                )}
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

"use client";

import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ChartCard } from "@/components/ChartCard";
import { ImageModal, type ModalImage } from "@/components/ImageModal";
import { motion, AnimatePresence } from "framer-motion";
import {
  ImageIcon, ChevronLeft, ChevronRight, Maximize2,
  BarChart2, TrendingUp, PieChart, Grid3x3,
} from "lucide-react";

// ── Category definitions (stems must match API response labels) ──────────────
const CATEGORIES: {
  label: string;
  icon: React.ReactNode;
  stems: string[];
}[] = [
  {
    label: "Distributions",
    icon: <BarChart2 size={14} />,
    stems: [
      "Charges Distribution",
      "Age Distribution",
      "BMI Distribution",
      "Sex Distribution",
      "Children vs Charges",
    ],
  },
  {
    label: "Relationships",
    icon: <TrendingUp size={14} />,
    stems: [
      "Age vs Charges",
      "BMI vs Charges",
      "Charges by Smoker Status",
    ],
  },
  {
    label: "Categorical",
    icon: <PieChart size={14} />,
    stems: [
      "Charges by Region",
      "Charges by BMI Category",
    ],
  },
  {
    label: "Advanced",
    icon: <Grid3x3 size={14} />,
    stems: ["Correlation Heatmap", "Pairplot"],
  },
];

const ALL_LABEL = "All";

export default function EDAPage() {
  const [figures,  setFigures]  = useState<ModalImage[]>([]);
  const [selected, setSelected] = useState(0);
  const [featured, setFeatured] = useState(false);
  const [activeTab, setActiveTab] = useState(ALL_LABEL);
  const [loading,  setLoading]  = useState(true);

  useEffect(() => {
    fetch("http://localhost:8000/api/eda-figures")
      .then((r) => r.json())
      .then((d: any[]) => {
        setFigures(d.map((f) => ({ url: f.url, label: f.label })));
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  if (loading)
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] text-muted-foreground gap-4">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
        <p>Loading EDA visualisations…</p>
      </div>
    );

  if (!figures.length)
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] text-muted-foreground gap-3">
        <ImageIcon size={40} className="opacity-30" />
        <p>No EDA figures found. Run the training pipeline first.</p>
        <code className="bg-muted px-3 py-2 rounded text-xs font-mono">
          python -m src.train
        </code>
      </div>
    );

  const fig = figures[selected];

  // Which figures are visible in the current tab
  const visibleFigures =
    activeTab === ALL_LABEL
      ? figures
      : figures.filter((f) => {
          const cat = CATEGORIES.find((c) => c.label === activeTab);
          return cat ? cat.stems.includes(f.label) : true;
        });

  const tabs = [
    { label: ALL_LABEL, icon: <ImageIcon size={14} /> },
    ...CATEGORIES.map((c) => ({ label: c.label, icon: c.icon })),
  ];

  return (
    <div className="space-y-7 max-w-7xl">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-1">
          Exploratory Data Analysis
        </h1>
        <p className="text-muted-foreground">
          {figures.length} visualisations generated during the ML training
          pipeline.{" "}
          <span className="inline-flex items-center gap-1 text-xs text-muted-foreground/70">
            Click any chart or <Maximize2 size={11} className="inline" /> to
            open full-screen.
          </span>
        </p>
      </div>

      {/* ── Featured viewer ─────────────────────────────────────────────── */}
      <Card className="bg-card/40 backdrop-blur-md border-primary/20 overflow-hidden">
        <CardHeader className="flex flex-row items-center justify-between py-2.5 px-3 border-b border-border/50 bg-secondary/10">
          <div className="flex items-center gap-2 min-w-0">
            <ImageIcon size={15} className="text-primary shrink-0" />
            <CardTitle className="text-sm font-medium truncate">{fig.label}</CardTitle>
          </div>

          <div className="flex items-center gap-1 shrink-0 ml-2">
            <span className="text-xs text-muted-foreground tabular-nums mr-1">
              {selected + 1} / {figures.length}
            </span>
            <button
              onClick={() => setSelected((s) => (s - 1 + figures.length) % figures.length)}
              className="p-1.5 rounded-md hover:bg-secondary/60 text-muted-foreground hover:text-foreground transition-colors"
              title="Previous (←)"
            >
              <ChevronLeft size={15} />
            </button>
            <button
              onClick={() => setSelected((s) => (s + 1) % figures.length)}
              className="p-1.5 rounded-md hover:bg-secondary/60 text-muted-foreground hover:text-foreground transition-colors"
              title="Next (→)"
            >
              <ChevronRight size={15} />
            </button>
            <button
              onClick={() => setFeatured(true)}
              className="p-1.5 rounded-md hover:bg-secondary/60 text-muted-foreground hover:text-foreground transition-colors"
              title="Maximize"
            >
              <Maximize2 size={14} />
            </button>
          </div>
        </CardHeader>

        <CardContent className="p-0">
          <AnimatePresence mode="wait">
            <motion.div
              key={fig.url}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.18 }}
              className="flex items-center justify-center bg-white/[0.02] p-4 cursor-zoom-in"
              style={{ minHeight: 440 }}
              onClick={() => setFeatured(true)}
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={fig.url}
                alt={fig.label}
                className="max-h-[440px] max-w-full object-contain"
              />
            </motion.div>
          </AnimatePresence>
        </CardContent>
      </Card>

      {/* ── Thumbnail strip ─────────────────────────────────────────────── */}
      <div className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin">
        {figures.map((f, i) => (
          <button
            key={f.url}
            onClick={() => setSelected(i)}
            title={f.label}
            className={`shrink-0 w-[90px] h-[60px] rounded-lg overflow-hidden border-2 transition-all duration-150 ${
              i === selected
                ? "border-primary shadow-md shadow-primary/30 opacity-100 scale-105"
                : "border-border/40 hover:border-primary/40 opacity-50 hover:opacity-90"
            }`}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={f.url} alt={f.label} className="w-full h-full object-cover" />
          </button>
        ))}
      </div>

      {/* ── Category tabs + grid ────────────────────────────────────────── */}
      <div className="space-y-4">
        {/* Tab bar */}
        <div className="flex gap-1 bg-secondary/30 p-1 rounded-xl w-fit flex-wrap">
          {tabs.map(({ label, icon }) => (
            <button
              key={label}
              onClick={() => setActiveTab(label)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all duration-150 whitespace-nowrap ${
                activeTab === label
                  ? "bg-card shadow-sm text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {icon}
              {label}
              <span className="text-[10px] text-muted-foreground/60 ml-0.5">
                {label === ALL_LABEL
                  ? figures.length
                  : (figures.filter((f) => {
                      const cat = CATEGORIES.find((c) => c.label === label);
                      return cat ? cat.stems.includes(f.label) : false;
                    }).length)}
              </span>
            </button>
          ))}
        </div>

        {/* Grid — no duplicate, one clean view */}
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.18 }}
          >
            {visibleFigures.length === 0 ? (
              <div className="flex items-center justify-center h-32 text-muted-foreground text-sm">
                No figures in this category yet.
              </div>
            ) : (
              <div
                className={`grid gap-5 ${
                  /* Heatmap and Pairplot need wider cells */
                  visibleFigures.some((f) =>
                    f.label === "Correlation Heatmap" || f.label === "Pairplot"
                  )
                    ? "grid-cols-1 md:grid-cols-2"
                    : "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3"
                }`}
              >
                {visibleFigures.map((f) => {
                  const globalIdx = figures.indexOf(f);
                  const isLarge =
                    f.label === "Correlation Heatmap" || f.label === "Pairplot";
                  return (
                    <div
                      key={f.url}
                      className={isLarge ? "md:col-span-2" : ""}
                    >
                      <ChartCard
                        label={f.label}
                        url={f.url}
                        icon={<ImageIcon size={12} />}
                        siblings={figures}
                        siblingIndex={globalIdx}
                        className={
                          globalIdx === selected
                            ? "border-primary shadow-md"
                            : "border-border/40"
                        }
                      />
                    </div>
                  );
                })}
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Featured full-screen modal */}
      <ImageModal
        open={featured}
        image={fig}
        siblings={figures}
        siblingIndex={selected}
        onClose={() => setFeatured(false)}
        onMinimize={() => setFeatured(false)}
        onNavigate={(i) => setSelected(i)}
      />
    </div>
  );
}

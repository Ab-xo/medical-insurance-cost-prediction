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
import {
  BarChart2,
  Activity,
  Users,
  DollarSign,
  TrendingUp,
  AlertTriangle,
  Trophy,
  CheckCircle2,
  Hash,
  Tag,
} from "lucide-react";
import { motion } from "framer-motion";

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.08 } },
};
const item = { hidden: { opacity: 0, y: 18 }, show: { opacity: 1, y: 0 } };

const NUMERIC_FEATURES = [
  { name: "age",      range: "18 – 64",  desc: "Age of primary beneficiary",        unit: "years" },
  { name: "bmi",      range: "15.9 – 53.1", desc: "Body Mass Index",               unit: "kg/m²" },
  { name: "children", range: "0 – 5",    desc: "Number of dependants on policy",    unit: "count" },
];

const CATEGORICAL_FEATURES_RAW = [
  { name: "sex",    values: ["male", "female"],                                          desc: "Biological sex of beneficiary" },
  { name: "smoker", values: ["yes", "no"],                                               desc: "Current smoker status" },
  { name: "region", values: ["northeast", "northwest", "southeast", "southwest"],        desc: "US residential region" },
];

const ENGINEERED_FEATURES = [
  { name: "age_group",    desc: "young / middle_age / senior" },
  { name: "bmi_category", desc: "underweight / normal / overweight / obese" },
  { name: "is_obese",     desc: "1 if BMI ≥ 30" },
  { name: "smoker_obese", desc: "1 if smoker AND obese (interaction)" },
  { name: "age_bmi",      desc: "age × bmi / 1000" },
  { name: "has_children", desc: "1 if dependants > 0" },
  { name: "family_size",  desc: "individual / small_family / large_family" },
  { name: "age_smoker",   desc: "age × smoker flag" },
];

export default function HomePage() {
  const [summary, setSummary] = useState<any>(null);
  const [metrics, setMetrics]  = useState<any[]>([]);
  const [isError, setIsError]  = useState(false);

  useEffect(() => {
    Promise.all([
      fetch("http://localhost:8000/api/dataset/summary"),
      fetch("http://localhost:8000/api/metrics"),
    ])
      .then(async ([rs, rm]) => {
        if (!rs.ok || !rm.ok) throw new Error("Backend unavailable");
        const [s, m] = await Promise.all([rs.json(), rm.json()]);
        setSummary(s);
        setMetrics(m);
      })
      .catch(() => setIsError(true));
  }, []);

  /* ── error state ── */
  if (isError) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] text-center space-y-4">
        <Activity className="h-12 w-12 text-destructive opacity-50" />
        <h2 className="text-xl font-bold">Cannot connect to Backend</h2>
        <p className="text-muted-foreground">
          Start the FastAPI server first.
        </p>
        <code className="bg-muted px-3 py-2 rounded-lg text-sm font-mono">
          uvicorn app.main:app --reload --port 8000
        </code>
      </div>
    );
  }

  /* ── loading state ── */
  if (!summary) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );
  }

  const bestModel = metrics.find((m) => m.is_best);
  const smokerRatio =
    summary.smoker.yes && summary.smoker.no
      ? (summary.smoker.yes.mean / summary.smoker.no.mean).toFixed(1)
      : "3.8";

  return (
    <div className="space-y-10">
      {/* ── Page header ── */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-1">
          🏥 Medical Insurance Cost Prediction
        </h1>
        <p className="text-muted-foreground max-w-2xl">
          End-to-end machine learning pipeline predicting annual insurance
          charges from demographics and health attributes.
        </p>
      </div>

      {/* ── KPI cards ── */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4"
      >
        <motion.div variants={item}>
          <Card className="bg-card/40 backdrop-blur-md border-primary/20 hover:border-primary/50 transition-colors">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Total Policyholders
              </CardTitle>
              <Users className="h-4 w-4 text-primary" />
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">{summary.rows.toLocaleString()}</p>
              <p className="text-xs text-muted-foreground mt-1">
                {summary.missing} missing · {summary.duplicates} duplicate
              </p>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div variants={item}>
          <Card className="bg-card/40 backdrop-blur-md border-primary/20 hover:border-primary/50 transition-colors">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Average Charge
              </CardTitle>
              <DollarSign className="h-4 w-4 text-primary" />
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">
                ${summary.target.mean.toLocaleString()}
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                Median ${summary.target.median.toLocaleString()} · Skew{" "}
                {summary.target.skew}
              </p>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div variants={item}>
          <Card className="bg-card/40 backdrop-blur-md border-destructive/20 hover:border-destructive/50 transition-colors">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Smoker Avg Charge
              </CardTitle>
              <AlertTriangle className="h-4 w-4 text-destructive" />
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">
                ${summary.smoker.yes?.mean.toLocaleString()}
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                {summary.smoker.yes?.count} smokers · {smokerRatio}× non-smoker
              </p>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div variants={item}>
          <Card className="bg-card/40 backdrop-blur-md border-green-500/20 hover:border-green-500/40 transition-colors">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                Non-Smoker Avg
              </CardTitle>
              <BarChart2 className="h-4 w-4 text-green-500" />
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold">
                ${summary.smoker.no?.mean.toLocaleString()}
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                {summary.smoker.no?.count} non-smokers
              </p>
            </CardContent>
          </Card>
        </motion.div>
      </motion.div>

      {/* ── Middle row: Key Insights + Regional ── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35 }}
        className="grid gap-6 md:grid-cols-2"
      >
        <Card className="bg-card/40 backdrop-blur-md">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp size={18} className="text-primary" /> Key Findings
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3 text-sm">
              <li className="flex gap-3 items-start">
                <span className="text-destructive mt-0.5">🚬</span>
                <span>
                  <strong>Smoker status</strong> is the dominant predictor —
                  smokers pay <strong>{smokerRatio}×</strong> more on average.
                </span>
              </li>
              <li className="flex gap-3 items-start">
                <span className="mt-0.5">⚖️</span>
                <span>
                  <strong>BMI ≥ 30 + smoking</strong> creates the
                  highest-charge group via the{" "}
                  <code className="bg-muted px-1 rounded text-xs">smoker_obese</code>{" "}
                  interaction feature.
                </span>
              </li>
              <li className="flex gap-3 items-start">
                <span className="text-blue-400 mt-0.5">📅</span>
                <span>
                  <strong>Age effect is nonlinear</strong> — charges spike
                  sharply after age 50.
                </span>
              </li>
              <li className="flex gap-3 items-start">
                <span className="text-green-400 mt-0.5">✅</span>
                <span>
                  <strong>Linear models outperform</strong> tree ensembles once
                  interaction features are engineered. Lasso is the winner.
                </span>
              </li>
              <li className="flex gap-3 items-start">
                <span className="mt-0.5">📊</span>
                <span>
                  Charges are <strong>right-skewed</strong> (skewness ≈ 1.52)
                  with genuine high-value outliers (
                  {summary.target.outliers} detected).
                </span>
              </li>
            </ul>
          </CardContent>
        </Card>

        <Card className="bg-card/40 backdrop-blur-md">
          <CardHeader>
            <CardTitle>Regional Averages</CardTitle>
            <CardDescription>Average insurance charges by US region</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {summary.regions.map((region: any, i: number) => {
              const max = summary.regions[0].mean;
              const pct = Math.round((region.mean / max) * 100);
              const colors = [
                "bg-primary",
                "bg-blue-500",
                "bg-purple-500",
                "bg-indigo-500",
              ];
              return (
                <div key={region.name}>
                  <div className="flex items-center justify-between text-sm mb-1">
                    <span className="capitalize font-medium">{region.name}</span>
                    <span className="text-muted-foreground">
                      ${region.mean.toLocaleString()}
                    </span>
                  </div>
                  <div className="h-2 rounded-full bg-secondary/50 overflow-hidden">
                    <div
                      className={`h-full rounded-full ${colors[i] ?? "bg-primary"} transition-all duration-700`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </CardContent>
        </Card>
      </motion.div>

      {/* ── Best model banner ── */}
      {bestModel && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <Card className="bg-gradient-to-r from-primary/15 via-card to-card border-primary/30 shadow-lg">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2">
                <Trophy size={20} className="text-yellow-400" /> Best Model:{" "}
                {bestModel.model}
                <Badge className="ml-2 bg-primary/20 text-primary border border-primary/30">
                  <CheckCircle2 size={11} className="mr-1" /> Winner
                </Badge>
              </CardTitle>
              <CardDescription>
                Selected by lowest RMSE on the 20 % hold-out test set
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {[
                  { label: "RMSE", value: `$${bestModel.rmse.toLocaleString(undefined, { maximumFractionDigits: 0 })}` },
                  { label: "MAE",  value: `$${bestModel.mae.toLocaleString(undefined, { maximumFractionDigits: 0 })}` },
                  { label: "R²",   value: bestModel.r2.toFixed(4) },
                  { label: "Train Time", value: `${bestModel.train_time_sec.toFixed(3)}s` },
                ].map(({ label, value }) => (
                  <div key={label} className="bg-secondary/20 rounded-xl p-3 text-center">
                    <p className="text-xs text-muted-foreground mb-1">{label}</p>
                    <p className="text-lg font-bold font-mono">{value}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* ── Model Leaderboard table ── */}
      {metrics.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
        >
          <Card className="bg-card/40 backdrop-blur-md border-primary/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Trophy size={18} className="text-yellow-400" /> Model Leaderboard
              </CardTitle>
              <CardDescription>
                All 10 models sorted by RMSE — lower is better
              </CardDescription>
            </CardHeader>
            <CardContent className="p-0 overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border/50 bg-secondary/20">
                    <th className="text-left px-4 py-3 font-semibold w-8">#</th>
                    <th className="text-left px-4 py-3 font-semibold">Model</th>
                    <th className="text-right px-4 py-3 font-semibold">RMSE</th>
                    <th className="text-right px-4 py-3 font-semibold">MAE</th>
                    <th className="text-right px-4 py-3 font-semibold">R²</th>
                    <th className="text-right px-4 py-3 font-semibold hidden sm:table-cell">
                      Train
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.map((m, i) => (
                    <tr
                      key={m.model}
                      className={
                        m.is_best
                          ? "bg-primary/8 hover:bg-primary/12"
                          : "hover:bg-secondary/20"
                      }
                    >
                      <td className="px-4 py-2.5 text-muted-foreground">{i + 1}</td>
                      <td className="px-4 py-2.5 font-medium">
                        <span className="flex items-center gap-2">
                          {m.model}
                          {m.is_best && (
                            <Badge className="text-[9px] px-1.5 py-0 bg-yellow-500/20 text-yellow-400 border-yellow-500/30">
                              BEST
                            </Badge>
                          )}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-xs">
                        ${m.rmse.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-xs">
                        ${m.mae.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-xs text-primary">
                        {m.r2.toFixed(4)}
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-xs text-muted-foreground hidden sm:table-cell">
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

      {/* ── Dataset Features ── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="grid gap-6 md:grid-cols-2"
      >
        {/* Numeric */}
        <Card className="bg-card/40 backdrop-blur-md border-primary/20">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Hash size={16} className="text-primary" /> Numeric Features
            </CardTitle>
            <CardDescription>Continuous inputs fed into the model</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {NUMERIC_FEATURES.map((f) => (
              <div key={f.name} className="flex items-start justify-between gap-3 py-2 border-b border-border/30 last:border-0">
                <div>
                  <code className="text-xs font-mono text-primary font-semibold">{f.name}</code>
                  <p className="text-xs text-muted-foreground mt-0.5">{f.desc}</p>
                </div>
                <div className="text-right shrink-0">
                  <span className="text-xs font-mono bg-secondary/40 px-2 py-0.5 rounded">{f.range}</span>
                  <p className="text-[10px] text-muted-foreground mt-1">{f.unit}</p>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Categorical */}
        <Card className="bg-card/40 backdrop-blur-md border-primary/20">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Tag size={16} className="text-primary" /> Categorical Features
            </CardTitle>
            <CardDescription>Encoded with OneHotEncoder during training</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {CATEGORICAL_FEATURES_RAW.map((f) => (
              <div key={f.name} className="py-2 border-b border-border/30 last:border-0">
                <div className="flex items-center justify-between mb-1.5">
                  <code className="text-xs font-mono text-primary font-semibold">{f.name}</code>
                  <span className="text-[10px] text-muted-foreground">{f.desc}</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {f.values.map((v) => (
                    <Badge key={v} variant="secondary" className="text-[10px] px-2 py-0 font-mono">
                      {v}
                    </Badge>
                  ))}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </motion.div>

      {/* ── Engineered features ── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
      >
        <Card className="bg-card/40 backdrop-blur-md border-primary/20">
          <CardHeader>
            <CardTitle>🔧 8 Engineered Features</CardTitle>
            <CardDescription>
              Domain-driven features added on top of the 6 raw inputs
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
              {ENGINEERED_FEATURES.map((f) => (
                <div
                  key={f.name}
                  className="bg-secondary/20 rounded-xl p-3 border border-border/50 hover:border-primary/30 transition-colors"
                >
                  <code className="text-xs font-mono text-primary font-semibold block mb-1">
                    {f.name}
                  </code>
                  <p className="text-xs text-muted-foreground">{f.desc}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}

import type { Metadata } from "next";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Target, Lightbulb, Wrench, Database, Layers,
  FolderOpen, ExternalLink, Flame, Scale,
  TrendingUp, CheckCircle, BarChart2, Users2,
} from "lucide-react";

export const metadata: Metadata = {
  title: "About",
  description: "Team, project overview, tech stack and dataset details for MedPredict.ai.",
};

// ── Team ─────────────────────────────────────────────────────────────────────

const TEAM = [
  {
    name: "Abraham",
    initials: "AB",
    role: "ML Engineer",
    contribution: "Model training pipeline, feature engineering, LassoCV/RidgeCV tuning, model evaluation & selection",
    gradient: "from-indigo-500/20 via-primary/10 to-transparent",
    accent: "border-indigo-500/30",
    badge: "bg-indigo-500/20 text-indigo-300",
  },
  {
    name: "Eden",
    initials: "ED",
    role: "Backend & Data Engineer",
    contribution: "FastAPI REST backend, data preprocessing, visualization pipeline, EDA plots",
    gradient: "from-emerald-500/20 via-green-500/10 to-transparent",
    accent: "border-emerald-500/30",
    badge: "bg-emerald-500/20 text-emerald-300",
  },
  {
    name: "Hewan",
    initials: "HW",
    role: "Frontend Developer",
    contribution: "Next.js dashboard, UI components, prediction form, dataset & EDA pages",
    gradient: "from-purple-500/20 via-violet-500/10 to-transparent",
    accent: "border-purple-500/30",
    badge: "bg-purple-500/20 text-purple-300",
  },
];

// ── Supporting data ───────────────────────────────────────────────────────────

const TECH_STACK = [
  { section: "Frontend",        color: "text-blue-400",   items: ["Next.js 16", "React 19", "Tailwind CSS v4", "shadcn/ui", "Framer Motion"] },
  { section: "Backend API",     color: "text-green-400",  items: ["Python 3.13", "FastAPI", "Uvicorn", "Pydantic v2"] },
  { section: "Machine Learning",color: "text-purple-400", items: ["Scikit-Learn", "XGBoost", "Pandas", "NumPy", "Joblib"] },
  { section: "Visualisation",   color: "text-orange-400", items: ["Matplotlib", "Seaborn"] },
];

const FEATURES = [
  { name: "age_group",    desc: "young / middle_age / senior" },
  { name: "bmi_category", desc: "underweight / normal / overweight / obese" },
  { name: "is_obese",     desc: "1 if BMI ≥ 30" },
  { name: "smoker_obese", desc: "1 if smoker AND obese — key interaction" },
  { name: "age_bmi",      desc: "age × bmi / 1000" },
  { name: "has_children", desc: "1 if dependants > 0" },
  { name: "family_size",  desc: "individual / small_family / large_family" },
  { name: "age_smoker",   desc: "age × smoker flag" },
];

const DATASET_COLS = [
  { col: "age",      desc: "Age of the primary beneficiary" },
  { col: "sex",      desc: "male / female" },
  { col: "bmi",      desc: "Body Mass Index (kg/m²)" },
  { col: "children", desc: "Number of dependants covered" },
  { col: "smoker",   desc: "yes / no" },
  { col: "region",   desc: "US residential region (4 values)" },
  { col: "charges",  desc: "Annual insurance cost in USD", isTarget: true },
];

const KEY_FILES = [
  { path: "src/utils.py",               role: "Paths, logger, data loading" },
  { path: "src/preprocessing.py",       role: "Cleaning, validation, stratified split" },
  { path: "src/feature_engineering.py", role: "8 domain-driven derived features" },
  { path: "src/train.py",               role: "Training orchestrator — 10 models" },
  { path: "src/evaluate.py",            role: "Metrics, CV, learning curves" },
  { path: "src/predict.py",             role: "Inference — single & batch" },
  { path: "src/visualization.py",       role: "EDA + evaluation plots" },
  { path: "main.py",                    role: "FastAPI JSON API server" },
  { path: "frontend/src/",             role: "This Next.js dashboard" },
];

const FINDINGS = [
  { icon: <Flame    size={14} className="text-red-400    mt-0.5 shrink-0" />, text: <><strong>Smoker status</strong> is the dominant predictor — smokers pay <strong>3.8×</strong> more on average.</> },
  { icon: <Scale    size={14} className="text-yellow-400 mt-0.5 shrink-0" />, text: <><strong>BMI ≥ 30 + smoking</strong> creates the highest-charge group via the <code className="bg-muted px-1 rounded text-[11px]">smoker_obese</code> interaction.</> },
  { icon: <TrendingUp size={14} className="text-blue-400 mt-0.5 shrink-0" />, text: <><strong>Age effect is nonlinear</strong> — charges spike sharply after 50.</> },
  { icon: <CheckCircle size={14} className="text-green-400 mt-0.5 shrink-0" />, text: <><strong>Linear models beat tree ensembles</strong> once interaction features are engineered. Lasso wins.</> },
  { icon: <BarChart2 size={14} className="text-purple-400 mt-0.5 shrink-0" />, text: <>Charges are <strong>right-skewed</strong> (skewness ≈ 1.52) with real high-value outliers.</> },
];

// ── Helper ────────────────────────────────────────────────────────────────────

function SectionHeader({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="flex items-center gap-2.5">
      <div className="p-1.5 rounded-lg bg-primary/10 text-primary">{icon}</div>
      <span className="font-semibold text-base">{title}</span>
    </div>
  );
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function AboutPage() {
  return (
    <div className="space-y-10 max-w-5xl">

      {/* ── Hero header ── */}
      <div className="relative rounded-2xl overflow-hidden border border-primary/20 bg-gradient-to-br from-primary/15 via-card to-card p-8">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-primary/10 via-transparent to-transparent pointer-events-none" />
        <div className="relative space-y-2">
          <Badge className="bg-primary/20 text-primary border-primary/30 mb-2">
            Machine Learning Project
          </Badge>
          <h1 className="text-3xl font-bold tracking-tight">🏥 MedPredict.ai</h1>
          <p className="text-muted-foreground max-w-2xl leading-relaxed">
            An end-to-end machine learning pipeline that predicts annual US medical insurance
            charges from six policyholder attributes — trained on 1,338 real cases,
            served through a modern Next.js dashboard and FastAPI backend.
          </p>
          <div className="flex flex-wrap gap-2 pt-2">
            {["10 Models Trained", "LassoCV Winner", "RMSE $4,493", "R² 0.886", "52 Tests"].map((t) => (
              <Badge key={t} variant="secondary" className="text-xs">{t}</Badge>
            ))}
          </div>
        </div>
      </div>

      {/* ── Team ── */}
      <div className="space-y-4">
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-lg bg-primary/10 text-primary"><Users2 size={15} /></div>
          <h2 className="font-semibold text-lg">Meet the Team</h2>
        </div>
        <div className="grid sm:grid-cols-3 gap-4">
          {TEAM.map((m) => (
            <div key={m.name}
              className={`relative rounded-2xl border ${m.accent} bg-gradient-to-br ${m.gradient} p-5 space-y-3 overflow-hidden`}>
              <div className="absolute top-0 right-0 w-24 h-24 rounded-full blur-3xl opacity-20 bg-primary pointer-events-none" />
              <div className="flex items-center gap-3">
                <div className="w-12 h-12 rounded-full bg-background/60 border border-white/10 flex items-center justify-center font-bold text-base text-foreground shrink-0 shadow-inner">
                  {m.initials}
                </div>
                <div>
                  <p className="font-bold text-sm">{m.name}</p>
                  <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${m.badge}`}>
                    {m.role}
                  </span>
                </div>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">{m.contribution}</p>
            </div>
          ))}
        </div>
        <p className="text-xs text-muted-foreground text-center pt-1">
          Built as part of a machine learning course project — Addis Ababa University
        </p>
      </div>

      {/* ── Objective + Findings ── */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card className="bg-card/40 backdrop-blur-md border-primary/20">
          <CardHeader className="pb-3">
            <CardTitle><SectionHeader icon={<Target size={15} />} title="Objective" /></CardTitle>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground leading-relaxed space-y-2">
            <p>Build and deploy a regression pipeline that accurately predicts annual medical insurance charges for US policyholders.</p>
            <p>Demonstrates the full ML lifecycle: data understanding → cleaning → feature engineering → model training → evaluation → web deployment.</p>
          </CardContent>
        </Card>

        <Card className="bg-card/40 backdrop-blur-md border-primary/20">
          <CardHeader className="pb-3">
            <CardTitle><SectionHeader icon={<Lightbulb size={15} />} title="Key Findings" /></CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2.5">
              {FINDINGS.map((f, i) => (
                <li key={i} className="flex gap-2.5 items-start text-sm text-muted-foreground">
                  {f.icon}<span>{f.text}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </div>

      {/* ── Dataset + Tech Stack ── */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card className="bg-card/40 backdrop-blur-md border-primary/20">
          <CardHeader className="pb-3">
            <CardTitle><SectionHeader icon={<Database size={15} />} title="Dataset" /></CardTitle>
          </CardHeader>
          <CardContent className="space-y-4 text-sm">
            <p className="text-muted-foreground">
              Source:{" "}
              <a href="https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset"
                target="_blank" rel="noopener noreferrer"
                className="text-primary inline-flex items-center gap-1 hover:underline underline-offset-4">
                Kaggle — Medical Insurance Cost <ExternalLink size={11} />
              </a>
            </p>
            <div className="rounded-lg border border-border/40 overflow-hidden">
              <table className="w-full text-xs">
                <thead className="bg-secondary/30">
                  <tr>
                    <th className="text-left px-3 py-2 font-semibold">Column</th>
                    <th className="text-left px-3 py-2 font-semibold">Description</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/30">
                  {DATASET_COLS.map(({ col, desc, isTarget }) => (
                    <tr key={col} className={isTarget ? "bg-primary/5" : ""}>
                      <td className="px-3 py-2">
                        <code className="text-primary font-mono">{col}</code>
                        {isTarget && <Badge className="ml-1.5 text-[9px] px-1 py-0 bg-primary/20 text-primary border-primary/30">target</Badge>}
                      </td>
                      <td className="px-3 py-2 text-muted-foreground">{desc}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <ul className="text-xs text-muted-foreground space-y-1 pl-4 list-disc">
              <li>1,338 policyholders (1,337 after deduplication)</li>
              <li>No missing values</li>
              <li>Charges right-skewed — skewness ≈ 1.52</li>
            </ul>
          </CardContent>
        </Card>

        <Card className="bg-card/40 backdrop-blur-md border-primary/20">
          <CardHeader className="pb-3">
            <CardTitle><SectionHeader icon={<Layers size={15} />} title="Tech Stack" /></CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {TECH_STACK.map(({ section, color, items }) => (
              <div key={section}>
                <p className={`text-[11px] font-semibold uppercase tracking-wider mb-2 ${color}`}>{section}</p>
                <div className="flex flex-wrap gap-1.5">
                  {items.map((item) => (
                    <Badge key={item} variant="secondary" className="text-[11px] px-2 py-0.5">{item}</Badge>
                  ))}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* ── Engineered Features + Key Files ── */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card className="bg-card/40 backdrop-blur-md border-primary/20">
          <CardHeader className="pb-3">
            <CardTitle><SectionHeader icon={<Wrench size={15} />} title="8 Engineered Features" /></CardTitle>
          </CardHeader>
          <CardContent className="divide-y divide-border/40">
            {FEATURES.map((f) => (
              <div key={f.name} className="flex items-center justify-between py-2 gap-4">
                <code className="text-primary text-[11px] font-mono font-semibold shrink-0">{f.name}</code>
                <span className="text-muted-foreground text-xs text-right">{f.desc}</span>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card className="bg-card/40 backdrop-blur-md border-primary/20">
          <CardHeader className="pb-3">
            <CardTitle><SectionHeader icon={<FolderOpen size={15} />} title="Key Files" /></CardTitle>
          </CardHeader>
          <CardContent className="divide-y divide-border/30">
            {KEY_FILES.map(({ path, role }) => (
              <div key={path} className="flex gap-3 items-start py-2">
                <code className="text-primary font-mono text-[11px] shrink-0 pt-0.5">{path}</code>
                <span className="text-muted-foreground text-xs leading-relaxed">{role}</span>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

    </div>
  );
}

import type { Metadata } from "next";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Target, Lightbulb, Wrench, Database, Layers,
  Terminal, FolderOpen, ExternalLink, Flame, Scale,
  TrendingUp, CheckCircle, BarChart2, Users2, Github,
} from "lucide-react";

export const metadata: Metadata = {
  title: "About",
  description:
    "Project overview, tech stack, dataset details, and quick-start commands for the MedPredict.ai insurance cost prediction pipeline.",
};

// ─── Data ────────────────────────────────────────────────────────────────────

const TECH_STACK: { section: string; color: string; items: string[] }[] = [
  {
    section: "Frontend",
    color: "text-blue-400",
    items: ["Next.js 16", "React 19", "Tailwind CSS v4", "shadcn/ui", "Framer Motion"],
  },
  {
    section: "Backend API",
    color: "text-green-400",
    items: ["Python 3.13", "FastAPI", "Uvicorn", "Pydantic v2"],
  },
  {
    section: "Machine Learning",
    color: "text-purple-400",
    items: ["Scikit-Learn", "XGBoost", "Pandas", "NumPy", "Joblib"],
  },
  {
    section: "Visualisation",
    color: "text-orange-400",
    items: ["Matplotlib", "Seaborn"],
  },
];

const FEATURES: { name: string; desc: string }[] = [
  { name: "age_group",    desc: "young / middle_age / senior" },
  { name: "bmi_category", desc: "underweight / normal / overweight / obese" },
  { name: "is_obese",     desc: "1 if BMI ≥ 30" },
  { name: "smoker_obese", desc: "1 if smoker AND obese (interaction term)" },
  { name: "age_bmi",      desc: "age × bmi / 1000" },
  { name: "has_children", desc: "1 if dependants > 0" },
  { name: "family_size",  desc: "individual / small_family / large_family" },
  { name: "age_smoker",   desc: "age × smoker flag" },
];

const DATASET_COLS: { col: string; desc: string; isTarget?: boolean }[] = [
  { col: "age",      desc: "Age of the primary beneficiary" },
  { col: "sex",      desc: "male / female" },
  { col: "bmi",      desc: "Body Mass Index (kg/m²)" },
  { col: "children", desc: "Number of dependants covered" },
  { col: "smoker",   desc: "yes / no" },
  { col: "region",   desc: "US residential region (4 values)" },
  { col: "charges",  desc: "Annual insurance cost in USD", isTarget: true },
];

const KEY_FILES: { path: string; role: string }[] = [
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

const TEAM: {
  name: string;
  role: string;
  contribution: string;
  github?: string;
  initials: string;
  color: string;
}[] = [
  {
    name: "Abdullah",
    role: "ML Engineer & Full-Stack Developer",
    contribution: "ML pipeline, FastAPI backend, Next.js dashboard, feature engineering, model evaluation",
    github: "https://github.com/Ab-xo",
    initials: "AB",
    color: "from-primary/30 to-primary/10",
  },
  // Add more team members below — copy the block above
  // {
  //   name: "Team Member",
  //   role: "Role",
  //   contribution: "What they worked on",
  //   initials: "TM",
  //   color: "from-blue-500/30 to-blue-500/10",
  // },
];

const FINDINGS: { icon: React.ReactNode; text: React.ReactNode }[] = [
  {
    icon: <Flame size={15} className="text-red-400 mt-0.5 shrink-0" />,
    text: (
      <>
        <strong>Smoker status</strong> is the dominant predictor — smokers pay{" "}
        <strong>3.8×</strong> more on average.
      </>
    ),
  },
  {
    icon: <Scale size={15} className="text-yellow-400 mt-0.5 shrink-0" />,
    text: (
      <>
        <strong>BMI ≥ 30 + smoking</strong> creates the highest-charge group via
        the <code className="bg-muted px-1 rounded text-[11px]">smoker_obese</code> interaction.
      </>
    ),
  },
  {
    icon: <TrendingUp size={15} className="text-blue-400 mt-0.5 shrink-0" />,
    text: (
      <>
        <strong>Age effect is nonlinear</strong> — charges spike sharply after 50.
      </>
    ),
  },
  {
    icon: <CheckCircle size={15} className="text-green-400 mt-0.5 shrink-0" />,
    text: (
      <>
        <strong>Linear models beat tree ensembles</strong> once interaction
        features are engineered. Lasso wins.
      </>
    ),
  },
  {
    icon: <BarChart2 size={15} className="text-purple-400 mt-0.5 shrink-0" />,
    text: (
      <>
        Charges are <strong>right-skewed</strong> (skewness ≈ 1.52) with real
        high-value outliers in the data.
      </>
    ),
  },
];

// ─── Section header helper ────────────────────────────────────────────────────

function SectionHeader({
  icon,
  title,
}: {
  icon: React.ReactNode;
  title: string;
}) {
  return (
    <div className="flex items-center gap-2.5">
      <div className="p-1.5 rounded-lg bg-primary/10 text-primary">{icon}</div>
      <span className="font-semibold text-base">{title}</span>
    </div>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function AboutPage() {
  return (
    <div className="space-y-8 max-w-5xl">
      {/* Page header */}
      <div className="space-y-1">
        <h1 className="text-3xl font-bold tracking-tight">About This Project</h1>
        <p className="text-muted-foreground">
          Architecture, dataset, tech stack, and key design decisions.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* ── LEFT ─────────────────────────────────────────────────── */}
        <div className="space-y-6">

          {/* Objective */}
          <Card className="bg-card/40 backdrop-blur-md border-primary/20">
            <CardHeader className="pb-3">
              <CardTitle>
                <SectionHeader icon={<Target size={15} />} title="Objective" />
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground leading-relaxed space-y-2.5">
              <p>
                Build and deploy a regression pipeline that accurately predicts
                annual medical insurance charges for US policyholders.
              </p>
              <p>
                Demonstrates a full ML lifecycle: data understanding → cleaning
                → feature engineering → model training → evaluation → modern web
                deployment.
              </p>
            </CardContent>
          </Card>

          {/* Key findings */}
          <Card className="bg-card/40 backdrop-blur-md border-primary/20">
            <CardHeader className="pb-3">
              <CardTitle>
                <SectionHeader icon={<Lightbulb size={15} />} title="Key Findings" />
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-3">
                {FINDINGS.map((f, i) => (
                  <li key={i} className="flex gap-2.5 items-start text-sm text-muted-foreground">
                    {f.icon}
                    <span>{f.text}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>

          {/* Engineered features */}
          <Card className="bg-card/40 backdrop-blur-md border-primary/20">
            <CardHeader className="pb-3">
              <CardTitle>
                <SectionHeader icon={<Wrench size={15} />} title="8 Engineered Features" />
              </CardTitle>
            </CardHeader>
            <CardContent className="divide-y divide-border/40">
              {FEATURES.map((f) => (
                <div
                  key={f.name}
                  className="flex items-center justify-between py-2 gap-4"
                >
                  <code className="text-primary text-[11px] font-mono font-semibold shrink-0">
                    {f.name}
                  </code>
                  <span className="text-muted-foreground text-xs text-right">
                    {f.desc}
                  </span>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* ── RIGHT ────────────────────────────────────────────────── */}
        <div className="space-y-6">

          {/* Dataset */}
          <Card className="bg-card/40 backdrop-blur-md border-primary/20">
            <CardHeader className="pb-3">
              <CardTitle>
                <SectionHeader icon={<Database size={15} />} title="Dataset" />
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <p className="text-muted-foreground">
                Source:{" "}
                <a
                  href="https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary inline-flex items-center gap-1 hover:underline underline-offset-4"
                >
                  Kaggle — Medical Insurance Cost
                  <ExternalLink size={11} />
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
                          {isTarget && (
                            <Badge className="ml-1.5 text-[9px] px-1 py-0 bg-primary/20 text-primary border-primary/30">
                              target
                            </Badge>
                          )}
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

          {/* Tech stack */}
          <Card className="bg-card/40 backdrop-blur-md border-primary/20">
            <CardHeader className="pb-3">
              <CardTitle>
                <SectionHeader icon={<Layers size={15} />} title="Tech Stack" />
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {TECH_STACK.map(({ section, color, items }) => (
                <div key={section}>
                  <p className={`text-[11px] font-semibold uppercase tracking-wider mb-2 ${color}`}>
                    {section}
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {items.map((item) => (
                      <Badge
                        key={item}
                        variant="secondary"
                        className="text-[11px] px-2 py-0.5"
                      >
                        {item}
                      </Badge>
                    ))}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Quick commands */}
          <Card className="bg-card/40 backdrop-blur-md border-primary/20">
            <CardHeader className="pb-3">
              <CardTitle>
                <SectionHeader icon={<Terminal size={15} />} title="Quick Commands" />
              </CardTitle>
            </CardHeader>
            <CardContent>
              <pre className="bg-muted/50 border border-border/40 rounded-lg px-4 py-3.5 text-[11px] font-mono leading-relaxed overflow-x-auto text-foreground/80">
{`# Train all 10 models
$env:PYTHONPATH = "."
python -m src.train

# Start FastAPI backend
uvicorn main:app --reload --port 8000

# Start Next.js dashboard
cd frontend && npm run dev

# Run tests
pytest tests/ -v`}
              </pre>
            </CardContent>
          </Card>

          {/* Key files */}
          <Card className="bg-card/40 backdrop-blur-md border-primary/20">
            <CardHeader className="pb-3">
              <CardTitle>
                <SectionHeader icon={<FolderOpen size={15} />} title="Key Files" />
              </CardTitle>
            </CardHeader>
            <CardContent className="divide-y divide-border/30">
              {KEY_FILES.map(({ path, role }) => (
                <div key={path} className="flex gap-3 items-start py-2">
                  <code className="text-primary font-mono text-[11px] shrink-0 pt-0.5">
                    {path}
                  </code>
                  <span className="text-muted-foreground text-xs leading-relaxed">
                    {role}
                  </span>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* ── Team ─────────────────────────────────────────────────── */}
      <Card className="bg-card/40 backdrop-blur-md border-primary/20">
        <CardHeader className="pb-3">
          <CardTitle>
            <SectionHeader icon={<Users2 size={15} />} title="Team" />
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {TEAM.map((member) => (
              <div
                key={member.name}
                className={`bg-gradient-to-br ${member.color} border border-primary/20 rounded-xl p-4 space-y-3`}
              >
                {/* Avatar + name */}
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-primary/20 border border-primary/30 flex items-center justify-center font-bold text-sm text-primary shrink-0">
                    {member.initials}
                  </div>
                  <div>
                    <p className="font-semibold text-sm">{member.name}</p>
                    <p className="text-xs text-primary">{member.role}</p>
                  </div>
                </div>

                {/* Contribution */}
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {member.contribution}
                </p>

                {/* GitHub link */}
                {member.github && (
                  <a
                    href={member.github}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-primary transition-colors"
                  >
                    <Github size={12} /> GitHub
                  </a>
                )}
              </div>
            ))}
          </div>
          <p className="text-xs text-muted-foreground mt-4 text-center">
            Built as part of a machine learning course project.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

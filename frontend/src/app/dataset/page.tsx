"use client";

import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Activity, Database, TrendingUp, Users } from "lucide-react";
import { motion } from "framer-motion";

const item = { hidden: { opacity: 0, y: 16 }, show: { opacity: 1, y: 0 } };
const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.1 } },
};

export default function DatasetPage() {
  const [summary, setSummary] = useState<any>(null);
  const [sample,  setSample]  = useState<any[]>([]);
  const [isError, setIsError] = useState(false);

  useEffect(() => {
    Promise.all([
      fetch("http://localhost:8000/api/dataset/summary"),
      fetch("http://localhost:8000/api/dataset/sample"),
    ])
      .then(async ([rs, rsp]) => {
        if (!rs.ok || !rsp.ok) throw new Error("Backend unavailable");
        setSummary(await rs.json());
        setSample(await rsp.json());
      })
      .catch(() => setIsError(true));
  }, []);

  if (isError)
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] text-center space-y-4">
        <Activity className="h-12 w-12 text-destructive opacity-50" />
        <h2 className="text-xl font-bold">Cannot connect to Backend</h2>
        <p className="text-muted-foreground">
          Ensure FastAPI server is running on port 8000.
        </p>
        <code className="bg-muted px-3 py-2 rounded-lg text-sm font-mono">
          uvicorn app.main:app --reload --port 8000
        </code>
      </div>
    );

  if (!summary)
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
      </div>
    );

  const colHeaders = sample.length ? Object.keys(sample[0]) : [];

  return (
    <div className="space-y-8 max-w-7xl">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-1">
          📊 Dataset Overview
        </h1>
        <p className="text-muted-foreground">
          Medical insurance dataset — structure, statistics, and target
          distribution.
        </p>
      </div>

      {/* KPI row */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="grid grid-cols-2 md:grid-cols-4 gap-4"
      >
        {[
          { label: "Rows",           value: summary.rows,       icon: Users,    color: "text-primary" },
          { label: "Columns",        value: summary.columns,    icon: Database, color: "text-blue-400" },
          { label: "Missing Values", value: summary.missing,    icon: Activity, color: "text-green-400" },
          { label: "Duplicates",     value: summary.duplicates, icon: TrendingUp, color: "text-yellow-400" },
        ].map(({ label, value, icon: Icon, color }) => (
          <motion.div key={label} variants={item}>
            <Card className="bg-card/40 backdrop-blur-md border-primary/20">
              <CardHeader className="flex flex-row items-center justify-between pb-1">
                <CardTitle className="text-xs text-muted-foreground font-medium">
                  {label}
                </CardTitle>
                <Icon className={`h-4 w-4 ${color}`} />
              </CardHeader>
              <CardContent>
                <p className="text-3xl font-bold">{value}</p>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </motion.div>

      {/* Sample data table */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <Card className="bg-card/40 backdrop-blur-md border-primary/20 overflow-hidden">
          <CardHeader>
            <CardTitle>Raw Data — First 20 Rows</CardTitle>
            <CardDescription>
              The 6 raw input features and the target{" "}
              <code className="bg-muted px-1 rounded text-xs">charges</code>
            </CardDescription>
          </CardHeader>
          <CardContent className="p-0">
            <div className="overflow-x-auto max-h-80">
              <Table>
                <TableHeader className="bg-secondary/50 sticky top-0 backdrop-blur-md">
                  <TableRow>
                    {colHeaders.map((k) => (
                      <TableHead key={k} className="font-semibold capitalize whitespace-nowrap">
                        {k}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sample.map((row, i) => (
                    <TableRow key={i} className="hover:bg-secondary/20">
                      {colHeaders.map((k) => (
                        <TableCell key={k} className="whitespace-nowrap">
                          {k === "charges"
                            ? `$${Number(row[k]).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                            : String(row[k])}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* Target analysis + Smoker breakdown */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="grid md:grid-cols-2 gap-6"
      >
        {/* Target stats */}
        <Card className="bg-card/40 backdrop-blur-md border-primary/20">
          <CardHeader>
            <CardTitle>🎯 Target Variable — Charges</CardTitle>
            <CardDescription>Annual insurance cost in USD</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-0 divide-y divide-border/50">
              {[
                { label: "Min",      value: `$${summary.target.min.toLocaleString()}` },
                { label: "Max",      value: `$${summary.target.max.toLocaleString()}` },
                { label: "Mean",     value: `$${summary.target.mean.toLocaleString()}` },
                { label: "Median",   value: `$${summary.target.median.toLocaleString()}` },
                { label: "Std Dev",  value: `$${summary.target.std.toLocaleString()}` },
                { label: "Q1",       value: `$${summary.target.q1.toLocaleString()}` },
                { label: "Q3",       value: `$${summary.target.q3.toLocaleString()}` },
                { label: "Skewness", value: summary.target.skew },
                { label: "Outliers", value: summary.target.outliers },
              ].map(({ label, value }) => (
                <div
                  key={label}
                  className="flex justify-between items-center py-2.5"
                >
                  <span className="text-sm text-muted-foreground">{label}</span>
                  <span className="text-sm font-semibold font-mono">{value}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <div className="space-y-6">
          {/* Smoker breakdown */}
          <Card className="bg-card/40 backdrop-blur-md border-primary/20">
            <CardHeader>
              <CardTitle>🚬 Charges by Smoker Status</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {Object.entries(summary.smoker).map(
                ([status, stats]: [string, any]) => (
                  <div key={status}>
                    <p className="font-semibold mb-2 text-sm capitalize">
                      {status === "yes" ? "🚬 Smoker" : "✅ Non-Smoker"}
                    </p>
                    <div className="grid grid-cols-3 gap-2 text-center text-sm">
                      {[
                        { label: "Mean",   value: `$${stats.mean.toLocaleString()}` },
                        { label: "Median", value: `$${stats.median.toLocaleString()}` },
                        { label: "Count",  value: stats.count },
                      ].map(({ label, value }) => (
                        <div
                          key={label}
                          className="bg-secondary/30 rounded-lg p-2 border border-border/40"
                        >
                          <p className="text-muted-foreground text-xs mb-0.5">
                            {label}
                          </p>
                          <p className="font-bold">{value}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )
              )}
            </CardContent>
          </Card>

          {/* Region breakdown */}
          <Card className="bg-card/40 backdrop-blur-md border-primary/20">
            <CardHeader>
              <CardTitle>🗺️ Charges by Region</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {summary.regions.map((r: any) => (
                <div key={r.name} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-primary" />
                    <span className="capitalize text-sm font-medium">{r.name}</span>
                  </div>
                  <div className="text-right text-sm">
                    <span className="font-semibold">
                      ${r.mean.toLocaleString()}
                    </span>
                    <span className="text-muted-foreground text-xs ml-2">
                      med ${r.median.toLocaleString()}
                    </span>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </motion.div>
    </div>
  );
}

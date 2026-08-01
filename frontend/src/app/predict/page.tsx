"use client";

import { useState } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import * as z from "zod";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  ShieldCheck,
  AlertCircle,
  AlertTriangle,
  TrendingUp,
  Users,
  RefreshCw,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CardFooter,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";

const schema = z.object({
  age:      z.coerce.number().min(0).max(120),
  sex:      z.string().min(1, "Required"),
  bmi:      z.coerce.number().min(10).max(80),
  children: z.coerce.number().min(0).max(10),
  smoker:   z.string().min(1, "Required"),
  region:   z.string().min(1, "Required"),
});

type FormValues = z.infer<typeof schema>;

const EXAMPLE_PROFILES: { label: string; values: FormValues }[] = [
  { label: "Young healthy male",        values: { age: 22, sex: "male",   bmi: 22.0, children: 0, smoker: "no",  region: "northeast" } },
  { label: "Middle-age female, 2 kids", values: { age: 35, sex: "female", bmi: 28.5, children: 2, smoker: "no",  region: "northwest" } },
  { label: "Obese smoker male",         values: { age: 45, sex: "male",   bmi: 33.0, children: 1, smoker: "yes", region: "southeast" } },
  { label: "Senior obese smoker female",values: { age: 60, sex: "female", bmi: 35.0, children: 0, smoker: "yes", region: "southwest" } },
];

export default function PredictPage() {
  const [prediction, setPrediction] = useState<any>(null);
  const [isLoading,  setIsLoading]  = useState(false);
  const [error,      setError]      = useState<string | null>(null);

  const form = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: { age: 35, sex: "male", bmi: 28.5, children: 0, smoker: "no", region: "northwest" },
  });

  async function onSubmit(values: FormValues) {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch("http://localhost:8000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(values),
      });
      if (!res.ok) throw new Error((await res.json()).detail ?? "Prediction failed");
      setPrediction(await res.json());
    } catch (e: any) {
      setError(e.message ?? "Unexpected error");
    } finally {
      setIsLoading(false);
    }
  }

  function applyProfile(p: FormValues) {
    Object.entries(p).forEach(([k, v]) =>
      form.setValue(k as keyof FormValues, v as any)
    );
  }

  const Field = ({
    label, error, children,
  }: {
    label: string; error?: string; children: React.ReactNode;
  }) => (
    <div className="space-y-1.5">
      <label className="text-sm font-medium leading-none">{label}</label>
      {children}
      {error && (
        <p className="text-xs text-destructive flex items-center gap-1">
          <AlertCircle size={11} /> {error}
        </p>
      )}
    </div>
  );

  return (
    <div className="space-y-8 max-w-5xl">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-1">
          💰 Predict Insurance Charges
        </h1>
        <p className="text-muted-foreground">
          Enter policyholder details to estimate annual charges using the
          trained <strong>Lasso Regression</strong> model.
        </p>
      </div>

      {/* Example profiles */}
      <div className="flex flex-wrap gap-2">
        <span className="text-xs text-muted-foreground self-center">Quick profiles:</span>
        {EXAMPLE_PROFILES.map((p) => (
          <button
            key={p.label}
            type="button"
            onClick={() => applyProfile(p.values)}
            className="text-xs px-3 py-1.5 rounded-lg bg-secondary/40 hover:bg-secondary/70 border border-border/50 hover:border-primary/40 transition-all"
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="grid md:grid-cols-2 gap-8 items-start">
        {/* ── Form ── */}
        <Card className="bg-card/40 backdrop-blur-md shadow-lg border-primary/20">
          <CardHeader>
            <CardTitle>Policyholder Profile</CardTitle>
            <CardDescription>
              Fill in the 6 raw input features.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-5">
              <div className="grid grid-cols-2 gap-4">
                <Field label="Age" error={form.formState.errors.age?.message}>
                  <Input type="number" {...form.register("age")} />
                </Field>
                <Field label="Sex" error={form.formState.errors.sex?.message}>
                  <Select
                    onValueChange={(v) => form.setValue("sex", v as string)}
                    defaultValue={form.getValues("sex") ?? ""}
                  >
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="male">Male</SelectItem>
                      <SelectItem value="female">Female</SelectItem>
                    </SelectContent>
                  </Select>
                </Field>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <Field label="BMI (kg/m²)" error={form.formState.errors.bmi?.message}>
                  <Input type="number" step="0.1" {...form.register("bmi")} />
                </Field>
                <Field label="Dependents" error={form.formState.errors.children?.message}>
                  <Input type="number" {...form.register("children")} />
                </Field>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <Field label="Smoker" error={form.formState.errors.smoker?.message}>
                  <Select
                    onValueChange={(v) => form.setValue("smoker", v as string)}
                    defaultValue={form.getValues("smoker") ?? ""}
                  >
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="no">No</SelectItem>
                      <SelectItem value="yes">Yes</SelectItem>
                    </SelectContent>
                  </Select>
                </Field>
                <Field label="Region" error={form.formState.errors.region?.message}>
                  <Select
                    onValueChange={(v) => form.setValue("region", v as string)}
                    defaultValue={form.getValues("region") ?? ""}
                  >
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="northeast">Northeast</SelectItem>
                      <SelectItem value="northwest">Northwest</SelectItem>
                      <SelectItem value="southeast">Southeast</SelectItem>
                      <SelectItem value="southwest">Southwest</SelectItem>
                    </SelectContent>
                  </Select>
                </Field>
              </div>

              <Button
                type="submit"
                className="w-full font-semibold"
                disabled={isLoading}
              >
                {isLoading ? (
                  <span className="flex items-center gap-2">
                    <Activity className="animate-spin" size={16} />
                    Calculating…
                  </span>
                ) : (
                  "Calculate Premium"
                )}
              </Button>

              {error && (
                <p className="text-sm text-destructive flex items-center gap-2">
                  <AlertCircle size={15} /> {error}
                </p>
              )}
            </form>
          </CardContent>
        </Card>

        {/* ── Result panel ── */}
        <div className="relative min-h-[340px]">
          <AnimatePresence mode="wait">
            {prediction ? (
              <motion.div
                key="result"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ type: "spring", stiffness: 200, damping: 20 }}
                className="space-y-4"
              >
                {/* Main result card */}
                <Card className="bg-gradient-to-br from-primary/20 via-card to-card border-primary/40 shadow-2xl overflow-hidden relative">
                  <div className="absolute top-0 right-0 p-4 opacity-5">
                    <ShieldCheck size={130} />
                  </div>
                  <CardHeader>
                    <CardTitle className="text-primary flex items-center gap-2">
                      <ShieldCheck size={18} /> Estimated Annual Premium
                    </CardTitle>
                    <CardDescription>
                      Model: {prediction.model_used}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-5">
                    <div>
                      <p className="text-xs text-muted-foreground mb-1 uppercase tracking-wider">
                        Predicted Charges
                      </p>
                      <h2 className="text-5xl font-black tracking-tighter">
                        {prediction.formatted}
                      </h2>
                    </div>

                    {/* Context benchmarks */}
                    {prediction.context && (
                      <div className="grid grid-cols-3 gap-2 text-center text-xs">
                        {[
                          {
                            label: "Your Prediction",
                            value: `$${prediction.predicted_charges.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
                            color: "text-foreground",
                          },
                          {
                            label: "Dataset Avg",
                            value: `$${prediction.context.overall_mean.toLocaleString()}`,
                            color: "text-muted-foreground",
                          },
                          prediction.input.smoker === "yes"
                            ? {
                                label: "Avg Smoker",
                                value: `$${prediction.context.smoker_mean.toLocaleString()}`,
                                color: "text-destructive",
                              }
                            : {
                                label: "Avg Non-Smoker",
                                value: `$${prediction.context.nonsmoker_mean.toLocaleString()}`,
                                color: "text-green-400",
                              },
                        ].map(({ label, value, color }) => (
                          <div
                            key={label}
                            className="bg-secondary/20 rounded-lg p-2 border border-border/40"
                          >
                            <p className="text-muted-foreground mb-0.5">{label}</p>
                            <p className={`font-bold ${color}`}>{value}</p>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Risk flags */}
                    {prediction.risk_flags?.length > 0 && (
                      <div>
                        <p className="text-xs font-semibold mb-2 flex items-center gap-1.5">
                          <AlertTriangle size={13} className="text-yellow-400" />
                          Risk Factors
                        </p>
                        <div className="flex flex-col gap-1.5">
                          {prediction.risk_flags.map((flag: string, i: number) => (
                            <Badge
                              key={i}
                              variant="destructive"
                              className="bg-destructive/15 text-destructive-foreground border border-destructive/30 justify-start text-xs font-normal py-1.5 px-2.5 h-auto"
                            >
                              {flag}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </CardContent>
                  <CardFooter className="bg-secondary/20 border-t border-border/50 py-3 justify-between text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <Users size={12} /> {prediction.context?.overall_mean
                        ? `Dataset mean: $${prediction.context.overall_mean.toLocaleString()}`
                        : ""}
                    </span>
                    <span className="flex items-center gap-1">
                      <TrendingUp size={12} /> R² ≈ 0.87
                    </span>
                  </CardFooter>
                </Card>

                {/* Input summary */}
                <Card className="bg-card/40 backdrop-blur-md border-border/40">
                  <CardHeader className="py-3 border-b border-border/40">
                    <CardTitle className="text-sm">Input Summary</CardTitle>
                  </CardHeader>
                  <CardContent className="py-3">
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      {Object.entries(prediction.input).map(([k, v]) => (
                        <div key={k} className="bg-secondary/20 rounded-lg p-2">
                          <p className="text-muted-foreground capitalize mb-0.5">
                            {k}
                          </p>
                          <p className="font-semibold capitalize">{String(v)}</p>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Button
                  variant="outline"
                  size="sm"
                  className="w-full"
                  onClick={() => { setPrediction(null); form.reset(); }}
                >
                  <RefreshCw size={14} className="mr-2" /> Reset & Predict Again
                </Button>
              </motion.div>
            ) : (
              <motion.div
                key="empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="h-full flex items-center justify-center p-8 border-2 border-dashed border-border/50 rounded-2xl text-muted-foreground text-center bg-card/10 backdrop-blur-sm min-h-[340px]"
              >
                <div>
                  <Activity size={44} className="mx-auto mb-4 opacity-20" />
                  <p className="text-sm">
                    Fill out the form and click
                    <br />
                    <strong className="text-foreground">Calculate Premium</strong>
                    <br />
                    to see the prediction.
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
